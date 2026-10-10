"""Zesyde Decision Model Architecture.

Backbone: ModernBERT-large (bidirectional encoder, ~395M params).
Decision Head: Option-marker scoring via pure MLP head on candidate [MASK] token positions.
Act/Escalate Head: Binary logit head indicating whether the prediction should escalate to human/teacher.
"""
from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

class SemanticTransformerHead(nn.Module):
    """Transformer-based option scoring head with candidate cross-attention.

    Processes candidate representations with self-attention across options,
    allowing candidate options to compete against one another before scoring.
    """
    def __init__(self, in_features: int, num_layers: int = 1, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=in_features,
            nhead=num_heads,
            dim_feedforward=in_features * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.scorer = DecisionHead(in_features, num_layers=2, dropout=dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            x = features.unsqueeze(0)
            out = self.transformer(x).squeeze(0)
            return self.scorer(out)
        out = self.transformer(features)
        return self.scorer(out)

class DecisionHead(nn.Module):
    """Context-conditioned option scorer with candidate-prompt interaction.
    
    Accepts candidate representations (and fused context features) to predict
    calibrated scalar compatibility scores for each candidate option.
    """
    def __init__(self, in_features: int, intermediate_size: Optional[int] = None, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        inter_dim = intermediate_size if intermediate_size else max(256, in_features // 2)
        layers = []
        curr_dim = in_features
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(curr_dim, inter_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(inter_dim))
            layers.append(nn.Dropout(dropout))
            curr_dim = inter_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
        
ZesydeMLPHead = DecisionHead

class ZesydeModel(nn.Module):
    """Zesyde Non-Autoregressive Decision Model.

    Encodes:
    [CLS] context [SEP] question [SEP] [MASK] Option 1 [MASK] Option 2 ... [SEP]

    Produces:
    - per-option logits for each question with context-conditioned candidate scoring
    - act/escalate logit (for low-confidence routing / escalation)
    """
    def __init__(
        self,
        backbone_name_or_path: str = 'jhu-clsp/mmBERT-base',
        head_layers: int = 2,
        head_num_heads: Optional[int] = None,
        head_dropout: float = 0.1,
        use_peft: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[Union[List[str], str]] = None,
        use_context_fusion: bool = True,
    ):
        super().__init__()
        self.use_peft = use_peft
        self.use_context_fusion = use_context_fusion
        self.backbone = AutoModel.from_pretrained(backbone_name_or_path)
        hidden_size = self.backbone.config.hidden_size

        if use_peft:
            from peft import LoraConfig, get_peft_model
            targets = lora_target_modules or "all-linear"
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=targets,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
                
        feature_dim = hidden_size * 3 if use_context_fusion else hidden_size
        if head_num_heads is not None:
            self.option_head = SemanticTransformerHead(
                in_features=feature_dim,
                num_layers=1,
                num_heads=head_num_heads,
                dropout=head_dropout
            )
        else:
            self.option_head = DecisionHead(
                in_features=feature_dim,
                num_layers=head_layers,
                dropout=head_dropout
            )

        self.act_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def merge_peft(self):
        """Merges LoRA adapter weights into the base backbone for export or standalone inference."""
        if getattr(self, 'use_peft', False) and hasattr(self.backbone, 'merge_and_unload'):
            self.backbone = self.backbone.merge_and_unload()
            self.use_peft = False
        return self

    def print_trainable_parameters(self):
        """Prints the count of trainable parameters across backbone and heads."""
        if hasattr(self.backbone, "print_trainable_parameters"):
            print("Backbone PEFT parameters:")
            self.backbone.print_trainable_parameters()
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total model parameters: {total_params:,} | Trainable: {trainable_params:,} ({100 * trainable_params / max(1, total_params):.2f}%)")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, marker_indices: List[List[int]]) -> Dict[str, Any]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        device = last_hidden.device
        B, S, H = last_hidden.shape
        
        # Vectorized mean-pooling of option tokens
        M = max(len(idx) for idx in marker_indices) if marker_indices else 0
        cls_hidden = last_hidden[:, 0, :]
        act_logits = self.act_head(cls_hidden).squeeze(-1)

        if M == 0:
            return {
                'logits': [],
                'probs': [],
                'act_logits': act_logits
            }

        marker_positions = torch.zeros((B, M), dtype=torch.long, device=device)
        next_markers = torch.zeros((B, M), dtype=torch.long, device=device)
        seq_lens = attention_mask.sum(dim=-1)

        for b, indices in enumerate(marker_indices):
            k = len(indices)
            marker_positions[b, :k] = torch.tensor(indices, device=device)
            for i in range(k - 1):
                next_markers[b, i] = indices[i+1]
            if k > 0:
                next_markers[b, k - 1] = max(indices[-1] + 1, int(seq_lens[b].item()) - 1)
                
        seq_grid = torch.arange(S, device=device).view(1, 1, S)
        start_mask = seq_grid >= marker_positions.unsqueeze(-1)
        end_mask = seq_grid < next_markers.unsqueeze(-1)
        
        option_mask = (start_mask & end_mask).float() * attention_mask.unsqueeze(1).float()
        mask_sums = option_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        
        opt_hidden_all = torch.bmm(option_mask, last_hidden) / mask_sums # (B, M, H)

        if self.use_context_fusion:
            cls_expanded = cls_hidden.unsqueeze(1).expand(-1, M, -1)
            features = torch.cat([opt_hidden_all, cls_expanded, opt_hidden_all * cls_expanded], dim=-1) # (B, M, 3*H)
        else:
            features = opt_hidden_all
            
        opt_scores_all = self.option_head(features).squeeze(-1) # (B, M)
        
        batch_logits = []
        batch_probs = []
        for b, indices in enumerate(marker_indices):
            opt_scores = opt_scores_all[b, :len(indices)]
            batch_logits.append(opt_scores)
            batch_probs.append(torch.softmax(opt_scores, dim=-1))
        
        return {
            'logits': batch_logits,
            'probs': batch_probs,
            'act_logits': act_logits
        }

