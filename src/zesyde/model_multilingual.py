"""Zesyde Multilingual Decision Model Architecture.

Backbone: jhu-clsp/mmBERT (Massively Multilingual ModernBERT, trained on 1,800+ languages).
Decision Head: Option-marker scoring with candidate token pooling, context fusion, and optional cross-attention.
Act/Escalate Head: Calibrated binary routing logit from the [CLS] representation.
"""
from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
from zesyde.model import ZesydeModel, DecisionHead, SemanticTransformerHead


class ZesydeMultilingualModel(ZesydeModel):
    """Zesyde Multilingual Non-Autoregressive Decision Model.

    Extends ZesydeModel with default multilingual backbone (mmBERT-base, 1,800+ languages),
    vectorized option pooling, prompt-context fusion, and optional cross-option competition head.
    """
    def __init__(
        self,
        backbone_name_or_path: str = 'jhu-clsp/mmBERT-base',
        head_num_layers: Optional[int] = None,
        head_num_heads: Optional[int] = None,
        head_dropout: float = 0.1,
        head_layers: int = 2,
        use_peft: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[Union[List[str], str]] = None,
        use_context_fusion: bool = True,
    ):
        super().__init__(
            backbone_name_or_path=backbone_name_or_path,
            head_layers=head_num_layers if head_num_layers is not None else head_layers,
            head_num_heads=head_num_heads,
            head_dropout=head_dropout,
            use_peft=use_peft,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            use_context_fusion=use_context_fusion,
        )


