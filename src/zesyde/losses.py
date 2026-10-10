"""Loss functions for Zesyde decision training.

Supports:
- Soft cross-entropy (for teacher distillation probability distributions)
- Label-smoothed cross-entropy (for direct supervised training without teacher)
- Ranked Probability Score (RPS) / Ordinal loss for `score` primitives
- Binary Cross Entropy for act/escalate head
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def label_smoothed_cross_entropy(logits: torch.Tensor, target: int, smoothing: float = 0.1) -> torch.Tensor:
    k = logits.size(0)
    if k < 2:
        raise ValueError(f"Expected at least 2 options for cross entropy, got {k}")
        
    log_probs = F.log_softmax(logits, dim=-1)
    smooth_target = torch.full_like(logits, smoothing / (k - 1))
    smooth_target[target] = 1.0 - smoothing
    
    loss = -torch.sum(smooth_target * log_probs)
    return loss

def soft_distillation_loss(logits: torch.Tensor, teacher_probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    scaled_logits = logits / temperature
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    loss = -torch.sum(teacher_probs * log_probs) * (temperature ** 2)
    return loss

def ranked_probability_loss(logits: torch.Tensor, target: int) -> torch.Tensor:
    k = logits.size(0)
    if k < 2:
        return torch.tensor(0.0, device=logits.device)
    probs = F.softmax(logits, dim=-1)
    cdf_pred = torch.cumsum(probs, dim=-1)[:-1]
    target_one_hot = torch.zeros_like(probs)
    target_one_hot[target] = 1.0
    cdf_true = torch.cumsum(target_one_hot, dim=-1)[:-1]
    rps = torch.sum((cdf_pred - cdf_true) ** 2) / (k - 1)
    return rps

class RankedProbabilityLoss(nn.Module):
    """Batched Ranked Probability Score (RPS) loss for ordinal `score` questions.

    Module-style wrapper around :func:`ranked_probability_loss` so it can be
    used as a criterion::

        criterion = RankedProbabilityLoss()
        loss = criterion(logits, targets)  # logits [B, K], targets [B]

    Args:
        reduction: One of ``"mean"`` (default), ``"sum"``, or ``"none"``.
            ``"none"`` returns a per-example tensor of shape ``[B]``.

    Forward Args:
        logits: Raw scores of shape ``[K]`` or ``[B, K]``.
        targets: Integer ground-truth indices of shape ``[]`` or ``[B]``.
    """
    _VALID_REDUCTIONS = ('mean', 'sum', 'none')
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(f"reduction must be one of {self._VALID_REDUCTIONS}, got {reduction!r}")
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if logits.dim() != 2:
            raise ValueError(f"logits must have shape [K] or [B, K], got {tuple(logits.shape)}")
            
        targets = torch.as_tensor(targets, dtype=torch.long, device=logits.device)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        if targets.dim() != 1 or targets.size(0) != logits.size(0):
            raise ValueError(f"targets shape {tuple(targets.shape)} does not match batch size {logits.size(0)}")
            
        per_example = torch.stack([
            ranked_probability_loss(logits[i], int(targets[i].item()))
            for i in range(logits.size(0))
        ])
        
        if self.reduction == 'mean':
            return per_example.mean()
        if self.reduction == 'sum':
            return per_example.sum()
        return per_example

def compute_batch_loss(
    batch_logits: List[torch.Tensor],
    targets: torch.Tensor,
    question_types: List[str],
    teacher_probs_list: Optional[List[Optional[torch.Tensor]]] = None,
    smoothing: float = 0.1,
    act_logits: Optional[torch.Tensor] = None,
    act_targets: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_decision_loss = torch.tensor(0.0, device=batch_logits[0].device)
    batch_size = len(batch_logits)
    
    for i in range(batch_size):
        logits = batch_logits[i]
        tgt = targets[i].item()
        q_type = question_types[i]
        t_probs = teacher_probs_list[i] if teacher_probs_list is not None else None
        
        if t_probs is not None:
            ex_loss = soft_distillation_loss(logits, t_probs)
        elif q_type == 'score':
            ce_loss = label_smoothed_cross_entropy(logits, tgt, smoothing=smoothing)
            rps_loss = ranked_probability_loss(logits, tgt)
            ex_loss = ce_loss + 0.5 * rps_loss
        else:
            ex_loss = label_smoothed_cross_entropy(logits, tgt, smoothing=smoothing)
            
        total_decision_loss += ex_loss
        
    avg_decision_loss = total_decision_loss / batch_size
    act_loss = torch.tensor(0.0, device=batch_logits[0].device)
    if act_logits is not None and act_targets is not None:
        act_loss = F.binary_cross_entropy_with_logits(act_logits, act_targets.float())
        
    return avg_decision_loss + act_loss, avg_decision_loss, act_loss
