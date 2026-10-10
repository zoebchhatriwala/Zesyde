"""Reinforcement Learning for Calibrated Decisions (RLCD).

Implements Phase 2 of the Zesyde training procedure:
- Strictly proper scoring rule rewards (Spherical, Log, and Brier score for choice/noul, RPS for score).
- Policy exploration via Gaussian noise injected into logits.
- REINFORCE with Group Relative Policy Optimization (GRPO-style) baseline.
- Reference policy KL penalty to prevent drift and catastrophic forgetting.
- Heavy penalty on extreme confidence for incorrect predictions to prevent logit explosion.
- Logit regularization to constrain logits to stable magnitudes.
"""
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def spherical_score(probs: torch.Tensor, target: int, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.norm(probs, p=2, dim=-1, keepdim=False) + eps
    return probs[target] / norm

def log_score(probs: torch.Tensor, target: int, eps: float = 1e-7) -> torch.Tensor:
    p_clamped = torch.clamp(probs[target], min=eps, max=1.0)
    return torch.log(p_clamped)

def brier_score(probs: torch.Tensor, target: int) -> torch.Tensor:
    """Quadratic (Brier) strictly proper scoring rule: 1 - ||probs - target||_2^2.
    
    Ranges from 1.0 (certain correct) to -1.0 (certain incorrect).
    """
    target_one_hot = torch.zeros_like(probs)
    target_one_hot[target] = 1.0
    return 1.0 - torch.sum((probs - target_one_hot) ** 2)

def rps_score(probs: torch.Tensor, target: int, overconf_penalty_weight: float = 3.0) -> torch.Tensor:
    k = probs.size(0)
    if k < 2:
        return torch.tensor(1.0, device=probs.device)
    cdf_pred = torch.cumsum(probs, dim=-1)[:-1]
    target_one_hot = torch.zeros_like(probs)
    target_one_hot[target] = 1.0
    cdf_true = torch.cumsum(target_one_hot, dim=-1)[:-1]
    rps = torch.sum((cdf_pred - cdf_true) ** 2) / (k - 1)
    score = 1.0 - 2.0 * rps
    pred = int(torch.argmax(probs).item())
    if pred != target:
        conf = probs[pred]
        dist = abs(float(pred - target)) / (k - 1)
        score = score - overconf_penalty_weight * (conf ** 2) * (1.0 + dist)
    return score

def compute_reward(
    probs: torch.Tensor,
    target: int,
    question_type: str = 'choice',
    log_weight: float = 0.5,
    spherical_weight: float = 1.0,
    brier_weight: float = 1.0,
    overconf_penalty_weight: float = 3.0,
) -> torch.Tensor:
    """Computes strictly proper scoring rule reward.
    
    Heavily penalizes extreme confidence on incorrect predictions to prevent
    the model from gaming the objective by pushing logits to infinity.
    """
    if question_type == 'score':
        return rps_score(probs, target, overconf_penalty_weight=overconf_penalty_weight)
    
    s_score = spherical_score(probs, target)
    l_score = log_score(probs, target)
    b_score = brier_score(probs, target)
    
    reward = spherical_weight * s_score + log_weight * l_score + brier_weight * b_score
    
    # Heavily penalize extreme confidence on incorrect predictions
    pred = int(torch.argmax(probs).item())
    if pred != target:
        conf = probs[pred]
        # Quadratic penalty that escalates sharply as confidence in wrong answer -> 1.0
        reward = reward - overconf_penalty_weight * (conf ** 2)
        
    return reward

def rlcd_step_loss(
    base_logits: torch.Tensor,
    target: int,
    question_type: str = 'choice',
    ref_logits: Optional[torch.Tensor] = None,
    group_size: int = 4,
    noise_std: float = 0.5,
    kl_coef: float = 0.1,
    logit_reg: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    k = base_logits.size(0)
    device = base_logits.device
    
    policy_probs = F.softmax(base_logits, dim=-1)
    policy_log_probs = F.log_softmax(base_logits, dim=-1)
    
    noise = torch.randn(group_size, k, device=device) * noise_std
    sampled_logits = base_logits.detach().unsqueeze(0) + noise
    sampled_probs = F.softmax(sampled_logits, dim=-1)
    
    rewards = torch.zeros(group_size, device=device)
    for g in range(group_size):
        rewards[g] = compute_reward(sampled_probs[g], target, question_type)
        
    mean_reward = torch.mean(rewards)
    std_reward = torch.std(rewards)
    if std_reward < 1e-6:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        
    pg_loss = torch.tensor(0.0, device=device)
    for g in range(group_size):
        weighted_log_prob = torch.sum(sampled_probs[g] * policy_log_probs)
        pg_loss -= advantages[g] * weighted_log_prob
    pg_loss /= group_size
    
    # KL penalty against frozen baseline model to prevent distribution drift & catastrophic forgetting
    kl_loss = torch.tensor(0.0, device=device)
    kl_div_val = torch.tensor(0.0, device=device)
    if ref_logits is not None:
        if ref_logits.dim() == 1 and ref_logits.size(0) == k:
            if torch.allclose(ref_logits.sum(), torch.tensor(1.0, device=device), atol=1e-3) and (ref_logits >= 0).all():
                ref_log_probs = torch.log(ref_logits.clamp(min=1e-8))
            else:
                ref_log_probs = F.log_softmax(ref_logits.detach(), dim=-1)
            # KL divergence D_KL(policy || ref) = sum(p * (log p - log p_ref))
            kl_div_val = torch.sum(policy_probs * (policy_log_probs - ref_log_probs))
            kl_loss = kl_coef * torch.clamp(kl_div_val, min=0.0)
            
    # L2 logit regularization to directly prevent logit explosion
    logit_loss = logit_reg * torch.mean(base_logits ** 2)
    
    total_loss = pg_loss + kl_loss + logit_loss
    
    stats = {
        'mean_reward': float(mean_reward.item()),
        'max_reward': float(torch.max(rewards).item()),
        'min_reward': float(torch.min(rewards).item()),
        'pg_loss': float(pg_loss.item()),
        'kl_loss': float(kl_loss.item()),
        'kl_div': float(kl_div_val.item()),
        'logit_loss': float(logit_loss.item()),
        'max_logit': float(torch.max(base_logits).item()),
        'min_logit': float(torch.min(base_logits).item()),
    }
    
    return total_loss, stats
