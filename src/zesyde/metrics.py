"""Evaluation metrics calculation for Zesyde benchmarks.

Calculates:
- Decision Accuracy
- Ordinal Mean Absolute Error (MAE)
- Expected Calibration Error (ECE)
- Brier Score
"""
from typing import List, Dict, Any, Tuple
import numpy as np

def compute_metrics(
    predictions: List[int],
    targets: List[int],
    probabilities: List[np.ndarray],
    question_type: str = 'choice',
    num_bins: int = 10
) -> Dict[str, float]:
    preds = np.array(predictions)
    tgts = np.array(targets)
    n = len(preds)
    
    if n == 0:
        return {}
        
    acc = float(np.mean(preds == tgts))
    metrics = {'accuracy': acc, 'count': float(n)}
    
    top_confs = np.array([prob[pred] for prob, pred in zip(probabilities, preds)])
    accuracies = (preds == tgts).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for b in range(num_bins):
        bin_lower = bin_boundaries[b]
        bin_upper = bin_boundaries[b + 1]
        
        in_bin = (top_confs > bin_lower) & (top_confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(top_confs[in_bin])
            ece += np.abs(avg_acc_in_bin - avg_conf_in_bin) * prop_in_bin
            
    metrics['ece'] = float(ece)
    
    brier_scores = []
    for prob, tgt in zip(probabilities, tgts):
        one_hot = np.zeros_like(prob)
        one_hot[tgt] = 1.0
        brier_scores.append(np.sum((prob - one_hot) ** 2))
        
    metrics['brier_score'] = float(np.mean(brier_scores))
    
    if question_type == 'score':
        mae = float(np.mean(np.abs(preds - tgts)))
        metrics['mae'] = mae
        
    return metrics
