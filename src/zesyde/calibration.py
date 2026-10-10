"""Post-hoc temperature scaling calibration for Zesyde decisions.

Per Section 7.3 of SPECS.md:
"Fit one temperature per (question_type, option_count) bucket on the held-out split.
This is the single highest-leverage, lowest-cost step: Laya's own numbers show mean
ECE moving from 0.466 -> 0.081 from this alone, with zero added inference latency."
"""
from typing import Dict, Tuple, List, Optional
import numpy as np
import scipy.optimize as opt
import torch
import torch.nn.functional as F

from zesyde.metrics import compute_metrics

class TemperatureCalibrator:
    """Stores and fits scalar temperature parameters T > 0 per (question_type, option_count) bucket.

    Calibrated probability for logits z:
        p = softmax(z / T)
    """
    def __init__(self):
        self.temperatures = {}

    def get_temperature(self, question_type: str, option_count: int) -> float:
        return self.temperatures.get((question_type, option_count), 1.0)

    def fit_bucket(
        self,
        question_type: str,
        option_count: int,
        logits_list: List[torch.Tensor],
        targets: List[int],
        min_t: float = 0.05,
        max_t: float = 10.0,
    ) -> float:
        if not logits_list:
            return 1.0
            
        stacked_logits = torch.stack(logits_list)
        tgt_tensor = torch.tensor(targets, dtype=torch.long)
        
        def nll_objective(log_t: np.ndarray) -> float:
            # Bound log_t to prevent overflow / infinite flattening
            bounded_log_t = float(np.clip(log_t[0], np.log(min_t), np.log(max_t)))
            t = float(np.exp(bounded_log_t))
            scaled_logits = stacked_logits / t
            loss = F.cross_entropy(scaled_logits, tgt_tensor)
            return float(loss.item())
            
        res = opt.minimize(nll_objective, [0.0], method='Nelder-Mead')
        raw_t = float(np.exp(res.x[0]))
        best_t = float(np.clip(raw_t, min_t, max_t))
        self.temperatures[(question_type, option_count)] = best_t
        return best_t

    def calibrate_logits(self, logits: torch.Tensor, question_type: str) -> torch.Tensor:
        option_count = logits.size(-1)
        t = self.get_temperature(question_type, option_count)
        return logits / t

    def calibrate_probs(self, logits: torch.Tensor, question_type: str) -> torch.Tensor:
        scaled = self.calibrate_logits(logits, question_type)
        return F.softmax(scaled, dim=-1)
