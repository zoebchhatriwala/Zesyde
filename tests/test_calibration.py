import pytest
import torch
from zesyde.calibration import TemperatureCalibrator

def test_temperature_scaling():
    scaler = TemperatureCalibrator()
    assert scaler is not None

def test_temperature_clamping_on_zero_signal():
    scaler = TemperatureCalibrator()
    # Random uninformative logits (all zeros or noisy) where optimizer would try to push T -> infinity
    logits = [torch.tensor([0.0, 0.0]) for _ in range(20)]
    targets = [i % 2 for i in range(20)]
    
    fitted_t = scaler.fit_bucket('noul', 2, logits, targets, min_t=0.05, max_t=10.0)
    # Must be safely capped at max_t (10.0) rather than blowing up to astronomical numbers
    assert fitted_t <= 10.0
    assert fitted_t >= 0.05
    assert scaler.get_temperature('noul', 2) == fitted_t

def test_temperature_scaling_on_overconfident_logits():
    scaler = TemperatureCalibrator()
    # Overconfident wrong predictions
    logits = [torch.tensor([5.0, -5.0]) if t == 1 else torch.tensor([-5.0, 5.0]) for t in [0, 1] * 10]
    targets = [1, 0] * 10
    
    fitted_t = scaler.fit_bucket('choice', 2, logits, targets, min_t=0.05, max_t=10.0)
    assert fitted_t <= 10.0
    assert fitted_t >= 0.05

