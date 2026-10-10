import torch
import numpy as np
from zesyde.losses import label_smoothed_cross_entropy
from zesyde.metrics import compute_metrics

def test_cross_entropy():
    logits = torch.tensor([10.0, 0.0])
    target = 0
    loss = label_smoothed_cross_entropy(logits, target)
    assert loss.item() > 0

def test_accuracy():
    predictions = [0, 1]
    targets = [0, 1]
    probabilities = [np.array([0.9, 0.1]), np.array([0.1, 0.9])]
    metrics = compute_metrics(predictions, targets, probabilities)
    assert metrics['accuracy'] == 1.0

