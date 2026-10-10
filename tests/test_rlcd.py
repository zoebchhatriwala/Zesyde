import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from zesyde.rlcd import spherical_score, log_score, brier_score, rps_score, compute_reward, rlcd_step_loss
from zesyde.rlcd_train import train_rlcd_epoch


def test_brier_score():
    # 2 options, target = 0
    # Perfect correct
    probs_correct = torch.tensor([1.0, 0.0])
    score_correct = brier_score(probs_correct, target=0)
    assert score_correct.item() == pytest.approx(1.0, abs=1e-4)

    # 100% confident on wrong answer
    probs_wrong = torch.tensor([0.0, 1.0])
    score_wrong = brier_score(probs_wrong, target=0)
    assert score_wrong.item() == pytest.approx(-1.0, abs=1e-4)

    # Uniform uncertainty
    probs_uniform = torch.tensor([0.5, 0.5])
    score_uniform = brier_score(probs_uniform, target=0)
    assert score_uniform.item() == pytest.approx(0.5, abs=1e-4)
    assert score_uniform.item() > score_wrong.item()


def test_reward_penalizes_extreme_confidence_on_incorrect():
    # Target is 0
    target = 0
    # Confident correct: target prob = 0.99
    probs_correct = torch.tensor([0.99, 0.01])
    reward_correct = compute_reward(probs_correct, target=target, question_type='choice')

    # Uncertain: target prob = 0.5
    probs_uncertain = torch.tensor([0.5, 0.5])
    reward_uncertain = compute_reward(probs_uncertain, target=target, question_type='choice')

    # Confident incorrect: target prob = 0.01 (99% confident in wrong class)
    probs_wrong = torch.tensor([0.01, 0.99])
    reward_wrong = compute_reward(probs_wrong, target=target, question_type='choice')

    # Extreme overconfidence on incorrect prediction must be heavily penalized
    assert reward_correct.item() > reward_uncertain.item()
    assert reward_uncertain.item() > reward_wrong.item()
    # Confident wrong reward should be severely negative
    assert reward_wrong.item() < -5.0


def test_rps_score_penalizes_overconfidence_on_wrong_score():
    target = 0
    # Correct score 0 with high confidence
    probs_correct = torch.tensor([0.99, 0.01, 0.0, 0.0, 0.0])
    rps_correct = rps_score(probs_correct, target=target)

    # Incorrect score with extreme confidence (rating 4 vs true rating 0)
    probs_extreme_wrong = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    rps_extreme_wrong = rps_score(probs_extreme_wrong, target=target)

    assert rps_correct.item() > 0.8
    assert rps_extreme_wrong.item() < -2.0


def test_kl_divergence_penalty_in_step_loss():
    base_logits = torch.tensor([1.0, 0.0, -1.0])
    target = 0

    # With identical reference logits, KL divergence is 0
    loss_no_drift, stats_no_drift = rlcd_step_loss(
        base_logits, target=target, ref_logits=base_logits.clone(), kl_coef=0.2
    )
    assert stats_no_drift['kl_div'] == pytest.approx(0.0, abs=1e-4)

    # With divergent reference logits, KL divergence is positive and increases total loss
    drifted_ref = torch.tensor([-5.0, 5.0, 0.0])
    loss_drift, stats_drift = rlcd_step_loss(
        base_logits, target=target, ref_logits=drifted_ref, kl_coef=0.2
    )
    assert stats_drift['kl_div'] > 0.1
    assert stats_drift['kl_loss'] > 0.0
    assert loss_drift.item() > loss_no_drift.item()


class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor([1, 2, 3]),
            'attention_mask': torch.tensor([1, 1, 1]),
            'marker_indices': [0, 1],
            'targets': torch.tensor(idx % 2),
            'question_types': 'choice',
        }


def dummy_collate(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'marker_indices': [b['marker_indices'] for b in batch],
        'targets': torch.tensor([b['targets'] for b in batch]),
        'question_types': [b['question_types'] for b in batch],
    }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, input_ids, attention_mask, marker_indices):
        b = input_ids.size(0)
        return {
            'logits': [self.linear(input_ids[i].float()[:3]) for i in range(b)]
        }


def test_active_validation_early_stopping():
    model = DummyModel()
    ref_model = DummyModel()
    ref_model.eval()

    loader = DataLoader(DummyDataset(size=20), batch_size=4, collate_fn=dummy_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Val hook that returns collapsed accuracy after step 2
    call_count = 0
    def val_hook(m):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 0.80, 0.05  # Initial baseline
        return 0.20, 0.50  # Collapsed accuracy

    stats = train_rlcd_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        ref_model=ref_model,
        val_fn=val_hook,
        eval_steps=2,
        collapse_threshold=0.80,
    )

    # Early stopping must be triggered because accuracy collapsed from 0.80 to 0.20
    assert stats['early_stopped'] is True
