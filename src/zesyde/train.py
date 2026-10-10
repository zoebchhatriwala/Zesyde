"""Training loop implementations for Zesyde."""
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from zesyde.model import ZesydeModel
from zesyde.losses import compute_batch_loss


def train_epoch(
    model: ZesydeModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    device: str = "cpu",
    smoothing: float = 0.1,
    pause_controller: Optional[Any] = None,
    phase: str = "phase_1_warmstart",
    epoch: int = 1,
    total_epochs: int = 1,
    start_step: int = 0,
    extra_state: Optional[Dict[str, Any]] = None,
) -> float:
    """Run a single training epoch with pause/resume support and return the average loss."""
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(loader, desc="Training Epoch", leave=False)
    for batch_idx, batch in enumerate(progress_bar, start=1):
        if batch_idx <= start_step:
            continue

        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        marker_indices = batch['marker_indices']

        out = model(input_ids, attention_mask, marker_indices)
        logits_list = out['logits']
        act_logits = out.get('act_logits')
        targets = batch['targets'].to(device)

        question_types = batch.get('question_types', [])
        if not question_types:
            question_types = ["choice"] * len(targets)

        teacher_probs_list = [t.to(device) if t is not None else None for t in batch.get('teacher_probs', [None] * len(targets))]
        act_targets = batch.get('act_targets')
        if act_targets is not None:
            act_targets = act_targets.to(device)

        loss, _, _ = compute_batch_loss(
            batch_logits=logits_list,
            targets=targets,
            question_types=question_types,
            teacher_probs_list=teacher_probs_list,
            smoothing=smoothing,
            act_logits=act_logits,
            act_targets=act_targets
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        if pause_controller is not None:
            pause_controller.check_pause(
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
                step=batch_idx,
                total_steps=len(loader),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_state=extra_state,
            )

    num_steps = max(1, len(loader) - start_step)
    return total_loss / num_steps
