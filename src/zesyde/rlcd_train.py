import copy
from typing import Optional, Dict, Any, Callable, Tuple, List
import torch
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from zesyde.model import ZesydeModel
from zesyde.rlcd import rlcd_step_loss
from zesyde.metrics import compute_metrics


def evaluate_val(model: ZesydeModel, val_loader: DataLoader, device: str = "cpu") -> Tuple[float, float]:
    """Helper to evaluate raw accuracy and ECE on a validation DataLoader."""
    was_training = model.training
    model.eval()
    predictions = []
    targets = []
    probabilities = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            marker_indices = batch['marker_indices']
            out = model(input_ids, attention_mask, marker_indices)
            logits = out['logits']
            for i in range(len(logits)):
                probs = torch.softmax(logits[i], dim=-1).cpu().numpy()
                pred = int(torch.argmax(logits[i], dim=-1).item())
                predictions.append(pred)
                targets.append(int(batch['targets'][i].item()))
                probabilities.append(probs)
    if was_training:
        model.train()
    metrics = compute_metrics(predictions, targets, probabilities)
    return float(metrics.get('accuracy', 0.0)), float(metrics.get('ece', 0.0))


def train_rlcd(
    model: ZesydeModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-5,
    alpha: float = 0.5,
    kl_coef: float = 0.1,
    max_grad_norm: float = 1.0,
    eval_steps: int = 50,
    collapse_threshold: float = 0.80,
    ref_model: Optional[ZesydeModel] = None,
):
    """Trains the Zesyde model using RLCD with KL penalty, gradient clipping, and active validation."""
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    
    # Establish frozen baseline reference model for KL divergence constraint
    if ref_model is None:
        ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.to(accelerator.device)
    
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Compute initial baseline accuracy and ECE
    baseline_acc, baseline_ece = evaluate_val(model, val_loader, device=str(accelerator.device))
    if accelerator.is_local_main_process:
        print(f"Initial Baseline Accuracy: {baseline_acc * 100:.1f}% | Baseline ECE: {baseline_ece:.4f}")
        
    best_acc = baseline_acc
    best_state_dict = copy.deepcopy(model.state_dict())
    early_stopped = False
    global_step = 0
    
    for epoch in range(epochs):
        if early_stopped:
            break
            
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"RLCD Epoch {epoch + 1}/{epochs}", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            global_step += 1
            optimizer.zero_grad()
            out = model(batch['input_ids'], batch['attention_mask'], batch['marker_indices'])
            logits = out['logits']
            
            with torch.no_grad():
                ref_out = ref_model(batch['input_ids'], batch['attention_mask'], batch['marker_indices'])
                ref_logits = ref_out['logits']
            
            # Compute loss per item with KL divergence penalty against frozen baseline
            loss = torch.tensor(0.0, device=accelerator.device)
            for i in range(len(logits)):
                ref_l = ref_logits[i] if ref_logits is not None else None
                step_loss, _ = rlcd_step_loss(
                    logits[i],
                    int(batch['targets'][i].item()),
                    ref_logits=ref_l,
                    kl_coef=kl_coef,
                )
                loss += step_loss
            loss = loss / len(logits)
            accelerator.backward(loss)
            
            # Gradient clipping to prevent logit explosion
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Active Validation Hook during epoch
            if eval_steps > 0 and global_step % eval_steps == 0:
                val_acc, val_ece = evaluate_val(model, val_loader, device=str(accelerator.device))
                if accelerator.is_local_main_process:
                    print(f"\n[Step {global_step}] Active Val | Acc: {val_acc * 100:.1f}% (Baseline: {baseline_acc * 100:.1f}%) | ECE: {val_ece:.4f}")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state_dict = copy.deepcopy(model.state_dict())
                    
                # Early stopping if baseline accuracy begins to collapse
                if baseline_acc > 0 and (val_acc < baseline_acc * collapse_threshold or val_acc < baseline_acc - 0.05):
                    if accelerator.is_local_main_process:
                        print(f"\n[EARLY STOPPING] Accuracy collapse detected ({val_acc * 100:.1f}% < Baseline: {baseline_acc * 100:.1f}%). Restoring best model weights.")
                    model.load_state_dict(best_state_dict)
                    early_stopped = True
                    break
            
        if early_stopped:
            break
            
        if accelerator.is_local_main_process:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} - Average RLCD Training Loss: {avg_loss:.4f}")
            
        # End of epoch validation
        val_acc, val_ece = evaluate_val(model, val_loader, device=str(accelerator.device))
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1} - RLCD Validation Accuracy: {val_acc * 100:.1f}% | ECE: {val_ece:.4f}\n")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            
        if baseline_acc > 0 and (val_acc < baseline_acc * collapse_threshold or val_acc < baseline_acc - 0.05):
            if accelerator.is_local_main_process:
                print(f"[EARLY STOPPING] Accuracy collapse detected at epoch end. Restoring best model weights.")
            model.load_state_dict(best_state_dict)
            break

    return model


def train_rlcd_epoch(
    model: ZesydeModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ref_model: ZesydeModel,
    device: str = "cpu",
    pause_controller: Optional[Any] = None,
    phase: str = "phase_1_5_rlcd",
    epoch: int = 1,
    total_epochs: int = 1,
    start_step: int = 0,
    extra_state: Optional[Dict[str, Any]] = None,
    kl_coef: float = 0.1,
    max_grad_norm: float = 1.0,
    val_fn: Optional[Callable[[ZesydeModel], Tuple[float, float]]] = None,
    eval_steps: int = 100,
    collapse_threshold: float = 0.80,
) -> dict:
    """Run a single epoch of RLCD training with KL penalty, gradient clipping, and active validation."""
    model.train()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
        
    total_loss = 0.0
    total_reward = 0.0
    early_stopped = False
    
    # Establish baseline metrics if validation hook is present
    baseline_acc = 0.0
    baseline_ece = 0.0
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    
    if val_fn is not None and start_step == 0:
        baseline_acc, baseline_ece = val_fn(ref_model)
        best_acc = baseline_acc
        print(f"  [Active Val] Initial Baseline: Acc={baseline_acc * 100:.1f}% | ECE={baseline_ece:.4f}")
    
    progress_bar = tqdm(loader, desc="RLCD Epoch", leave=False)
    for batch_idx, batch in enumerate(progress_bar, start=1):
        if batch_idx <= start_step:
            continue

        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        marker_indices = batch['marker_indices']
        
        out = model(input_ids, attention_mask, marker_indices)
        logits = out['logits']
        with torch.no_grad():
            ref_out = ref_model(input_ids, attention_mask, marker_indices)
            ref_logits = ref_out['logits']
            
        targets = batch['targets'].to(device)
        
        question_types = batch.get('question_types', [])
        if not question_types:
            question_types = ["choice"] * len(targets)
        
        loss = torch.tensor(0.0, device=device)
        reward = 0.0
        
        for i in range(len(logits)):
            step_loss, step_stats = rlcd_step_loss(
                logits[i],
                int(targets[i].item()),
                question_type=question_types[i],
                ref_logits=ref_logits[i],
                kl_coef=kl_coef,
            )
            loss += step_loss
            reward += step_stats['mean_reward']
            
        loss = loss / len(logits)
        loss.backward()
        
        # Gradient clipping to prevent extreme logit spikes
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        total_reward += reward / len(logits)
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'reward': f"{reward / len(logits):.3f}"})

        # Active Validation Hooks: evaluate raw accuracy and ECE periodically during training epoch
        if val_fn is not None and eval_steps > 0 and batch_idx % eval_steps == 0:
            val_acc, val_ece = val_fn(model)
            print(f"\n  [Active Val Step {batch_idx}/{len(loader)}] Acc: {val_acc * 100:.1f}% (Baseline: {baseline_acc * 100:.1f}%) | ECE: {val_ece:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())
                
            # Early stopping if baseline accuracy begins to collapse
            if baseline_acc > 0 and (val_acc < baseline_acc * collapse_threshold or val_acc < baseline_acc - 0.05):
                print(f"\n  [EARLY STOPPING] Accuracy collapse detected ({val_acc * 100:.1f}% < Baseline: {baseline_acc * 100:.1f}%). Restoring best model weights and stopping RLCD early.")
                model.load_state_dict(best_weights)
                early_stopped = True
                break

        if pause_controller is not None:
            pause_controller.check_pause(
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
                step=batch_idx,
                total_steps=len(loader),
                model=model,
                optimizer=optimizer,
                scheduler=None,
                extra_state=extra_state,
            )
        
    num_steps = max(1, (batch_idx if early_stopped else len(loader)) - start_step)
    return {
        'loss': total_loss / num_steps,
        'mean_reward': total_reward / num_steps,
        'early_stopped': early_stopped,
        'best_acc': best_acc,
    }
