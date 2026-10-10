"""Zesyde Training Checkpointing, Pause, and Resume Controller.

Provides:
- In-console live pause/resume ([P]ause, [R]esume, [S]ave & exit)
- Graceful Ctrl+C (KeyboardInterrupt / SIGINT) interception with state preservation
- Sentinel file control (touch models/checkpoints/.pause to pause from another terminal)
- Full PyTorch state serialization (weights, optimizer, LR scheduler, RNG seeds, pipeline phase)
- Seamless resumption CLI via --resume or --resume-checkpoint
"""
from __future__ import annotations

import os
import sys
import time
import random
import signal
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import numpy as np

# Cross-platform non-blocking keypress check
try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

try:
    import select
    HAS_SELECT = True
except ImportError:
    HAS_SELECT = False


class TrainingInterruptedException(Exception):
    """Raised when training is gracefully paused/interrupted and checkpointed."""
    def __init__(self, message: str, checkpoint_path: Optional[Path] = None):
        super().__init__(message)
        self.checkpoint_path = checkpoint_path


def get_non_blocking_key() -> Optional[str]:
    """Check if a key has been pressed in the terminal without blocking."""
    if not sys.stdin.isatty():
        return None

    if HAS_MSVCRT:
        if msvcrt.kbhit():
            try:
                ch = msvcrt.getch()
                # Handle special/arrow keys
                if ch in (b'\x00', b'\xe0'):
                    msvcrt.getch()
                    return None
                return ch.decode('utf-8', errors='ignore').lower()
            except Exception:
                return None
        return None

    if HAS_SELECT:
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            try:
                return sys.stdin.read(1).lower()
            except Exception:
                return None
    return None


class PauseController:
    """Monitors terminal keystrokes, signals, and sentinel files to pause/resume training."""

    def __init__(
        self,
        checkpoint_dir: Path = Path("models/checkpoints"),
        checkpoint_name: str = "train_checkpoint.pt",
        enable_interactive: bool = True,
        enable_signals: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / checkpoint_name
        self.sentinel_file = self.checkpoint_dir / ".pause"
        self.enable_interactive = enable_interactive
        self.interrupted = False
        self._last_key_check = 0.0

        if enable_signals:
            self._setup_signals()

    def _setup_signals(self) -> None:
        """Register graceful SIGINT handler."""
        def sigint_handler(signum, frame):
            print("\n\n⚠️  [INTERRUPT] Caught interrupt signal (Ctrl+C). Preparing graceful pause...")
            self.interrupted = True

        try:
            signal.signal(signal.SIGINT, sigint_handler)
        except (ValueError, AttributeError):
            # Not in main thread or unsupported platform
            pass

    def check_pause(
        self,
        phase: str,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called at the end of each training batch/step to check if pause is requested."""
        now = time.time()
        pause_requested = False
        source = ""

        # 1. Signal flag check (Ctrl+C)
        if self.interrupted:
            pause_requested = True
            source = "Ctrl+C signal"

        # 2. Sentinel file check (e.g., from python scripts/pause_training.py)
        if not pause_requested and self.sentinel_file.exists():
            pause_requested = True
            source = f"sentinel file ({self.sentinel_file.name})"
            try:
                self.sentinel_file.unlink()
            except OSError:
                pass

        # 3. Interactive keyboard check (throttled to at most ~10 checks/sec)
        if not pause_requested and self.enable_interactive and (now - self._last_key_check > 0.08):
            self._last_key_check = now
            key = get_non_blocking_key()
            if key == 'p':
                pause_requested = True
                source = "keyboard 'p'"

        if not pause_requested:
            return

        # Handle pause
        self._handle_pause(
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            step=step,
            total_steps=total_steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_state=extra_state,
            source=source,
        )

    def _handle_pause(
        self,
        phase: str,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        extra_state: Optional[Dict[str, Any]],
        source: str,
    ) -> None:
        """Pause prompt handler: allow instant resume or saving and cleanly exiting."""
        # Reset signal flag
        self.interrupted = False

        print("\n" + "=" * 65)
        print(f"⏸️   TRAINING PAUSED ({source})")
        print(f"    Phase: {phase} | Epoch: {epoch}/{total_epochs} | Step: {step}/{total_steps}")
        print("=" * 65)
        print("  Options:")
        print("    [R] Resume training immediately")
        print("    [S] Save checkpoint to disk and exit cleanly")
        print("    [Q] Quit without saving")
        print("=" * 65)

        # If non-interactive environment (CI, background without tty), auto-save and exit
        if not sys.stdin.isatty():
            print("Non-interactive terminal detected. Auto-saving checkpoint...")
            ckpt_path = self.save_state(
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
                step=step,
                total_steps=total_steps,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_state=extra_state,
            )
            raise TrainingInterruptedException(
                f"Training paused via {source} and saved to {ckpt_path}",
                checkpoint_path=ckpt_path,
            )

        # Interactive loop
        while True:
            try:
                choice = input("Enter choice [R/S/Q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = 's'

            if choice in ('r', 'resume'):
                print("▶️   Resuming training...\n")
                return
            elif choice in ('s', 'save', ''):
                print("\n💾 Saving training state...")
                ckpt_path = self.save_state(
                    phase=phase,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    step=step,
                    total_steps=total_steps,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra_state=extra_state,
                )
                print(f"✅ Checkpoint successfully saved -> {ckpt_path}")
                print(f"\n👉 To resume later, run:\n   python scripts/build_prod_model.py --resume\n")
                raise TrainingInterruptedException(
                    f"Training paused and checkpoint saved to {ckpt_path}",
                    checkpoint_path=ckpt_path,
                )
            elif choice in ('q', 'quit'):
                print("❌ Exiting without saving changes.")
                sys.exit(0)
            else:
                print("Invalid option. Please enter 'r' to resume, 's' to save & exit, or 'q' to quit.")

    def save_state(
        self,
        phase: str,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        custom_path: Optional[Path] = None,
    ) -> Path:
        """Atomically saves the complete training state to disk."""
        target_path = custom_path or self.checkpoint_path
        tmp_path = target_path.with_suffix(".tmp")

        checkpoint = {
            "version": 1,
            "timestamp": time.time(),
            "phase": phase,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": step,
            "total_steps": total_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "extra_state": extra_state or {},
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

        # Write to temporary file first, then atomic rename
        torch.save(checkpoint, tmp_path)
        if target_path.exists():
            target_path.unlink()
        tmp_path.rename(target_path)
        return target_path

    @staticmethod
    def load_state(
        checkpoint_path: Path,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Loads and verifies a training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return ckpt

    @staticmethod
    def restore_rng(rng_dict: Dict[str, Any]) -> None:
        """Restores PyTorch, NumPy, and random module RNG states."""
        if not rng_dict:
            return
        if "torch" in rng_dict and rng_dict["torch"] is not None:
            torch.set_rng_state(rng_dict["torch"].cpu())
        if "cuda" in rng_dict and rng_dict["cuda"] is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng_dict["cuda"]])
            except Exception:
                pass
        if "numpy" in rng_dict and rng_dict["numpy"] is not None:
            np.random.set_state(rng_dict["numpy"])
        if "python" in rng_dict and rng_dict["python"] is not None:
            random.setstate(rng_dict["python"])
