"""
Build the production Zesyde artifact end-to-end: train -> act head -> calibrate -> export -> verify.
"""
import argparse
import random
import sys
from pathlib import Path
import numpy as np
import torch

import os

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load .env file natively if it exists
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ[k.strip()] = v.strip(' "\'')

# Force UTF-8 for torch.onnx exporter
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

from zesyde.pipeline import BuildPipeline
from zesyde.pipeline_utils import DEFAULT_BACKBONE, DEFAULT_MULTILINGUAL_BACKBONE, _calibrate_only

def main() -> None:
    parser = argparse.ArgumentParser(description="Train, calibrate and export the production Zesyde artifact.")
    parser.add_argument("--backbone", type=str, default=None, help=f"Transformer backbone (default: {DEFAULT_MULTILINGUAL_BACKBONE} if --multilingual else {DEFAULT_BACKBONE}).")
    parser.add_argument("--multilingual", dest="multilingual", action="store_true", default=False, help="Use the ZesydeMultilingualModel architecture instead of the base ZesydeModel (default: False).")
    parser.add_argument("--no-multilingual", dest="multilingual", action="store_false", help="Use the base English ZesydeModel architecture.")
    parser.add_argument("--use-peft", dest="use_peft", action="store_true", default=True, help="Use PEFT (LoRA) adapters (default: True).")
    parser.add_argument("--no-peft", dest="use_peft", action="store_false", help="Disable PEFT (LoRA) and fine-tune all backbone parameters.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA attention rank (default: 16).")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor (default: 32).")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate (default: 0.05).")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (cuda, cpu). Default: auto-detect.")
    parser.add_argument("--rlcd-epochs", type=int, default=1, help="Number of RLCD fine-tuning epochs with strictly proper scoring rules.")
    parser.add_argument("--sla-latency", type=float, default=60.0, help="p99 SLA latency target in ms (default: 40.0 for large on CPU / 15.0 for base).")
    parser.add_argument("--train-limit", type=int, default=5000, help="Total number of multi-task records to train on (default: 5,000).")
    parser.add_argument("--head-heads", type=int, default=None, help="Number of attention heads for cross-option attention head (default: None for context-fused MLP head).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--eval-limit", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length (default: 1024).")
    parser.add_argument("--lr-head", type=float, default=5e-4, help="Learning rate for decision and act heads (default: 5e-4).")
    parser.add_argument("--lr-backbone", type=float, default=3e-4, help="Learning rate for backbone / LoRA adapters (default: 3e-4).")
    parser.add_argument("--lr-rlcd", type=float, default=1e-5, help="Learning rate for RLCD phase (default: 1e-5).")
    parser.add_argument("--rlcd-kl-coef", type=float, default=0.1, help="KL penalty coefficient for RLCD against frozen baseline (default: 0.1).")
    parser.add_argument("--eval-steps", type=int, default=100, help="Step interval for active validation during RLCD (default: 100).")
    parser.add_argument("--collapse-threshold", type=float, default=0.80, help="Relative accuracy threshold triggering early stopping on collapse (default: 0.80).")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient norm for clipping during training (default: 1.0).")
    parser.add_argument("--act-epochs", type=int, default=1)
    parser.add_argument("--act-threshold", type=float, default=0.6)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/zesyde_trained.pt"))
    parser.add_argument("--calibration", type=Path, default=Path("models/zesyde_calibration.json"))
    parser.add_argument("--fp32-out", type=Path, default=Path("models/zesyde_prod_fp32.onnx"))
    parser.add_argument("--int8-out", type=Path, default=Path("models/zesyde_prod_int8.onnx"))
    parser.add_argument("--benchmark-runs", type=int, default=100)
    parser.add_argument("--calibrate-only", action="store_true", help="Skip training/export; refit temperatures from an existing checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint in --checkpoint-dir.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume training from a specific checkpoint path.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"), help="Directory where intermediate pause/resume checkpoints are stored.")
    parser.add_argument("--no-interactive-pause", dest="interactive_pause", action="store_false", default=True, help="Disable interactive terminal [p] key monitoring.")
    
    args = parser.parse_args()
    if args.backbone is None:
        args.backbone = DEFAULT_MULTILINGUAL_BACKBONE if args.multilingual else DEFAULT_BACKBONE

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    pipeline = BuildPipeline(args)

    if args.calibrate_only:
        pipeline.phase_0_data()
        _calibrate_only(args, pipeline.collator, pipeline.calib_sets, pipeline.test_sets)
        return

    pipeline.run_all()

if __name__ == "__main__":
    main()