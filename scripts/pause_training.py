"""Command-line utility to signal active training to pause or resume.

Usage:
    # From another terminal tab or window to pause active training:
    python scripts/pause_training.py

    # To remove the pause signal:
    python scripts/pause_training.py --unpause
"""
import argparse
import sys
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Pause or unpause an active Zesyde training process.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Directory where training checkpoints and sentinel files are stored.",
    )
    parser.add_argument(
        "--unpause",
        action="store_true",
        help="Clear pause sentinel flag.",
    )
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sentinel = args.checkpoint_dir / ".pause"

    if args.unpause:
        if sentinel.exists():
            sentinel.unlink()
            print("▶️  Pause signal cleared. Training will resume if waiting.")
        else:
            print("No active pause signal was found.")
    else:
        sentinel.touch()
        print(f"⏸️  Pause signal sent! (created {sentinel})")
        print("The active training process will pause and save its checkpoint at the end of the current batch.")
        print("\nTo resume training afterward, run:")
        print("   python scripts/build_prod_model.py --resume")

if __name__ == "__main__":
    main()
