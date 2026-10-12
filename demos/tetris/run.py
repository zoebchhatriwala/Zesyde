"""Entrypoint launcher for the Zesyde Neural Tetris Demo.

Run with:
    python demos/tetris/run.py
    python demos/tetris/run.py --port 8000 --int8
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root and src/ are on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn
from demos.tetris.server import app, setup_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Zesyde Neural Tetris Demo.")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory with ONNX models.")
    parser.add_argument("--tokenizer", type=str, default="jhu-clsp/mmBERT-base", help="Tokenizer name.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--int8", action="store_true", default=False, help="Prefer INT8 quantized model.")
    args = parser.parse_args()

    print("=" * 60)
    print("  ZESYDE NEURAL TETRIS DEMO (demos/tetris)")
    print("  Autonomous Decision Router | 150M-class Foundation Model")
    print("=" * 60)
    print(f"Loading weights from: {args.model_dir}")
    print(f"Precision:            {'INT8 Quantized' if args.int8 else 'FP32 Full Precision'}")
    print(f"Tokenizer:            {args.tokenizer}")

    setup_agent(
        model_dir=args.model_dir,
        prefer_fp32=not args.int8,
        tokenizer=args.tokenizer,
    )

    print(f"\nServing interactive demo on http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
