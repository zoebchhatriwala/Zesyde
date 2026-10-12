"""Entrypoint launcher for the Zesyde Dino Showcase Demo.

Run with:
    python demos/dino/run.py
    python demos/dino/run.py --port 8088 --int8
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root and src/ are on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn
from demos.dino.server import app, init_router

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Zesyde Dino Showcase Demo.")
    parser.add_argument("--port", type=int, default=8088, help="Port to bind (default: 8088).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1).")
    parser.add_argument("--int8", action="store_true", default=True, help="Use INT8 quantized ONNX graph.")
    parser.add_argument("--fp32", action="store_true", default=False, help="Use FP32 full precision graph.")
    args = parser.parse_args()

    print("=" * 60)
    print("  ZESYDE DINO SHOWCASE (demos/dino)")
    print("  Autonomous Decision Router | 150M-class Foundation Model")
    print("=" * 60)
    print(f"Precision: {'FP32 Full Precision' if args.fp32 else 'INT8 Quantized'}")

    init_router(prefer_int8=not args.fp32)

    print(f"\nServing interactive Dino showcase on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
