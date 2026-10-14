#!/usr/bin/env python3
"""Launcher for Zesyde Global Trust & Safety Sentinel Demo."""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root and src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))

import uvicorn
from demos.live_sentinel.server import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zesyde Live Global Sentinel Runner")
    parser.add_argument("--port", type=int, default=8085, help="Port to bind (default: 8085)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantized ONNX graph instead of full-precision FP32")
    args = parser.parse_args()

    import os
    if args.int8:
        os.environ["ZESYDE_USE_INT8"] = "1"
        precision_str = "INT8 ONNX"
    else:
        os.environ["ZESYDE_USE_INT8"] = "0"
        precision_str = "FP32 ONNX (Full Precision, Zero Quantization Drift)"

    print("\n" + "=" * 65)
    print("  [Zesyde] GLOBAL TRUST & SAFETY SENTINEL")
    print(f"  [Zesyde] Engine: {precision_str}")
    print(f"  [Zesyde] Real-Time Multilingual Live Firehose Demo")
    print(f"  [Zesyde] Open in browser: http://{args.host}:{args.port}")
    print("=" * 65 + "\n")

    uvicorn.run(
        "demos.live_sentinel.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
