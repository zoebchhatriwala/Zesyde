"""Launcher for Zesyde Adversarial Menu & Escalation Playground.

Run with:
    python demos/bad_menu/run.py
    python demos/bad_menu/run.py --port 8082
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root and src/ are on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn
from demos.bad_menu.server import app, init_router

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Zesyde Adversarial Menu Playground")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind (default: 8082)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    print("=" * 65)
    print("  ZESYDE ADVERSARIAL MENU & ESCALATION PLAYGROUND (demos/bad_menu)")
    print("  Stress-testing out-of-distribution, 'None of these', & ambiguity")
    print("=" * 65)

    init_router()

    print(f"\nServing interactive demo on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
