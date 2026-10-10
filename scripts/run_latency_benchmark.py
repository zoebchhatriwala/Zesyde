"""ONNX CPU latency verification script for Zesyde.

Benchmarks batch-1 inference latency across INT8 and FP32 ONNX sessions.
"""
import argparse
import sys
from pathlib import Path

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from zesyde.pipeline_utils import build_ort_session, benchmark_latency

def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX model CPU latency.")
    parser.add_argument("--model", type=str, default="models/zesyde_prod_int8.onnx", help="Path to ONNX model file")
    parser.add_argument("--seq-len", type=int, default=128, help="Benchmark sequence length (default: 128)")
    parser.add_argument("--options", type=int, default=4, help="Number of candidate options (default: 4)")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations (default: 100)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    print(f"Loading ONNX session from {model_path}...")
    session = build_ort_session(model_path)

    # Synthetic batch-1 input
    dummy_input_ids = np.zeros((1, args.seq_len), dtype=np.int64)
    dummy_attention_mask = np.ones((1, args.seq_len), dtype=np.int64)
    dummy_marker_positions = np.array([[args.seq_len // 2 + i * 4 for i in range(args.options)]], dtype=np.int64)

    inputs = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask,
        "marker_positions": dummy_marker_positions,
    }

    print(f"Benchmarking batch-1 latency over {args.runs} runs (seq_len={args.seq_len}, options={args.options})...")
    metrics = benchmark_latency(session, inputs, num_runs=args.runs)
    print(f"  p50 : {metrics['p50']:.2f} ms")
    print(f"  p90 : {metrics['p90']:.2f} ms")
    print(f"  p99 : {metrics['p99']:.2f} ms")
    print(f"  mean: {metrics['mean']:.2f} ms")

if __name__ == "__main__":
    main()
