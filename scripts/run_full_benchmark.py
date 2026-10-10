"""
Full benchmarking script for Zesyde.
Evaluates accuracy, Expected Calibration Error (ECE), and Brier Score
on the full held-out eval datasets, directly comparing FP32 vs INT8 ONNX graphs.
"""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import sys

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zesyde.collator import ZesydeCollator
from zesyde.data import load_frozen_dataset
from zesyde.schema import DecisionExample
from zesyde.metrics import compute_metrics
from scipy.special import softmax

def build_ort_session(model_path: Path) -> ort.InferenceSession:
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

def onnx_forward_batch(session, collator, examples):
    score_rows = []
    for ex in examples:
        batch = collator([ex])
        ort_inputs = {
            "input_ids": batch["input_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
            "marker_positions": np.array([batch["marker_indices"][0]], dtype=np.int64),
        }
        scores, _ = session.run(None, ort_inputs)
        score_rows.append(scores[0][: len(ex.options)].astype(np.float64))
    return score_rows

def lookup_temperature(temperatures, question_type, option_count):
    # Parse the json sidecar keys "qtype|opt_count"
    k1 = f"{question_type}|{option_count}"
    k2 = f"{question_type}|0"
    k3 = "*|0"
    return float(temperatures.get(k1, temperatures.get(k2, temperatures.get(k3, 1.0))))

def metrics_from_scores(examples, scores, temperatures):
    preds, targets, probs = [], [], []
    for ex, logits in zip(examples, scores):
        temp = lookup_temperature(temperatures, ex.question_type.value, len(ex.options))
        scaled = logits / temp
        p = softmax(scaled)
        preds.append(int(np.argmax(p)))
        targets.append(ex.target)
        probs.append(p)
    return compute_metrics(preds, targets, probs, question_type=examples[0].question_type.value)

def run_full_benchmark():
    fp32_path = Path("models/zesyde_prod_fp32.onnx")
    int8_path = Path("models/zesyde_prod_int8.onnx")
    calib_path = Path("models/zesyde_calibration.json")
    
    if not (fp32_path.exists() and int8_path.exists() and calib_path.exists()):
        print("Models not found. Run 'python scripts/build_prod_model.py' first to train and export the models.")
        return
        
    with open(calib_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
        temperatures = calib_data.get("temperatures", {})
        
    print(f"Loaded temperatures: {temperatures}")

    backbone = calib_data.get("backbone_name", "jhu-clsp/mmBERT-base")
    print(f"Using tokenizer for backbone: {backbone}")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    collator = ZesydeCollator(tokenizer=tokenizer, max_length=8192)

    fp32_session = build_ort_session(fp32_path)
    int8_session = build_ort_session(int8_path)
    
    datasets = {}
    if Path("data/eval_multilingual").exists():
        datasets.update({
            "belebele_es (choice/4)": "data/eval_multilingual/belebele_choice_es.jsonl",
            "belebele_hi (choice/4)": "data/eval_multilingual/belebele_choice_hi.jsonl",
            "belebele_ar (choice/4)": "data/eval_multilingual/belebele_choice_ar.jsonl",
            "xnli_fr (choice/3)": "data/eval_multilingual/xnli_choice_fr.jsonl",
            "xnli_de (noul/2)": "data/eval_multilingual/xnli_noul_de.jsonl",
        })
    if Path("data/eval").exists():
        datasets.update({
            "ag_news (choice/4)": "data/eval/ag_news.jsonl",
            "emotion_choice (choice/6)": "data/eval/emotion_choice.jsonl",
            "emotion_noul (noul/2)": "data/eval/emotion_noul.jsonl",
            "sst5_score (score/5)": "data/eval/sst5_score.jsonl",
        })

    print("\n=== Running Full Accuracy & Calibration Benchmark ===")
    for name, path in datasets.items():
        # Taking the report half to match build script (index 1::2)
        # This prevents leaking the calibration half (index 0::2) into the final metrics
        examples = load_frozen_dataset(Path(path))[1::2]
        
        print(f"\nEvaluating: {name} (N={len(examples)})")
        
        fp32_scores = onnx_forward_batch(fp32_session, collator, examples)
        int8_scores = onnx_forward_batch(int8_session, collator, examples)
        
        fp32_metrics = metrics_from_scores(examples, fp32_scores, temperatures)
        int8_metrics = metrics_from_scores(examples, int8_scores, temperatures)
        
        def fmt(m):
            s = f"Acc: {m.get('accuracy',0)*100:.1f}% | ECE: {m.get('ece',0):.4f} | Brier: {m.get('brier_score',0):.4f}"
            if "mae" in m: s += f" | MAE: {m['mae']:.3f}"
            return s
            
        print(f"  FP32 : {fmt(fp32_metrics)}")
        print(f"  INT8 : {fmt(int8_metrics)}")
        print(f"  Drift: {abs(fp32_metrics['accuracy'] - int8_metrics['accuracy'])*100:.1f}pp")

if __name__ == "__main__":
    run_full_benchmark()
