import time
import json
import torch
import numpy as np
import onnxruntime as ort
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from zesyde.calibration import TemperatureCalibrator
from zesyde.collator import ZesydeCollator
from zesyde.metrics import compute_metrics
from zesyde.model import ZesydeModel
from zesyde.schema import DecisionExample
from zesyde.train import train_epoch

DEFAULT_BACKBONE = "answerdotai/ModernBERT-large"
DEFAULT_MULTILINGUAL_BACKBONE = "jhu-clsp/mmBERT-base"


def make_collate(collator: ZesydeCollator):
    """DataLoader collate_fn that keeps per-example metadata alongside the tensors."""

    def collate_fn(batch_examples: List[DecisionExample]) -> Dict[str, Any]:
        batch = collator(batch_examples)
        batch["question_types"] = [ex.question_type.value for ex in batch_examples]
        return batch

    return collate_fn


def majority_baseline(examples: List[DecisionExample]) -> float:
    """Accuracy of always predicting the most frequent target class."""
    counts = Counter(ex.target for ex in examples)
    return counts.most_common(1)[0][1] / len(examples)


def collect_predictions(
    model: ZesydeModel,
    collator: ZesydeCollator,
    examples: List[DecisionExample],
    batch_size: int = 32,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """One forward pass over `examples`, returning per-example logits plus metadata.

    `shuffle=False` guarantees row order matches `examples`, so the running offset
    can be used to recover per-example metadata.
    """
    model.eval()
    loader = DataLoader(
        examples, batch_size=batch_size, shuffle=False, collate_fn=make_collate(collator)
    )
    rows: List[Dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting Predictions"):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                marker_indices=batch["marker_indices"],
            )
            act_logits = out["act_logits"].cpu().numpy()
            for i, logits in enumerate(out["logits"]):
                ex = examples[offset + i]
                rows.append(
                    {
                        "id": ex.id,
                        "question_type": ex.question_type.value,
                        "option_count": len(ex.options),
                        "target": ex.target,
                        "logits": logits.cpu().numpy().astype(np.float64),
                        "act_logit": float(act_logits[i]),
                    }
                )
            offset += len(out["logits"])
    return rows


def lookup_temperature(
    temperatures: Dict[Tuple[str, int], float],
    question_type: str,
    option_count: int,
) -> float:
    """Resolve a temperature with graceful fallback: exact bucket -> per-type -> global."""
    return float(
        temperatures.get(
            (question_type, option_count),
            temperatures.get((question_type, 0), temperatures.get(("*", 0), 1.0)),
        )
    )


def metrics_from_rows(
    rows: List[Dict[str, Any]],
    temperatures: Optional[Dict[Tuple[str, int], float]] = None,
) -> Dict[str, float]:
    """Accuracy / ECE / Brier for a row set, optionally temperature-calibrated."""
    preds: List[int] = []
    targets: List[int] = []
    probs: List[np.ndarray] = []
    for r in rows:
        logits = r["logits"]
        if temperatures is not None:
            logits = logits / lookup_temperature(
                temperatures, r["question_type"], r["option_count"]
            )
        p = softmax(logits)
        preds.append(int(np.argmax(p)))
        targets.append(r["target"])
        probs.append(p)
    q_type = rows[0]["question_type"] if rows else "choice"
    return compute_metrics(preds, targets, probs, question_type=q_type)


def derive_act_targets(
    rows: List[Dict[str, Any]], uncertain_threshold: float = 0.6
) -> List[float]:
    """Escalate target from a *frozen* reference model: 1 if wrong or under-confident.

    Computing this from a frozen snapshot keeps the act head's objective independent
    of its own parameters, so it cannot collapse to a trivial constant.
    """
    targets: List[float] = []
    for r in rows:
        probs = softmax(r["logits"])
        wrong = int(np.argmax(probs)) != r["target"]
        unsure = float(probs[r["target"]]) < uncertain_threshold
        targets.append(1.0 if (wrong or unsure) else 0.0)
    return targets


def train_act_head(
    model: ZesydeModel,
    collator: ZesydeCollator,
    examples: List[DecisionExample],
    act_targets: List[float],
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    pause_controller: Optional[Any] = None,
    start_epoch: int = 1,
    extra_state: Optional[Dict[str, Any]] = None,
) -> float:
    """Train only the act/escalate head against precomputed targets with pause/resume support."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.option_head.parameters():
        p.requires_grad = False
    for p in model.act_head.parameters():
        p.requires_grad = True

    items = [{"example": ex, "act_target": t} for ex, t in zip(examples, act_targets)]

    def collate_fn(batch_items):
        exs = [it["example"] for it in batch_items]
        batch = collator(exs)
        batch["question_types"] = [ex.question_type.value for ex in exs]
        batch["act_targets"] = torch.tensor(
            [it["act_target"] for it in batch_items], dtype=torch.float32
        )
        return batch

    loader = DataLoader(items, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.act_head.parameters(), lr=lr, weight_decay=0.1)
    last_loss = 0.0
    for ep in range(start_epoch, epochs + 1):
        last_loss = train_epoch(
            model,
            loader,
            optimizer,
            device=device,
            pause_controller=pause_controller,
            phase="phase_4_act_head",
            epoch=ep,
            total_epochs=epochs,
            extra_state=extra_state,
        )

    for p in model.option_head.parameters():
        p.requires_grad = True
    return last_loss


def mean_nll(rows: List[Dict[str, Any]], temperatures: Optional[Dict[Tuple[str, int], float]] = None) -> float:
    """Mean negative log-likelihood of the gold class.

    NLL is a strictly proper score, so it is the right quantity to *validate* a
    temperature fit on data the fit never saw. Binned ECE is too noisy for this at
    the handful-of-hundreds-of-rows scale of a single bucket.
    """
    if not rows:
        return float("inf")
    total = 0.0
    for r in rows:
        logits = r["logits"]
        if temperatures is not None:
            logits = logits / lookup_temperature(
                temperatures, r["question_type"], r["option_count"]
            )
        probs = softmax(logits)
        total -= float(np.log(max(float(probs[r["target"]]), 1e-12)))
    return total / len(rows)


def fit_temperatures(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], float]:
    """Fit one temperature per (question_type, option_count) bucket, plus fallbacks.

    `(qtype, 0)` holds a per-primitive temperature and `("*", 0)` a global one so the
    serving path always has a sane value for option counts never seen in calibration.

    The rows are split fit/validate. A fitted temperature is only kept when it lowers
    the *out-of-sample* NLL for the rows that bucket will actually serve; otherwise
    T = 1.0 is kept. Without this guard a sharpening temperature fitted on a few
    hundred rows can make an already-calibrated bucket measurably worse.
    """
    fit_rows = rows[0::2]
    val_rows = rows[1::2]

    calibrator = TemperatureCalibrator()
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in fit_rows:
        buckets[(r["question_type"], r["option_count"])].append(r)

    def _fit(qt: str, n: int, group: List[Dict[str, Any]]) -> None:
        if not group:
            return
        # Aggregate buckets such as (question_type, 0) mix option counts, so rows are
        # padded to the widest option set with the same -1e4 sentinel used by
        # onnx_export.py. Padded columns contribute exp(-1e4) ~= 0 to the softmax, so
        # the NLL objective is numerically identical to fitting each row at its own width.
        width = max(len(g["logits"]) for g in group)
        padded: List[torch.Tensor] = []
        for g in group:
            row = np.full(width, -1e4, dtype=np.float32)
            row[: len(g["logits"])] = g["logits"]
            padded.append(torch.tensor(row, dtype=torch.float32))
        calibrator.fit_bucket(qt, n, padded, [g["target"] for g in group])

    for (qt, n), group in sorted(buckets.items()):
        _fit(qt, n, group)
    for qt in sorted({r["question_type"] for r in fit_rows}):
        _fit(qt, 0, [r for r in fit_rows if r["question_type"] == qt])
    _fit("*", 0, fit_rows)

    # --- Out-of-sample guard -------------------------------------------------
    # Specific buckets are validated against the validation rows they will serve;
    # aggregate buckets (n = 0) and the global bucket are validated against all
    # validation rows, since they exist for option counts calibration never saw.
    guarded: Dict[Tuple[str, int], float] = {}
    minimum_support = 20
    for key, fitted_t in calibrator.temperatures.items():
        qt, n = key
        if key == ("*", 0):
            support = val_rows
        elif n == 0:
            support = [r for r in val_rows if r["question_type"] == qt]
        else:
            support = [
                r
                for r in val_rows
                if r["question_type"] == qt and r["option_count"] == n
            ]

        if len(support) < minimum_support:
            guarded[key] = fitted_t
            continue

        # Accept only when the fit helps on the validation slice by *both* measures:
        # NLL (the strictly proper score the fit targets) and ECE (the metric we report).
        # Rejecting falls back to T = 1.0, which is always a safe default.
        unscaled_nll = mean_nll(support, {key: 1.0})
        scaled_nll = mean_nll(support, {key: fitted_t})
        unscaled_ece = metrics_from_rows(support, {key: 1.0})["ece"]
        scaled_ece = metrics_from_rows(support, {key: fitted_t})["ece"]
        guarded[key] = fitted_t if (
            scaled_nll <= unscaled_nll and scaled_ece <= unscaled_ece
        ) else 1.0

    return guarded


def serialize_temperatures(temperatures: Dict[Tuple[str, int], float]) -> Dict[str, float]:
    """JSON-safe form: tuple keys become "<question_type>|<option_count>" strings."""
    return {f"{k[0]}|{k[1]}": float(v) for k, v in temperatures.items()}


def write_sidecar(
    path: Path,
    temperatures: Dict[Tuple[str, int], float],
    backbone_name: str = DEFAULT_BACKBONE,
) -> None:
    """Persist the temperature table consumed by api/main.py (no torch needed to read)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone_name": backbone_name,
                "fitted_on": "calibration half of data/eval/*.jsonl",
                "temperatures": serialize_temperatures(temperatures),
            },
            f,
            indent=2,
        )


def build_ort_session(model_path: Path) -> "ort.InferenceSession":
    """Session options identical to `api/main.py` so benchmarks reflect served latency."""
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )


def onnx_forward_batch(
    session: "ort.InferenceSession",
    collator: ZesydeCollator,
    examples: List[DecisionExample],
) -> Tuple[List[np.ndarray], List[float]]:
    """Run the exported graph once per example, exactly as the API does for a batch."""
    score_rows: List[np.ndarray] = []
    act_rows: List[float] = []
    for ex in tqdm(examples, desc="ONNX forward"):
        batch = collator([ex])
        markers = batch["marker_indices"][0]
        ort_inputs = {
            "input_ids": batch["input_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
            "marker_positions": np.array([markers], dtype=np.int64),
        }
        scores, act_logits = session.run(None, ort_inputs)
        score_rows.append(scores[0][: len(ex.options)].astype(np.float64))
        act_rows.append(float(act_logits[0]))
    return score_rows, act_rows


def rows_from_onnx(
    session: "ort.InferenceSession",
    collator: ZesydeCollator,
    examples: List[DecisionExample],
) -> List[Dict[str, Any]]:
    scores, acts = onnx_forward_batch(session, collator, examples)
    return [
        {
            "id": ex.id,
            "question_type": ex.question_type.value,
            "option_count": len(ex.options),
            "target": ex.target,
            "logits": s,
            "act_logit": a,
        }
        for ex, s, a in zip(examples, scores, acts)
    ]


def benchmark_latency(
    session: "ort.InferenceSession",
    ort_inputs: Dict[str, np.ndarray],
    num_runs: int = 100,
    warmup_runs: int = 15,
) -> Dict[str, float]:
    """Percentile latency in milliseconds for a fixed batch-1 input."""
    for _ in tqdm(range(warmup_runs), desc="Warmup", leave=False):
        session.run(None, ort_inputs)
    latencies = []
    for _ in tqdm(range(num_runs), desc="Benchmark", leave=False):
        t0 = time.perf_counter()
        session.run(None, ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(latencies)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Compact one-line metric rendering for terminal reports."""
    parts = [
        f"acc={metrics.get('accuracy', 0.0) * 100:.1f}%",
        f"ECE={metrics.get('ece', 0.0):.4f}",
        f"Brier={metrics.get('brier_score', 0.0):.4f}",
    ]
    if "mae" in metrics:
        parts.append(f"MAE={metrics['mae']:.3f}")
    return " | ".join(parts)


EVAL_FILES = {
    "ag_news (choice/4)": "data/eval/ag_news.jsonl",
    "emotion_choice (choice/6)": "data/eval/emotion_choice.jsonl",
    "emotion_noul (noul/2)": "data/eval/emotion_noul.jsonl",
    "sst5_score (score/5)": "data/eval/sst5_score.jsonl",
}

MULTILINGUAL_EVAL_FILES = {
    "belebele_es (choice/4)": "data/eval_multilingual/belebele_choice_es.jsonl",
    "belebele_hi (choice/4)": "data/eval_multilingual/belebele_choice_hi.jsonl",
    "belebele_ar (choice/4)": "data/eval_multilingual/belebele_choice_ar.jsonl",
    "xnli_fr (choice/3)": "data/eval_multilingual/xnli_choice_fr.jsonl",
    "xnli_de (noul/2)": "data/eval_multilingual/xnli_noul_de.jsonl",
}


def _calibrate_only(args, collator, calib_sets, test_sets):
    """Refit temperatures from an existing checkpoint without retraining or exporting.

    The temperature fit depends only on logits, not on the export step, so once a
    checkpoint exists calibration can be re-run (for example after tightening the
    out-of-sample guard) in seconds instead of re-running the whole pipeline.
    """
    print(f"\n--- Calibrate-only: loading {args.checkpoint} ---")
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = ZesydeModel(backbone_name_or_path=payload.get("backbone_name", DEFAULT_BACKBONE))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    calib_rows: List[Dict[str, Any]] = []
    per_set_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name in test_sets:
        calib_rows += collect_predictions(model, collator, calib_sets[name])
        per_set_rows[name] = collect_predictions(model, collator, test_sets[name])
    test_rows = [row for name in test_sets for row in per_set_rows[name]]

    before = metrics_from_rows(test_rows)
    temperatures = fit_temperatures(calib_rows)
    after = metrics_from_rows(test_rows, temperatures)

    print("  Fitted temperatures (the guard may pin a fit to 1.0):")
    for key in sorted(temperatures, key=lambda k: (k[0], k[1])):
        note = "  <- guard rejected fit, serving uncalibrated" if temperatures[key] == 1.0 else ""
        print(f"    T[{key[0]}|{key[1]}] = {temperatures[key]:.4f}{note}")

    print("\n  Report half, before vs after calibration:")
    for name in test_sets:
        raw = metrics_from_rows(per_set_rows[name])
        scaled = metrics_from_rows(per_set_rows[name], temperatures)
        print(
            f"    {name:28s} ECE {raw['ece']:.4f} -> {scaled['ece']:.4f}"
            f" | Brier {raw['brier_score']:.4f} -> {scaled['brier_score']:.4f}"
        )
    print(
        f"    {'OVERALL':28s} ECE {before['ece']:.4f} -> {after['ece']:.4f}"
        f" | Brier {before['brier_score']:.4f} -> {after['brier_score']:.4f}"
    )

    write_sidecar(args.calibration, temperatures)
    print(f"\nCalibration sidecar -> {args.calibration}")


