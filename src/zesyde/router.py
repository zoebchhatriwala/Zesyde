"""High-level Zesyde Router interface for sub-15ms calibrated decisions.

Provides a 3-line drop-in API for agent systems, tool routing, and classification:
    router = ZesydeRouter.from_pretrained("path/to/model")
    decision = router.decide(context="...", options=["Tool A", "Tool B"])
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from zesyde.collator import ZesydeCollator
from zesyde.schema import DecisionExample, QuestionType


@dataclass
class DecisionResult:
    """Strongly typed output of a Zesyde routing decision."""
    selected_option: str
    confidence: float
    probabilities: Dict[str, float]
    should_escalate: bool
    escalate_probability: float
    latency_ms: float
    question_type: str
    options: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_option": self.selected_option,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "should_escalate": self.should_escalate,
            "escalate_probability": self.escalate_probability,
            "latency_ms": self.latency_ms,
            "question_type": self.question_type,
            "options": self.options,
        }


class ZesydeRouter:
    """Calibrated, non-autoregressive decision router powered by ONNX Runtime CPU.
    
    Features:
    - INT8 CPU batch-1 latency of ~11-13ms
    - Out-of-sample temperature scaling calibration
    - Native act/escalate head for low-confidence or high-risk routing
    - Dynamic candidate sets (arbitrary K options at runtime)
    """

    def __init__(
        self,
        int8_session: Optional[ort.InferenceSession] = None,
        fp32_session: Optional[ort.InferenceSession] = None,
        tokenizer: AutoTokenizer = None,
        temperatures: Dict[str, float] = None,
        max_length: int = 8192,
        precision_routing: bool = True,
        prefer_fp32: bool = True,
    ):
        if int8_session is None and fp32_session is None:
            raise ValueError("At least one of int8_session or fp32_session must be provided.")
        self.int8_session = int8_session
        self.fp32_session = fp32_session
        self.tokenizer = tokenizer
        self.temperatures = temperatures or {}
        self.max_length = max_length
        self.precision_routing = precision_routing
        self.prefer_fp32 = prefer_fp32
        self.collator = ZesydeCollator(tokenizer=tokenizer, max_length=max_length)

    @classmethod
    def from_pretrained(
        cls,
        model_dir_or_path: Union[str, Path] = "models",
        tokenizer_name: Optional[str] = None,
        max_length: int = 8192,
        precision_routing: bool = True,
        prefer_fp32: bool = True,
    ) -> "ZesydeRouter":
        """Loads a ZesydeRouter from a local model directory or file path.
        
        Args:
            model_dir_or_path: Path containing zesyde_prod_fp32.onnx/zesyde_prod_int8.onnx and calibration JSON
            tokenizer_name: Hugging Face tokenizer identifier or directory (defaults to backbone in calibration JSON or jhu-clsp/mmBERT-base)
            max_length: Context window (default: 8192)
            precision_routing: Route ordinal 'score' questions to FP32 graph if available
            prefer_fp32: Default to full precision FP32 graph (recommended; ~13ms on CPU with zero quantization drift)
        """
        p = Path(model_dir_or_path)
        
        if p.is_file():
            model_dir = p.parent
            direct_file = p
        else:
            model_dir = p
            direct_file = None

        # Set up CPU session options (4 intra-op threads is optimal)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load FP32 session if available
        fp32_session = None
        fp32_candidates = [
            model_dir / "zesyde_prod_fp32.onnx",
            model_dir / "onnx" / "zesyde_prod_fp32.onnx",
            model_dir / "onnx" / "model_fp32.onnx",
        ]
        if direct_file and "fp32" in direct_file.name.lower():
            fp32_candidates.insert(0, direct_file)
        for candidate in fp32_candidates:
            if candidate.exists():
                try:
                    fp32_session = ort.InferenceSession(
                        str(candidate), sess_options=sess_opts, providers=["CPUExecutionProvider"]
                    )
                    break
                except Exception:
                    pass

        # Load INT8 session if available
        int8_session = None
        int8_candidates = [
            model_dir / "zesyde_prod_int8.onnx",
            model_dir / "onnx" / "zesyde_prod_int8.onnx",
            model_dir / "onnx" / "model_int8.onnx",
        ]
        if direct_file and "int8" in direct_file.name.lower():
            int8_candidates.insert(0, direct_file)
        for candidate in int8_candidates:
            if candidate.exists():
                try:
                    int8_session = ort.InferenceSession(
                        str(candidate), sess_options=sess_opts, providers=["CPUExecutionProvider"]
                    )
                    break
                except Exception:
                    pass

        if fp32_session is None and int8_session is None:
            raise FileNotFoundError(f"Neither FP32 nor INT8 ONNX models found at {model_dir}")

        # Load calibration temperatures
        calib_file = model_dir / "zesyde_calibration.json"
        if not calib_file.exists():
            calib_file = model_dir / "calibration.json"
        
        temperatures = {}
        backbone_from_calib = None
        if calib_file.exists():
            try:
                data = json.loads(calib_file.read_text(encoding="utf-8"))
                temperatures = data.get("temperatures", data)
                backbone_from_calib = data.get("backbone_name")
            except Exception:
                temperatures = {}

        # Load tokenizer
        if (model_dir / "tokenizer.json").exists():
            tok_source = str(model_dir)
        elif tokenizer_name is not None:
            tok_source = tokenizer_name
        elif backbone_from_calib:
            tok_source = backbone_from_calib
        else:
            tok_source = "jhu-clsp/mmBERT-base"

        tokenizer = AutoTokenizer.from_pretrained(tok_source)

        return cls(
            int8_session=int8_session,
            fp32_session=fp32_session,
            tokenizer=tokenizer,
            temperatures=temperatures,
            max_length=max_length,
            precision_routing=precision_routing,
        )

    def get_temperature(self, question_type: str, option_count: int) -> float:
        """Lookup temperature with hierarchical fallbacks."""
        # 1. Exact match: "choice|4"
        key_exact = f"{question_type}|{option_count}"
        if key_exact in self.temperatures:
            return float(self.temperatures[key_exact])
        # 2. Primitive fallback: "choice|0"
        key_primitive = f"{question_type}|0"
        if key_primitive in self.temperatures:
            return float(self.temperatures[key_primitive])
        # 3. Global fallback: "*|0"
        key_global = "*|0"
        if key_global in self.temperatures:
            return float(self.temperatures[key_global])
        return 1.0

    def decide(
        self,
        context: str,
        options: List[str],
        question: str = "Select the best option:",
        question_type: Union[QuestionType, str] = "choice",
        escalation_threshold: float = 0.5,
    ) -> DecisionResult:
        """Executes a calibrated routing decision in ~11ms.
        
        Args:
            context: The text/query/prompt to evaluate.
            options: List of 2 to 64 candidate strings.
            question: Prompt framing (default: 'Select the best option:').
            question_type: 'choice' (categorical), 'noul' (null fallback), or 'score' (ordinal).
            escalation_threshold: Probability threshold above which act/escalate triggers.
            
        Returns:
            DecisionResult with calibrated probabilities and escalation status.
        """
        if not options:
            raise ValueError("options list cannot be empty")
        
        qtype_str = question_type.value if isinstance(question_type, QuestionType) else str(question_type).lower()
        t0 = time.perf_counter()

        # Select inference session:
        # Prefer FP32 for maximum accuracy and calibration fidelity (~13ms on CPU).
        # Fall back gracefully to INT8 if FP32 is unavailable or prefer_fp32 is False.
        if self.prefer_fp32 and self.fp32_session is not None:
            session = self.fp32_session
        elif self.precision_routing and qtype_str == "score" and self.fp32_session is not None:
            session = self.fp32_session
        elif self.int8_session is not None:
            session = self.int8_session
        else:
            session = self.fp32_session

        example = DecisionExample(
            id="query_0",
            context=context,
            question=question,
            question_type=QuestionType(qtype_str),
            options=options,
            target=0,
        )

        batch = self.collator([example])
        input_ids = batch["input_ids"].numpy()
        attention_mask = batch["attention_mask"].numpy()
        marker_positions = np.array([batch["marker_indices"][0]], dtype=np.int64)

        raw_scores, act_logits = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "marker_positions": marker_positions,
            },
        )

        num_opts = len(options)
        valid_scores = raw_scores[0, :num_opts].astype(np.float64)

        # Apply temperature calibration
        temperature = self.get_temperature(qtype_str, num_opts)
        scaled_scores = valid_scores / temperature

        # Numerically stable softmax
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probs = exp_scores / np.sum(exp_scores)

        # Escalation prediction: sigmoid of act_logits[0]
        act_logit = float(act_logits[0])
        escalate_prob = float(1.0 / (1.0 + np.exp(-act_logit)))
        should_escalate = bool(escalate_prob >= escalation_threshold)

        best_idx = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DecisionResult(
            selected_option=options[best_idx],
            confidence=float(probs[best_idx]),
            probabilities={opt: float(p) for opt, p in zip(options, probs)},
            should_escalate=should_escalate,
            escalate_probability=float(escalate_prob),
            latency_ms=round(latency_ms, 2),
            question_type=qtype_str,
            options=options,
        )

