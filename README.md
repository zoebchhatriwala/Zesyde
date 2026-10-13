---
language:
- multilingual
- en
- es
- hi
- ar
- fr
- de
- zh
- tr
- vi
- ru
license: mit
library_name: transformers
tags:
- decision-model
- classification
- non-autoregressive
- calibrated
- calibration
- onnx
- zero-shot
- routing
- agent-router
datasets:
- facebook/belebele
- xnli
metrics:
- accuracy
- brier_score
pipeline_tag: text-classification
---

# Zesyde: Calibrated, Non-Autoregressive Decision & Routing Foundation Model

[![PyPI version](https://img.shields.io/pypi/v/zesyde.svg)](https://pypi.org/project/zesyde/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Latency](https://img.shields.io/badge/Latency-12--35ms_CPU-brightgreen.svg)]()
[![Model Size](https://img.shields.io/badge/ONNX_INT8-318MB-success.svg)]()
[![Parameters](https://img.shields.io/badge/Parameters-313M_(150M_Core)-blue.svg)]()
[![Context](https://img.shields.io/badge/Context-8192_Tokens-orange.svg)]()

**Zesyde** is a production-ready, non-autoregressive foundation model engineered for **ultra-fast, mathematically calibrated zero-shot decisions and agent routing** across 1,800+ languages.

Instead of generating text token-by-token with massive generative LLMs, Zesyde scores arbitrary candidate sets simultaneously in **12–35 milliseconds on pure CPU** using candidate-token pooling, prompt-context fusion, and out-of-sample temperature scaling.

---

## 🚀 Key Highlights

* **Blazing Fast (Non-Autoregressive)**: Generates decisions in **12–35ms** on pure CPU via INT8 ONNX Runtime (p50 ~12ms on short sequences / AVX-512, ~33ms average across benchmarks). No autoregressive token generation overhead.
* **Mathematically Calibrated**: Built-in post-hoc temperature scaling fitted via strictly proper scoring rules. Minimizes Expected Calibration Error (**ECE < 2.5%**). Predicted confidence matches empirical accuracy.
* **Massive Cross-Lingual Coverage**: Native support across **1,800+ languages** utilizing `jhu-clsp/mmBERT-base` with zero-dependency heuristic script routing. 313M total parameters (150M-class ModernBERT transformer core + 256k-token multilingual embedding table).
* **8,192-Token Architectural Context**: RoPE-based rotary positional embeddings and dynamic ONNX axes support sequences up to **8,192 tokens** (fine-tuned up to 1,024 tokens).
* **Autonomous Escalation Head (`Act/Escalate`)**: Built-in binary routing logit that autonomously flags low-confidence or high-risk inputs for human review or fallback to heavier frontier models.
* **Drop-in CPU Serving**: Optimized FP32 and INT8 ONNX graphs with dynamic quantization and a standalone FastAPI + aesthetic web dashboard.

---

## 📊 Benchmark & Evaluation Results

Evaluated on held-out test splits after full 150K multi-task multilingual training with Regularized Log-Loss Calibration Distillation (RLCD):

| Benchmark / Task | Language & Primitive | Accuracy | Expected Calibration Error (ECE) | Brier Score |
| :--- | :--- | :---: | :---: | :---: |
| **`belebele_es`** | Spanish (4-way QA) | **`89.0%`** | **`0.0447`** *(4.5%)* | **`0.1608`** |
| **`belebele_ar`** | Arabic (4-way QA) | **`83.5%`** | **`0.0342`** *(3.4%)* | **`0.2362`** |
| **`belebele_hi`** | Hindi (4-way QA) | **`73.5%`** | **`0.0786`** *(7.9%)* | **`0.3452`** |
| **`xnli_de`** | German (Boolean NLI) | **`87.0%`** | **`0.0577`** *(5.8%)* | **`0.2153`** |
| **`xnli_fr`** | French (3-way NLI) | **`79.0%`** | **`0.0881`** *(8.8%)* | **`0.3259`** |
| **OVERALL AGGREGATE** | **Cross-Lingual Held-Out** | **`82.4%`** | **`0.0188`** *(1.9%)* | **`0.2567`** |

### Verified Post-Hoc Calibration Temperatures
Fitted via strictly proper scoring rules on independent held-out calibration partitions:
* $T[\text{choice} \mid K=4] = \mathbf{0.8528}$
* $T[\text{choice} \mid K=3] = \mathbf{1.8290}$
* $T[\text{noul} \mid K=2] = \mathbf{1.0408}$
* $T[\text{default}] = \mathbf{1.0000}$

*All temperature scaling factors are closely grouped around 1.0, proving natural calibration without logit explosion or collapse.*

---

## ⚡ Quickstart

### Installation

Install via PyPI:

```bash
pip install zesyde
```

Or install from source for development:

```bash
git clone https://github.com/zoebchhatriwala/Zesyde.git
cd Zesyde
pip install -e .
```

### 1. Python SDK (3-Line Inference)

```python
from zesyde.router import ZesydeRouter

# Load directly from Hugging Face Hub (or local directory "models")
router = ZesydeRouter.from_pretrained("zoebchhatriwala/Zesyde-150M-Multilingual-Router", prefer_fp32=False)

# Make a calibrated, fast routing decision
result = router.decide(
    context="The new flagship battery lasts for over 3 days on a single charge.",
    question="What is the sentiment of this review?",
    options=["Negative", "Neutral", "Positive"]
)

print(f"Selected: {result.selected_option}")       # "Positive"
print(f"Confidence: {result.confidence:.3f}")        # 0.982
print(f"Should Escalate: {result.should_escalate}")  # False
print(f"Latency: {result.latency_ms:.1f}ms")         # ~12.4ms (short sequence)
```

---

### 2. Autonomous Escalation & Risk Routing

Zesyde includes a dedicated `Act/Escalate` decision head that triggers when input ambiguity is high:

```python
decision = router.decide(
    context="Ambiguous text or high-risk medical prompt...",
    question="Determine action plan:",
    options=["Auto-Approve", "Reject"]
)

if decision.should_escalate:
    # Route to human review queue or heavier reasoning LLM (e.g. Gemini / Claude)
    escalate_to_human_reviewer(decision)
else:
    # Safely execute automated decision
    execute_action(decision.selected_option)
```

---

### 3. REST API & Web Dashboard

Launch the FastAPI production server:

```bash
python scripts/api.py --model-dir models --port 8000
```

* **Interactive Web Dashboard**: Navigate to `http://localhost:8000/` for a dark-mode, real-time testing playground.
* **REST Endpoint**:

```bash
curl -X POST http://localhost:8000/decide \
  -H 'Content-Type: application/json' \
  -d '{
    "context": "El nuevo smartphone es increíble, la batería dura tres días seguidos.",
    "question": "What is the sentiment?",
    "options": ["Negative", "Neutral", "Positive"]
  }'
```

---

## 💻 Developer & Integration Guide

This guide walks you through building applications, AI agents, and production services using Zesyde.

### 1. Acquiring Model Weights

Zesyde models can be loaded directly from Hugging Face Hub or pre-downloaded locally:

* **Automatic Download (Recommended)**: Pass the Hugging Face repo ID directly to `from_pretrained()`:
  ```python
  router = ZesydeRouter.from_pretrained("zoebchhatriwala/Zesyde-150M-Multilingual-Router")
  ```
  This automatically downloads, verifies, and caches the ONNX models and calibration factors in your local Hugging Face cache.

* **Manual Download (Air-gapped / Offline)**:
  ```bash
  pip install huggingface_hub
  huggingface-cli download zoebchhatriwala/Zesyde-150M-Multilingual-Router --local-dir models/
  ```
  Then initialize pointing to your local directory:
  ```python
  router = ZesydeRouter.from_pretrained("models")
  ```

---

### 2. Core API Reference

#### Initializing the Router

```python
from zesyde.router import ZesydeRouter

router = ZesydeRouter.from_pretrained(
    model_dir_or_path="models",  # Path containing ONNX graphs and calibration JSON
    prefer_fp32=False,           # False uses INT8 (~12ms CPU latency, 318MB RAM)
    max_length=8192,             # Architectural context window (supports up to 8k tokens)
    precision_routing=True       # Dynamically routes ordinal 'score' queries to FP32 if available
)
```

#### Making Decisions: `router.decide()`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `context` | `str` | *required* | The prompt, user message, or document context (up to 8,192 tokens). |
| `options` | `List[str]` | *required* | Candidate choices (2 to 20 arbitrary options dynamically scored). |
| `question` | `str` | `"Select the best option:"` | The decision question or routing objective. |
| `question_type` | `str` | `"choice"` | Decision primitive: `'choice'` (multi-class), `'noul'` (boolean), or `'score'` (ordinal). |
| `escalation_threshold`| `float` | `0.10` | Binary risk threshold above which `should_escalate` triggers. |

#### Decision Output: `DecisionResult`

| Property | Type | Description |
| :--- | :--- | :--- |
| `selected_option` | `str` | The winning candidate option. |
| `confidence` | `float` | Mathematically calibrated probability ($0.0 \dots 1.0$) with Expected Calibration Error $< 2.5\%$. |
| `probabilities` | `Dict[str, float]` | Full probability distribution across all provided candidate options. |
| `should_escalate` | `bool` | True if ambiguity/risk exceeds `escalation_threshold`. |
| `escalate_probability`| `float` | Raw scalar probability of epistemic risk predicted by the `Act/Escalate` head. |
| `latency_ms` | `float` | Inference latency in milliseconds (typically 12–35ms on CPU). |

---

### 3. Developer Recipes & Design Patterns

#### Recipe A: AI Agent Fast Tool Selection (LangChain / LlamaIndex / Custom)
Instead of asking an autoregressive LLM to select tools (costing money and adding ~1,000ms latency), route locally in 15ms:

```python
from zesyde.router import ZesydeRouter

router = ZesydeRouter.from_pretrained("models", prefer_fp32=False)

tools = ["search_web", "query_database", "send_email", "calculate_mortgage"]

decision = router.decide(
    context="User message: 'Can you look up the current stock price of Apple?'",
    question="Select the optimal tool to fulfill this request:",
    options=tools
)

print(f"Tool selected: {decision.selected_option} (confidence: {decision.confidence:.2%})")
# Executes: search_web
```

#### Recipe B: Cost & Latency Gatekeeper (The LLM Cascade)
Route routine queries to instant automated handling, while reserving expensive frontier models (GPT-4o, Claude 3.5 Sonnet) only for edge cases:

```python
decision = router.decide(
    context=incoming_customer_ticket,
    question="Categorize inquiry type:",
    options=["Password Reset", "Billing Question", "Feature Request", "Complex Legal Issue"]
)

# If Zesyde detects ambiguity or low confidence, escalate to a heavy reasoning model
if decision.should_escalate or decision.confidence < 0.75:
    response = call_frontier_llm(incoming_customer_ticket)
else:
    response = handle_automated_workflow(decision.selected_option)
```

#### Recipe C: Multilingual Script & Language Routing
For global applications across 1,800+ languages, use `LanguageRouter` for microsecond script detection and dynamic cross-lingual dispatch:

```python
from zesyde.lang_router import LanguageRouter

lang_router = LanguageRouter(model_dir="models")

# Automatically detects Spanish, Arabic, Hindi, German, etc.
result = lang_router.decide(
    context="El servicio al cliente fue excepcional y resolvieron mi problema rápidamente.",
    question="¿Cuál es el sentimiento del cliente?",
    options=["Negativo", "Neutral", "Positivo"]
)

print(f"Detected Language: {result.detected_language}")  # 'es'
print(f"Selected: {result.selected_option}")              # 'Positivo'
```

#### Recipe D: Binary Verification / Fact-Checking (`noul`)
Use the `noul` primitive for calibrated boolean verification (strictly 2 options):

```python
result = router.decide(
    context="Statement: The capital of Australia is Canberra.",
    question="Is this factual statement accurate?",
    options=["No", "Yes"],
    question_type="noul"
)

print(f"True? {result.selected_option == 'Yes'} (Confidence: {result.confidence:.3f})")
```

#### Recipe E: Ordinal Ratings & Scoring (`score`)
Preserve monotonic distance across Likert scales or priority ratings:

```python
result = router.decide(
    context="User comment: The product broke within 2 hours of use.",
    question="Rate customer satisfaction on a scale from 1 to 5:",
    options=["1", "2", "3", "4", "5"],
    question_type="score"
)

print(f"Satisfaction Rating: {result.selected_option} / 5")
```

---

### 4. Performance Tuning & CPU Optimization

* **Multi-threading**: By default, `ZesydeRouter` sets `intra_op_num_threads = 4`, which yields the lowest batch-1 latency on modern CPUs. On small or containerized instances with 1–2 cores, you can adjust this directly in the ONNX session options.
* **FP32 vs INT8**:
  * **INT8** (`prefer_fp32=False`): Model file is **318 MB**. Delivers ~12–35ms latency and lowest RAM footprint.
  * **FP32** (`prefer_fp32=True`): Model file is **1.2 GB**. Delivers exact mathematical parity with zero quantization noise, with minimal latency difference (~13ms on AVX-512).

---

## 🏗️ Model Architecture

```text
Sequence Layout:
[CLS] context [SEP] question [SEP] [MASK] Option 1 [MASK] Option 2 ... [MASK] Option K [SEP]
                                      │                  │                   │
                                      ▼                  ▼                   ▼
                          Candidate Mean-Pooling  Candidate Mean-Pooling  Candidate Mean-Pooling
                                      │                  │                   │
                                      └──────────────────┼───────────────────┘
                                                         ▼
                                                Context Fusion Layer
                                          [h_opt, h_cls, h_opt ⊙ h_cls]
                                                         ▼
                                            Calibrated Decision Head
                                                         ▼
                                         Temperature Scaling Matrix T[Type|K]
                                                         ▼
                                          Calibrated Probability Vector
```

1. **Backbone**: Pretrained `jhu-clsp/mmBERT-base` (multilingual) or `answerdotai/ModernBERT-large` (English).
2. **Context Fusion**: Candidate option representations $\mathbf{h}_{\text{opt}}$ are concatenated with the sequence representation $\mathbf{h}_{\text{cls}}$ and their interaction $\mathbf{h}_{\text{opt}} \odot \mathbf{h}_{\text{cls}}$, yielding a $3H$-dimensional feature vector per option.
3. **Decision Head**: Regularized multi-layer perceptron with `GELU` and `LayerNorm` predicting unconstrained scalar compatibility logits.
4. **Escalation Head**: Binary classifier over $\mathbf{h}_{\text{cls}}$ predicting escalation probability.
5. **Temperature Scaling**: Post-hoc isotonic temperature table $T[\text{type} \mid K]$ learned on held-out calibration partitions.

---

## 🛠️ Reproduction & Training Pipeline

To reproduce the 128K production build end-to-end (supervised warm-start, RLCD distillation, temperature calibration, and INT8 ONNX export):

```bash
python scripts/build_prod_model.py \
  --multilingual \
  --train-limit 128000 \
  --epochs 1 \
  --batch-size 16 \
  --max-length 1024
```

### Mid-Training Pause & Resumption
Training can be paused safely at any point:
* Press **`p`** in the terminal to pause, save an in-memory or disk checkpoint, and quit.
* Press **`Ctrl+C`** for graceful signal capture to save `models/checkpoints/train_checkpoint.pt`.
* Resume training seamlessly with:
  ```bash
  python scripts/build_prod_model.py --resume
  ```

---

## 📦 Export Artifacts

Running the pipeline outputs complete serving artifacts to `models/`:
* `zesyde_prod_int8.onnx`: Dynamically quantized 8-bit ONNX model for CPU inference (~318 MB / ~95 MB non-embedding weights).
* `zesyde_prod_fp32.onnx`: Full-precision 32-bit ONNX model (~1.16 GB).
* `zesyde_calibration.json`: Persisted temperature scaling lookup table.
* `zesyde_trained.pt`: Full PyTorch model checkpoint (~1.17 GB, 313.1M parameters).

---

## 📜 Citation & License

```bibtex
@software{zesyde2026,
  author = {Chhatriwala, Zoeb},
  title = {Zesyde: Calibrated, Non-Autoregressive Decision & Routing Foundation Models},
  year = {2026},
  version = {1.1.0},
  url = {https://huggingface.co/zoebchhatriwala/Zesyde-150M-Multilingual-Router}
}
```

Licensed under the [MIT License](LICENSE).
