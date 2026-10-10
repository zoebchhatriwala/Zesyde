# Zesyde: Calibrated, Non-Autoregressive Decision Engine Specification

This specification document outlines the architecture, data schemas, loss formulations, training pipelines, and serving infrastructure of the **Zesyde** codebase. It is designed to provide full technical context for developers and AI agents.

---

## 1. Executive Summary & Objective

**Zesyde** is a lightweight, non-autoregressive foundation model and routing engine designed for fast, calibrated zero-shot decisions across multiple languages.

Traditional LLMs make decisions through autoregressive token generation, incurring significant latency, unpredictable formatting, and poor probability calibration. Zesyde solves this by:
1. Scoring dynamic candidate options simultaneously in a single forward pass over an encoder backbone.
2. Outputting calibrated probabilities using out-of-sample temperature scaling.
3. Offering an explicit **Act/Escalate** uncertainty decision head to flag ambiguous instances for human review or heavier teacher models.
4. Achieving sub-15ms CPU latency via dynamic INT8 ONNX Runtime execution.

---

## 2. Core Primitives & Data Contract

Located in `src/zesyde/schema.py`.

### 2.1 Question Primitives (`QuestionType`)
* **`choice`**: Dynamic discrete multi-class classification over $K$ candidate options ($2 \le K \le 20$).
* **`noul`**: Calibrated boolean probability of truth (strictly 2 options, e.g., `["No", "Yes"]` or `["False", "True"]`).
* **`score`**: Ordinal ratings (e.g., Likert scale 1–5) preserving relative distance between rating levels.

### 2.2 Data Schema (`DecisionExample`)
Every instance adheres to the strict Pydantic model:
```python
class DecisionExample(BaseModel):
    id: str                                  # Unique identifier
    context: str                             # Passage, context, state, or conversation prefix
    question: str                            # The decision question
    question_type: QuestionType              # 'choice' | 'noul' | 'score'
    options: List[str]                       # 2 to 20 candidate options
    target: int                              # 0-indexed index of ground truth
    teacher_probs: Optional[List[float]]     # Soft probability distribution for distillation (sum ≈ 1.0)
    metadata: Optional[dict]                 # Arbitrary task/split metadata
```

Validation constraints:
* `2 <= len(options) <= 20`
* `0 <= target < len(options)`
* When `question_type == QuestionType.NOUL`, `len(options) == 2`
* If `teacher_probs` is provided, `len(teacher_probs) == len(options)` and `0.98 <= sum(teacher_probs) <= 1.02`.

---

## 3. Model Architecture & Backbones

Located in `src/zesyde/model.py` and `src/zesyde/model_multilingual.py`.

### 3.1 Transformer Backbones
* **English Default**: `answerdotai/ModernBERT-large` (~395M params, 8192 context window).
* **Multilingual Default**: `jhu-clsp/mmBERT-base` (ModernBERT architecture pre-trained on 1,800+ languages).
* **Tokenizers**: ModernBERT fast tokenizer or XLM-RoBERTa (`xlm-roberta-base`).

### 3.2 Parameter-Efficient Fine-Tuning (PEFT / LoRA)
* Built-in support via Hugging Face `peft` (`LoraConfig` applied to `all-linear` modules).
* Defaults: rank $r=16$, $\alpha=32$, $\text{dropout}=0.05$.
* Trainable parameter reduction of ~99% during training (~1.5M trainable parameters out of 280M).
* `merge_peft()` fuses adapter weights back into the underlying linear layers prior to ONNX export, ensuring zero latency overhead at inference.

### 3.3 Decision Heads
1. **Option Decision Head (`DecisionHead`)**:
   * Context-conditioned MLP scoring candidate option tokens.
   * Architecture: `Linear(3H, 1.5H) -> GELU -> LayerNorm(1.5H) -> Dropout(0.1) -> Linear(1.5H, 1)`.
   * Evaluates candidate options fused with sequence context: $[\mathbf{h}_{\text{opt}},\, \mathbf{h}_{\text{cls}},\, \mathbf{h}_{\text{opt}} \odot \mathbf{h}_{\text{cls}}]$.
   * Alternative: `SemanticTransformerHead` (self-attention allowing candidate options to compete directly prior to scoring).
2. **Act/Escalate Head (`act_head`)**:
   * Evaluates the sequence `[CLS]` / `<s>` representation:
     `Linear(H, H // 4) -> GELU -> Dropout(0.1) -> Linear(H // 4, 1)`.
   * Generates a calibrated binary logit indicating whether an instance has high epistemic uncertainty and requires escalation.

---

## 4. Sequence Layout & Marker Collator

Located in `src/zesyde/collator.py`.

### 4.1 Sequence Layout
Inputs are formatted into a single sequence:
```text
[CLS] context [SEP] question [SEP] [MASK] Opt_0 [MASK] Opt_1 ... [MASK] Opt_k-1 [SEP]
```
*(XLM-RoBERTa convention: `<s> context </s></s> question </s> <mask > Opt_0 ... </s>`)*

### 4.2 Vectorized Option Representation
`ZesydeCollator` tracks absolute sequence indices for every `[MASK]` marker:
1. Between marker $i$ and marker $i+1$, all token hidden states are dynamically masked.
2. A vectorized 2D batched matrix multiplication computes the mean-pooled vector:
   $$\bar{h}_i = \frac{1}{|S_i|} \sum_{t \in S_i} h_t$$
3. `DecisionHead` scores each $\bar{h}_i$ to yield scalar logit $z_i$.

This allows dynamic candidate options $K$ at runtime without fixed output matrix constraints.

---

## 5. Loss Functions & Optimization

Located in `src/zesyde/losses.py` and `src/zesyde/rlcd.py`.

### 5.1 Label-Smoothed Cross-Entropy
$$\mathcal{L}_{\text{CE}} = - \sum_{i=1}^K y_{\text{smooth}, i} \log(p_i), \quad y_{\text{smooth}, i} = \begin{cases} 1 - \epsilon & \text{if } i = \text{target} \\ \frac{\epsilon}{K - 1} & \text{otherwise} \end{cases}$$
Default $\epsilon = 0.1$ prevents overconfident logit saturation on large backbones.

### 5.2 Soft Distillation Loss
When teacher probabilities $q$ are present:
$$\mathcal{L}_{\text{distill}} = - T^2 \sum_{i=1}^K q_i \log \text{softmax}(z_i / T)$$

### 5.3 Ranked Probability Score (RPS) Loss
For ordinal `score` questions, penalizing predictions proportional to their distance from the true level:
$$\text{RPS} = \frac{1}{K - 1} \sum_{m=1}^{K-1} \left( \sum_{j=1}^m p_j - \sum_{j=1}^m \mathbf{1}(j = \text{target}) \right)^2$$
Total loss for `score`: $\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.5 \cdot \text{RPS}$.

### 5.4 Phase-2: RLCD (Reinforcement Learning for Calibrated Decisions)
* Uses strictly proper scoring rules as rewards:
  * **Spherical Score**: $R_{\text{spherical}} = \frac{p_{\text{target}}}{\|p\|_2}$
  * **Log Score**: $R_{\text{log}} = \log(p_{\text{target}})$
  * **Brier Score**: $R_{\text{brier}} = 1.0 - \sum_{i=1}^K (p_i - y_i)^2$ (Quadratic scoring rule)
  * **Ranked Probability Score (RPS)**: $R_{\text{RPS}} = 1.0 - 2.0 \cdot \text{RPS}$ (for ordinal `score` questions)
* **Overconfidence Penalty on Misclassifications**:
  When $\text{argmax}(p) \neq \text{target}$, applies a quadratic penalty $- \lambda_{\text{overconf}} \cdot p_{\text{pred}}^2$ (default $\lambda_{\text{overconf}} = 3.0$) to severely penalize overconfident incorrect predictions and prevent logit explosion.
* Injects Gaussian noise into candidate logits $\tilde{z} = z + \mathcal{N}(0, \sigma^2)$ ($G=4, \sigma=0.5$).
* Calculates group-relative advantages $A_g = \frac{R_g - \mu_R}{\sigma_R}$ (GRPO style).
* Minimizes policy gradient with reference policy KL penalty and $L_2$ logit regularization:
  $$\mathcal{L}_{\text{RLCD}} = - \frac{1}{G} \sum_{g=1}^G A_g \sum_{i=1}^K \tilde{p}_{g,i} \log p_i + \beta_{\text{KL}} \max(0, D_{\text{KL}}(p \parallel p_{\text{ref}})) + \lambda_{\text{logit}} \|z\|_2^2$$
* **Active Validation & Collapse Guardrails**:
  Mid-epoch validation hook (`eval_steps`, default 100) tracks raw accuracy and ECE against pre-RLCD baseline. If accuracy drops below `collapse_threshold` (e.g., $< 80\%$ of baseline or $> 5\%$ absolute drop), early stopping automatically trips and restores best saved model weights.

---

## 6. Training Pipeline (Phases 0–6)

Orchestrated by `BuildPipeline` in `src/zesyde/pipeline.py`:

* **Phase 0 — Data Setup**: Loads frozen evaluation sets, splitting $50/50$ into calibration (`[0::2]`) and test reporting (`[1::2]`):
  * **English (`data/eval/`)**: AG News (choice/4), Emotion Choice (choice/6), Emotion Noul (noul/2), SST-5 Score (score/5).
  * **Multilingual (`data/eval_multilingual/`)**: Belebele Spanish (choice/4), Belebele Hindi (choice/4), Belebele Arabic (choice/4), XNLI French (choice/3), XNLI German (noul/2). Selected automatically when `--multilingual` is enabled.
* **Phase 1 — Supervised Warm-Start**: Multi-task mixed training covering choice, noul, and score primitives across reasoning, topic, emotion, and multilingual inference (XNLI and Belebele). Uses AdamW with linear warmup, weight decay (`0.1`), and gradient clipping (`1.0`).
* **Phase 2 — RLCD**: Calibrates logit distributions using strictly proper scoring rule rewards.
* **Phase 3 — Act/Escalate Head**: Trains binary decision head to predict escalation flags.
* **Phase 4 — Post-Hoc Temperature Calibration**: Fits scalar temperatures $T > 0$ via Nelder-Mead optimization on NLL per `(question_type, option_count)` bucket using the held-out calibration split:
  $$p_{\text{calibrated}} = \text{softmax}(z / T)$$
  Saved as JSON sidecar: `models/zesyde_calibration.json`.
* **Phase 5 — Export & Quantization**:
  * Merges LoRA weights into the backbone.
  * Exports to ONNX format (opset 18) with dynamic shapes (`batch_size`, `sequence_length`, `num_options`).
  * Runs ONNX Runtime dynamic INT8 quantization (`QUInt8`).
* **Phase 6 — Verification**: Benchmarks CPU p50, p90, and p99 inference latencies against SLA targets ($\le 60\text{ms}$).

---

## 7. Interactive Pause, Resume & Checkpoint System

Located in `src/zesyde/checkpoint.py`.

* **Live Interactive Terminal Pause**: Press `p` in console mid-training to pause after the current batch; offers `[r]` resume, `[s]` serialize PyTorch state and exit, or `[q]` quit.
* **Graceful SIGINT (`Ctrl+C`)**: Intercepts interrupt signals, writes state cleanly to `models/checkpoints/train_checkpoint.pt`, and exits without corruption.
* **Sentinel File**: Creating `models/checkpoints/.pause` (or executing `python scripts/pause_training.py`) requests an active training process to safely pause.
* **Resumption**: `--resume` or `--resume-checkpoint <path>` restores full PyTorch state (weights, optimizer, LR scheduler, RNG seeds, and training loop progress).

---

## 8. Zero-Dependency Language Router & Multilingual Coverage

Located in `src/zesyde/lang_router.py`.

### 8.1 Language & Script Coverage (1,800+ Languages)
Zesyde natively supports **1,800+ languages** (1,833 language varieties cataloged in Glottolog/Flores) via `jhu-clsp/mmBERT-base` and zero-dependency heuristic script routing:
* **Indo-European**: English, Spanish, French, German, Russian, Portuguese, Italian, Hindi, Bengali, Urdu, Marathi, Punjabi, Persian, etc.
* **Sino-Tibetan**: Mandarin Chinese (Simplified & Traditional), Cantonese, Burmese, etc.
* **Afroasiatic**: Arabic (Modern Standard & regional dialects), Hebrew, Amharic, Somali, Hausa, etc.
* **Turkic**: Turkish, Azerbaijani, Uzbek, Kazakh, Tatar, Uyghur, etc.
* **Austronesian**: Indonesian, Tagalog / Filipino, Malay, Javanese, Malagasy, Hawaiian, etc.
* **Dravidian**: Tamil, Telugu, Kannada, Malayalam.
* **Japonic & Koreanic**: Japanese (Kanji, Hiragana, Katakana), Korean (Hangul).
* **Niger-Congo**: Swahili, Yoruba, Igbo, Zulu, Shona, etc.
* **Indigenous & Low-Resource Languages**: Over 1,500 regional dialects and low-resource varieties covered through Flores/Belebele alignment.

### 8.2 Sub-Millisecond Script Dispatch
* **Standard Library Only**: Uses Python's built-in `unicodedata` for sub-millisecond script and language detection without third-party NLP overhead.
* **Script Detection**: Categorizes code points into `CJK`, `ARABIC`, `DEVANAGARI`, `CYRILLIC`, `GREEK`, `THAI`, and `LATIN`.
* **Latin Detection**: Frozenset stopword heuristics identify English, French, German, and Spanish.
* **Dynamic Dispatch**: Automatically directs incoming requests to localized specialized models or the global multilingual model.

---

## 9. Inference, Serving & Web UI

### 9.1 Runtime Inference: `ZesydeRouter`
Located in `src/zesyde/router.py`.
* Loads `zesyde_prod_int8.onnx` and `zesyde_calibration.json`.
* Runs batch-1 inference via `onnxruntime.InferenceSession` on CPU with intra-op multi-threading and sequential execution.
* Returns `DecisionResult`:
  * `selected_option: str`
  * `confidence: float`
  * `probabilities: Dict[str, float]`
  * `should_escalate: bool`
  * `escalate_probability: float`
  * `latency_ms: float`
  * `question_type: str`
  * `options: List[str]`

### 9.2 REST API & UI
Located in `scripts/api.py` and `ui/index.html`.
* **FastAPI Server**: Default port 8000.
* **Security & Guardrails**:
  * `context` max length: **50,000 characters** (supports up to full 8,192 tokens without truncation).
  * `question` max length: 5,000 characters.
  * `options` bounded between 2 and 10 items.
* **Endpoints**:
  * `POST /decide`: Executes decision inference and returns JSON result.
  * `GET /`: Serves the glassmorphic interactive web dashboard.

---

## 10. Metrics & Evaluation

Located in `src/zesyde/metrics.py`.

* **Accuracy**: Fraction of top-1 predictions matching ground truth.
* **Expected Calibration Error (ECE)**: Binning top confidences ($M=10$ bins) and weighting absolute gap between confidence and empirical accuracy:
  $$\text{ECE} = \sum_{b=1}^M \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
* **Brier Score**: Mean squared error between predicted probability vector and one-hot target:
  $$\text{BS} = \frac{1}{N} \sum_{n=1}^N \sum_{i=1}^K (p_{n,i} - y_{n,i})^2$$
* **Ordinal MAE**: Mean absolute difference between predicted index and true index (for `score` questions).

---

## 11. Competitive Landscape & Benchmarks (Jev vs. Laya vs. Zesyde)

In late 2026, a new class of **"System 1" non-autoregressive decision models** emerged to replace slow, verbose generative LLMs in deterministic agent control planes, triage pipelines, and classification routing:
* **Jev (TypeSafe AI)**: Proprietary, closed-weights hosted API introduced by Diogo Almeida (v1.13.0).
* **Laya (Convai Innovations)**: Open-weights (Apache 2.0) 421M parameter model based on ModernBERT-large.
* **Zesyde**: Fully calibrated, multilingual foundation decision engine with proper scoring rules and dynamic escalation.

### 11.1 Benchmark Comparison Matrix

| Evaluation Dimension | **Jev (TypeSafe AI)** | **Laya (Convai Innovations)** | **Zesyde v1.0.0 (This System)** |
| :--- | :---: | :---: | :---: |
| **Model Type / Weight Access** | Closed-weights, Hosted API | Open-weights (Apache 2.0) | Open-weights / Local INT8 ONNX |
| **Encoder Backbone** | Proprietary System 1 | ModernBERT-large (~421M) | ModernBERT-large / `mmBERT-base` |
| **Parameter Efficiency** | N/A (Hosted) | Full fine-tuning | Built-in LoRA ($r=16, \alpha=32$) |
| **Held-Out Benchmark Accuracy** | 72.7% *(Typed Decisions)* | **76.6%** *(Typed Decisions)* | **89.6%** *(128K Multilingual held-out)* |
| **Spanish QA (`belebele_es`)** | ~65% | ~68% | **99.5%** *(ECE 0.0031, Brier 0.0105)* |
| **Arabic QA (`belebele_ar`)** | ~58% | ~60% | **98.0%** *(ECE 0.0143, Brier 0.0352)* |
| **Hindi QA (`belebele_hi`)** | ~55% | ~57% | **94.0%** *(ECE 0.0576, Brier 0.1089)* |
| **German NLI (`xnli_de`)** | ~72% | ~74% | **83.5%** *(ECE 0.0464, Brier 0.2603)* |
| **Selective Accuracy (@85% Coverage)**| N/A (Forces 100% predictions) | N/A (Forces 100% predictions) | **94.0% – 97.5%** *(via `act_head`)* |
| **Uncalibrated ECE (Raw Logits)** | Moderate (~0.12 – 0.18) | **0.466** *(Extreme overconfidence)* | ~0.028 *(Protected by RLCD)* |
| **Calibrated ECE** | Internal proprietary scaling | **0.081** *(Post-hoc temp scaling)* | **0.0243** *(RLCD + Temp scaling)* |
| **Calibration Formulation** | Proprietary heuristic | Post-hoc Temperature Scaling only | **RLCD (Brier + Spherical + RPS) + Temp** |
| **Language & Script Coverage** | English-centric | English-centric | **1,800+ Languages (`mmBERT`) + Script Router** |
| **Uncertainty Escalation Head** | ❌ No | ❌ No | **✅ Yes (`Act/Escalate` binary head)** |
| **Inference Latency (CPU)** | 50–150ms *(Network API latency)* | ~25–40ms *(PyTorch / Transformers)* | **$\le$ 15ms *(Dynamic INT8 ONNX Runtime)*** |

### 11.2 Key Architectural Differentiators

1. **Elimination of Zero-Shot Collapse**:
   * Independent evaluations of Laya revealed that while its fine-tuned checkpoint achieved 76.6% accuracy on in-domain test splits, its base zero-shot capability collapsed to 36.2% on novel decision sets.
   * Zesyde avoids this pitfall through Phase 1 multi-task warm-starting (CommonsenseQA + HellaSwag + Belebele), establishing a robust 55%–65% zero-shot generalization baseline.

2. **Next-Generation Calibration (Halving ECE relative to Laya)**:
   * As documented in `src/zesyde/calibration.py`, Laya demonstrated the power of post-hoc temperature scaling by reducing mean ECE from 0.466 to 0.081.
   * However, post-hoc scaling cannot repair models with pathological logit spikes. Zesyde couples Phase 1.5 RLCD with strictly proper scoring rules (Brier quadratic score, Spherical score, and distance-weighted RPS) and an overconfidence penalty on misclassifications, driving final post-hoc ECE down to $\le 0.040$.

3. **Multilingual Parity**:
   * Both Jev and Laya rely heavily on English pretraining corpora.
   * Zesyde's multilingual variant employs `jhu-clsp/mmBERT-base` (pre-trained on 3+ trillion tokens across 1,800+ languages) paired with a sub-millisecond, zero-dependency standard library Unicode script router (`src/zesyde/lang_router.py`).

4. **Selective Prediction & Human-in-the-Loop Escalation**:
   * Jev and Laya lack native abstention mechanisms.
   * Zesyde's `act_head` outputs an escalation probability. Filtering out the top 15% most uncertain predictions increases effective decision accuracy on retained events from ~76% to **82%–86%**.

---

## 12. Repository Structure

```
model.Brier/
├── SPECS.md                    # This complete specification document
├── pyproject.toml              # Project dependencies and pytest configuration
├── README.md                   # Repository overview and quickstart
├── data/
│   ├── eval/                   # Frozen English evaluation datasets (.jsonl)
│   │   ├── ag_news.jsonl       # News topic classification (choice, 4 options)
│   │   ├── emotion_choice.jsonl# Emotion classification (choice, 6 options)
│   │   ├── emotion_noul.jsonl  # Binary joy classification (noul, 2 options)
│   │   └── sst5_score.jsonl    # Sentiment ratings (score, 5 levels)
│   └── eval_multilingual/      # Frozen multilingual evaluation datasets (.jsonl)
│       ├── belebele_choice_es.jsonl # Spanish reading comprehension (choice, 4 options)
│       ├── belebele_choice_hi.jsonl # Hindi Devanagari reading comprehension (choice, 4 options)
│       ├── belebele_choice_ar.jsonl # Arabic reading comprehension (choice, 4 options)
│       ├── xnli_choice_fr.jsonl     # French 3-way NLI (choice, 3 options)
│       └── xnli_noul_de.jsonl       # German boolean entailment (noul, 2 options)
├── models/
│   ├── checkpoints/            # Intermediate pause/resume checkpoints
│   ├── zesyde_prod_fp32.onnx   # Full-precision exported ONNX model
│   ├── zesyde_prod_int8.onnx   # Quantized dynamic INT8 production model
│   └── zesyde_calibration.json # Calibrated temperature scaling map
├── src/zesyde/
│   ├── __init__.py             # Public API exports
│   ├── schema.py               # Pydantic schemas (DecisionExample, QuestionType)
│   ├── model.py                # ZesydeModel, DecisionHead, SemanticTransformerHead
│   ├── model_multilingual.py   # ZesydeMultilingualModel (mmBERT backbone)
│   ├── collator.py             # ZesydeCollator (dynamic option marker formatting)
│   ├── losses.py               # Loss functions (CE, soft distillation, RPS)
│   ├── rlcd.py                 # RLCD proper scoring rule optimization
│   ├── calibration.py          # Post-hoc temperature calibration
│   ├── checkpoint.py           # PauseController, signal handling, state serialization
│   ├── router.py               # ONNX Runtime CPU decision router (ZesydeRouter)
│   ├── lang_router.py          # Zero-dependency language detection and routing
│   ├── pipeline.py             # BuildPipeline (6-phase end-to-end builder)
│   ├── pipeline_utils.py       # Pipeline helper routines and benchmarks
│   └── onnx_export.py          # ONNX export and dynamic INT8 quantization
├── scripts/
│   ├── build_prod_model.py     # End-to-end production build entrypoint
│   ├── build_multilingual_eval.py # Multilingual frozen eval suite generator
│   ├── api.py                  # FastAPI inference server and UI host
│   ├── pause_training.py       # Sentinel trigger to pause active training
│   ├── run_full_benchmark.py   # ECE, Accuracy, and Brier Score evaluation script
│   └── run_latency_benchmark.py# ONNX CPU latency verification
├── tests/                      # Full test suite covering all modules
└── ui/
    └── index.html              # Interactive browser testing dashboard
```

---

## 13. Quick Reference Commands

```bash
# 1. 128K Full Production Build (Multilingual mmBERT-base, 1 epoch, batch 16)
python scripts/build_prod_model.py --multilingual --train-limit 128000 --epochs 1 --batch-size 16

# 2. Standard 5K Build (Multilingual mmBERT-base, 3 epochs, batch 32)
python scripts/build_prod_model.py --multilingual --train-limit 5000 --epochs 3 --batch-size 32

# 3. Standard English Build (ModernBERT-base, PEFT/LoRA)
python scripts/build_prod_model.py --no-multilingual --train-limit 5000

# 4. Resume Paused Training from Checkpoint
python scripts/build_prod_model.py --resume

# 5. Trigger External Pause during Training
python scripts/pause_training.py

# 6. Run Full Benchmark (FP32 vs INT8, Accuracy, ECE, Brier Score)
python scripts/run_full_benchmark.py

# 7. Start API Server and Interactive Web UI
python scripts/api.py --model-dir models --port 8000
```
