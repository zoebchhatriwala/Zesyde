# Zesyde Sentinel: Real-Time Multilingual Trust & Safety Firewall (`demos/live_sentinel`)

An enterprise-grade, real-world demonstration of **Zesyde Foundation Router** running on **live external data** (Wikimedia Global EventStreams SSE). 

Demonstrates how platforms can triage **thousands of live multilingual events per minute** in **15–25ms on pure CPU**, with mathematically calibrated certainty and autonomous escalation.

---

## 🎯 Why This Answers LinkedIn Feedback

| Skeptic's Objection | How Zesyde Sentinel Proves the Enterprise Solution |
| :--- | :--- |
| *"Nice game/toy demo, but where's the real world case?"* | Content moderation, P0 incident triage, and brand safety are multi-million dollar problems for social platforms, marketplaces, wikis, and enterprise knowledge bases. |
| *"Is it tested on real live data?"* | **100% genuine, unfiltered real-time data** streaming live from Wikimedia's global firehose across 50+ languages (English, German, Spanish, Japanese, Arabic, French, etc.). |
| *"Why not just use GPT-4o or Claude 3.5?"* | A generative LLM takes **800–2,000ms** per event and costs **$5,000–$25,000/month** at high QPS. Zesyde runs in **~18ms on a standard CPU with $0 GPU bills** and has an Expected Calibration Error (ECE) < 2.5%. |
| *"How do you handle ambiguous cases?"* | Zesyde features a **native autonomous escalation head (`Act/Escalate`)**. High-confidence edits are approved or quarantined automatically; ambiguous or tight-margin cases are instantly escalated to human review. |

---

## 🏗️ Architecture

```
                                  ┌──────────────────────────────────────────────┐
                                  │      Wikimedia Global EventStreams SSE       │
                                  │   (50+ languages, 50-200 edits per second)   │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                               Async SSE Ingestion
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │         Zesyde Foundation Router             │
                                  │   - INT8 ONNX Runtime (Pure CPU)             │
                                  │   - 150M ModernBERT + mmBERT Embeddings      │
                                  │   - Out-of-sample Temperature Scaling        │
                                  │   - Latency: ~18ms per decision              │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                         ┌───────────────────────────────┴───────────────────────────────┐
                         ▼                                                               ▼
        ┌───────────────────────────────────┐                         ┌───────────────────────────────────┐
        │       Autonomous Decisions        │                         │      Calibrated Escalation        │
        │   - Auto-Approve (Clean content)  │                         │   - Low confidence (< 45%)        │
        │   - Auto-Quarantine (Vandalism)   │                         │   - Ambiguous top margin (< 12%)  │
        │   (Committed in < 25ms on CPU)    │                         │   - Routed to Human Moderator     │
        └───────────────────────────────────┘                         └───────────────────────────────────┘
```

---

## ⚡ Quickstart

### 1. Launch the Server

Run from the project root:

```bash
# Launch server on port 8085
python demos/live_sentinel/run.py --port 8085
```

### 2. Open in Browser

Navigate to **`http://127.0.0.1:8085`**

* **Watch the Live Waterfall**: Real edits from around the world stream in with live calibrated classifications and latency telemetry.
* **Inspect Decisions**: Click any event card to view the exact candidate probability distribution and escalation logit.
* **Filter by Language**: Filter between All, English, Spanish, German, French, Japanese, Arabic.
* **Test 1-Click Presets**: Test real multilingual challenge cases (Einstein vandalism, German casino spam, French election ambiguity, etc.).
* **Interactive Sandbox**: Enter any custom text or comment in any language to test sub-30ms CPU inference.
