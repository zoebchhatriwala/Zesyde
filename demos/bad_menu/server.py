"""FastAPI backend for Zesyde Adversarial Menu & Escalation Playground.

Demonstrates out-of-distribution detection, abstention ("None of these"),
ambiguity resolution, and calibrated confidence routing on pure CPU INT8 ONNX.
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
MODELS_PATH = PROJECT_ROOT / "models"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))

router_instance = None
try:
    from zesyde.router import ZesydeRouter
except Exception as e:
    print(f"[Warning] ZesydeRouter import error: {e}")

app = FastAPI(title="Zesyde Adversarial Menu Playground", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PRESET_SCENARIOS = [
    {
        "id": "missing_destination",
        "category": "out_of_distribution",
        "title": "The Missing Destination ('None of These')",
        "description": "User wants a billing invoice refund, but the menu only contains airline, password, and kitchen options.",
        "context": "My credit card was charged twice for monthly subscription invoice #48291. Please refund the duplicate $49.00 payment.",
        "options": [
            "Book an international flight",
            "Reset account security password",
            "Order commercial kitchen supplies",
        ],
        "question": "Select the correct department for this request:",
        "auto_none": True,
        "escalation_threshold": 0.15,
        "ambiguity_threshold": 0.15,
    },
    {
        "id": "overlapping_twins",
        "category": "ambiguity",
        "title": "The Overlapping Twins (Needs Clarification)",
        "description": "User says 'Cancel my reservation' without saying if it's hotel or flight. Two candidates strongly compete.",
        "context": "I need to cancel my booking immediately due to a family emergency. Please process the cancellation right away.",
        "options": [
            "Cancel hotel room reservation",
            "Cancel airline flight booking",
            "Update billing profile and phone number",
        ],
        "question": "Route to the appropriate cancellation desk:",
        "auto_none": False,
        "escalation_threshold": 0.15,
        "ambiguity_threshold": 0.20,
    },
    {
        "id": "vague_input",
        "category": "underspecified",
        "title": "The Underspecified Input (Low Confidence)",
        "description": "User input is extremely vague ('It is broken, fix it'). The router should detect low confidence across all candidates.",
        "context": "It is completely broken and nothing is working. Fix it now.",
        "options": [
            "Submit software bug report with logs",
            "Hardware server chassis replacement (RMA)",
            "Dispute merchant chargeback with credit bank",
        ],
        "question": "Select resolution path:",
        "auto_none": True,
        "escalation_threshold": 0.15,
        "ambiguity_threshold": 0.15,
    },
    {
        "id": "clean_match",
        "category": "high_confidence",
        "title": "The Clean Match (Control Baseline)",
        "description": "User input matches exactly one destination clearly. High confidence, zero ambiguity.",
        "context": "Where in my account settings can I update my debit card number and billing address?",
        "options": [
            "Manage payment methods and saved credit cards",
            "Book an airport shuttle",
            "View privacy policy and terms of service",
        ],
        "question": "Select destination:",
        "auto_none": False,
        "escalation_threshold": 0.15,
        "ambiguity_threshold": 0.15,
    },
    {
        "id": "multilingual_missing",
        "category": "multilingual",
        "title": "Multilingual Out-of-Distribution (Spanish/French/Hindi)",
        "description": "User writes in Spanish requesting a medical appointment, given a financial menu.",
        "context": "Necesito agendar una cita médica urgente con el cardiólogo para mañana por la mañana.",
        "options": [
            "Transferir fondos entre cuentas bancarias",
            "Consultar saldo de tarjeta de crédito",
            "Solicitar préstamo hipotecario residencial",
        ],
        "question": "Seleccione el departamento adecuado:",
        "auto_none": True,
        "escalation_threshold": 0.15,
        "ambiguity_threshold": 0.15,
    },
]


class RouteRequest(BaseModel):
    context: str = Field(..., description="User query or context")
    options: List[str] = Field(..., description="Candidate menu options")
    question: Optional[str] = Field(default="Select the best option:")
    auto_none: bool = Field(default=False, description="Whether to include an explicit 'None of these' candidate")
    none_label: Optional[str] = Field(default="None of these (Escalate to human review)")
    escalation_threshold: float = Field(default=0.15, ge=0.01, le=1.0)
    ambiguity_threshold: float = Field(default=0.15, ge=0.01, le=0.50, description="Margin between Top 1 and Top 2 to trigger clarification")
    prefer_fp32: bool = Field(default=False)


def calculate_entropy(probabilities: List[float]) -> float:
    """Computes normalized Shannon entropy in [0, 1]."""
    k = len(probabilities)
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(max(p, 1e-12)) for p in probabilities if p > 0)
    max_h = math.log(k)
    return round(float(h / max_h), 4) if max_h > 0 else 0.0


@app.get("/")
async def serve_index():
    index_path = APP_DIR / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
        )
    return HTMLResponse("<h1>Zesyde Bad Menu Playground Live</h1>")


@app.get("/api/presets")
async def get_presets():
    return PRESET_SCENARIOS


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "engine": "Zesyde 150M-class Foundation Router",
        "model_loaded": router_instance is not None,
        "target_latency": "15-35ms CPU",
    }


@app.post("/api/route")
async def route_decision(req: RouteRequest):
    if router_instance is None:
        raise HTTPException(status_code=503, detail="ZesydeRouter model not loaded.")

    options = list(req.options)
    none_option_added = False

    # Inject explicit None option if requested
    if req.auto_none and req.none_label:
        if req.none_label not in options:
            options.append(req.none_label)
            none_option_added = True

    if len(options) < 2:
        raise HTTPException(status_code=400, detail="Must provide at least 2 options.")

    t0 = time.perf_counter()
    res = router_instance.decide(
        context=req.context,
        options=options,
        question=req.question or "Select the best option:",
        question_type="choice",
        escalation_threshold=req.escalation_threshold,
    )
    raw_latency = (time.perf_counter() - t0) * 1000.0
    latency_ms = round(res.latency_ms if res.latency_ms > 0 else raw_latency, 2)

    # Sort probabilities descending
    sorted_probs = sorted(res.probabilities.items(), key=lambda x: x[1], reverse=True)
    top1_opt, top1_prob = sorted_probs[0]
    top2_opt, top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0.0)

    prob_values = [p for _, p in sorted_probs]
    entropy = calculate_entropy(prob_values)
    margin = round(top1_prob - top2_prob, 4)

    # Decision Categorization logic
    # 1. Did model explicitly choose "None of these"?
    if none_option_added and top1_opt == req.none_label:
        verdict = "NONE_OF_THESE"
        verdict_title = "🛑 NONE OF THESE (Missing Destination)"
        verdict_desc = f"The model detected that none of the provided choices match the user request with confidence {top1_prob:.1%}. Routing to human review."
        verdict_action = "ESCALATE_FALLBACK"

    # 2. Did the dedicated escalation head fire?
    elif res.should_escalate or res.escalate_probability >= req.escalation_threshold:
        verdict = "ESCALATED"
        verdict_title = "🛑 ESCALATED (Uncertainty Head Fired)"
        verdict_desc = f"Escalation probability ({res.escalate_probability:.1%}) exceeded threshold ({req.escalation_threshold:.1%}). Model abstains from forced choice."
        verdict_action = "ESCALATE_FALLBACK"

    # 3. Are the top two options ambiguous twins?
    elif len(sorted_probs) > 1 and margin < req.ambiguity_threshold and top1_prob < 0.65:
        verdict = "AMBIGUOUS"
        verdict_title = "⚠️ NEEDS CLARIFICATION (Ambiguous Twins)"
        verdict_desc = f"Top options '{top1_opt}' ({top1_prob:.1%}) and '{top2_opt}' ({top2_prob:.1%}) have an ambiguous margin of only {margin:.1%}. Clarification required."
        verdict_action = "ASK_CLARIFICATION"

    # 4. Low confidence floor across all choices (Implicit None)
    elif top1_prob < 0.45:
        verdict = "LOW_CONFIDENCE"
        verdict_title = "❓ LOW CONFIDENCE (Underspecified/Bad Menu)"
        verdict_desc = f"Top choice only achieved {top1_prob:.1%}. Menu lacks an adequate match for this query."
        verdict_action = "PROMPT_USER"

    # 5. Clean, authoritative match
    else:
        verdict = "ROUTE_ACCEPTED"
        verdict_title = f"✅ ROUTE ACCEPTED: {top1_opt}"
        verdict_desc = f"Authoritative match with {top1_prob:.1%} confidence and {margin:.1%} lead margin."
        verdict_action = "EXECUTE_ROUTE"

    return {
        "selected_option": res.selected_option,
        "confidence": round(res.confidence, 4),
        "probabilities": {opt: round(prob, 4) for opt, prob in sorted_probs},
        "verdict": verdict,
        "verdict_title": verdict_title,
        "verdict_desc": verdict_desc,
        "verdict_action": verdict_action,
        "top_margin": margin,
        "entropy": entropy,
        "escalate_probability": round(res.escalate_probability, 4),
        "should_escalate": res.should_escalate,
        "latency_ms": latency_ms,
        "options_evaluated": options,
        "top1_candidate": top1_opt,
        "top2_candidate": top2_opt,
    }


def init_router():
    global router_instance
    try:
        print("[Zesyde] Initializing ZesydeRouter for Bad Menu Lab...")
        router_instance = ZesydeRouter.from_pretrained(
            str(MODELS_PATH),
            prefer_fp32=False,  # INT8 CPU for ultra-low latency
        )
        print("[Zesyde] ZesydeRouter loaded successfully!")
    except Exception as e:
        print(f"[Zesyde Warning] Failed to initialize ZesydeRouter: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adversarial Menu Playground")
    parser.add_argument("--port", type=int, default=8082, help="Port (default: 8082)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    args = parser.parse_args()

    init_router()
    print(f"\n>> ADVERSARIAL MENU PLAYGROUND LIVE ON http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)
