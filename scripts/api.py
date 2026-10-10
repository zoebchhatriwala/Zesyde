import argparse
import sys
from pathlib import Path

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, conlist
from typing import List, Optional
import uvicorn
from zesyde.lang_router import LanguageRouter

app = FastAPI(title="Zesyde API")
zesyde_router = None
brier_router = None

class DecisionRequest(BaseModel):
    """
    Request model for the Zesyde decision endpoint.
    Supports up to 50,000 characters (accommodating full 8,192-token contexts).
    """
    context: str = Field(..., max_length=50000, description="The textual context to base the decision on (supports up to 8,192 tokens).")
    question: str = Field(..., max_length=5000, description="The question or objective.")
    options: conlist(str, min_length=2, max_length=10) = Field(..., description="List of candidate options (2 to 10).")
    question_type: str = Field(default="choice", description="Type of question (e.g., choice, score).")
    escalation_threshold: Optional[float] = Field(default=0.10, description="Risk threshold above which act/escalate triggers (default: 0.10).")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("ui/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/favicon.ico")
async def favicon():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      <rect width="100" height="100" rx="20" fill="#ff6c37"/>
      <text x="50%" y="50%" font-family="Arial" font-size="60" font-weight="bold" fill="white" dominant-baseline="central" text-anchor="middle">Z</text>
    </svg>'''
    return Response(content=svg, media_type="image/svg+xml")

@app.post("/decide")
async def decide(req: DecisionRequest):
    router = zesyde_router or brier_router
    if router is None:
        raise HTTPException(status_code=500, detail="Model router is not initialized.")
    
    try:
        decision = router.decide(
            context=req.context,
            question=req.question,
            options=req.options,
            question_type=req.question_type,
            escalation_threshold=req.escalation_threshold if req.escalation_threshold is not None else 0.10,
        )
        return decision.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Zesyde Inference API.")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing the ONNX models and calibration JSON.")
    parser.add_argument("--tokenizer", type=str, default="jhu-clsp/mmBERT-base", help="Hugging Face tokenizer to use.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on.")
    parser.add_argument("--int8", action="store_true", default=False, help="Force INT8 session instead of FP32.")
    args = parser.parse_args()

    print(f"Loading Zesyde Model from: {args.model_dir}")
    print(f"Using Tokenizer: {args.tokenizer}")
    print(f"Precision: {'INT8 (Quantized)' if args.int8 else 'FP32 (Full Precision, Recommended)'}")
    
    zesyde_router = LanguageRouter.from_checkpoints(
        multilingual_path=args.model_dir,
        tokenizer_name=args.tokenizer,
        prefer_fp32=not args.int8,
        default_key='multilingual'
    )
    brier_router = zesyde_router
    
    print(f"\nModel loaded successfully! Starting server on http://localhost:{args.port}")
    print(f"Test it with:\ncurl -X POST http://localhost:{args.port}/decide -H 'Content-Type: application/json' -d '{{\"context\": \"The new smartphone battery lasts for 3 days!\", \"question\": \"What is the sentiment?\", \"options\": [\"Negative\", \"Neutral\", \"Positive\"]}}'\n")
    
    uvicorn.run(app, host="127.0.0.1", port=args.port)
