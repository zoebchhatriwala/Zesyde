"""FastAPI Server for Zesyde Tetris Demo.

Exposes game APIs and serves the dark obsidian web client.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from zesyde.router import ZesydeRouter
from demos.tetris.engine import TetrisBoard, TETROMINOES
from demos.tetris.agent import ZesydeTetrisAgent

app = FastAPI(title="Zesyde Tetris Demo")

# Global game session state
board = TetrisBoard()
router_instance: Optional[ZesydeRouter] = None
agent: Optional[ZesydeTetrisAgent] = None
model_precision = "Unknown"


class MoveRequest(BaseModel):
    action: str  # "MOVE_LEFT", "MOVE_RIGHT", "ROTATE", "DROP"


class ApplyRequest(BaseModel):
    target_x: int
    target_rot: int


class DecideRequest(BaseModel):
    state: str
    questions: Dict[str, Any]


def setup_agent(model_dir: str = "models", prefer_fp32: bool = True, tokenizer: str = "jhu-clsp/mmBERT-base"):
    global router_instance, agent, model_precision
    models_path = Path(model_dir)
    model_precision = "FP32 (High Fidelity)" if prefer_fp32 else "INT8 (Fast)"
    router_instance = ZesydeRouter.from_pretrained(
        str(models_path),
        prefer_fp32=prefer_fp32,
        tokenizer_name=tokenizer,
    )
    agent = ZesydeTetrisAgent(router=router_instance)


@app.on_event("startup")
async def startup_event():
    global router_instance
    if router_instance is None:
        setup_agent()


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = Path(__file__).resolve().parent / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/info")
async def get_info():
    return {
        "model_name": "Zesyde 150M-class Foundation Router",
        "precision": model_precision,
        "calibration": {
            "temperature": 0.8528,
            "ece": "< 2.5%",
            "scoring": "Strictly Proper Scoring Rules (RLCD)",
        },
        "mode": "Zero-Gimmicks Autonomous Decision Router",
        "board_size": "10x18",
    }


@app.post("/api/reset")
async def reset_game():
    global board
    board = TetrisBoard()
    return {
        "status": "reset",
        "state": board.to_dict(),
    }


@app.get("/api/state")
async def get_state():
    return {
        "state": board.to_dict(),
    }


@app.post("/api/step")
async def model_step():
    """Runs a full model forward pass to evaluate the piece and drops it."""
    global board, agent
    if agent is None:
        raise HTTPException(status_code=500, detail="Tetris agent not initialized.")

    if board.game_over:
        return {
            "state": board.to_dict(),
            "decision": {"game_over": True},
        }

    decision = agent.decide_piece(board)
    board.apply_placement(decision["target_x"], decision["target_rot"])

    return {
        "state": board.to_dict(),
        "decision": decision,
    }


@app.post("/api/plan")
async def model_plan():
    """Runs model forward pass to pick the target placement, returning the plan for animation."""
    global board, agent
    if agent is None:
        raise HTTPException(status_code=500, detail="Tetris agent not initialized.")

    if board.game_over:
        return {"state": board.to_dict(), "decision": {"game_over": True}}

    decision = agent.decide_piece(board)
    return {
        "state": board.to_dict(),
        "decision": decision,
    }


@app.post("/api/apply")
async def apply_plan(req: ApplyRequest):
    """Applies a previously planned placement."""
    global board
    board.apply_placement(req.target_x, req.target_rot)
    return {
        "state": board.to_dict(),
    }


@app.post("/api/move")
async def human_move(req: MoveRequest):
    """Manual arrow-key move."""
    global board
    if board.game_over:
        return {"state": board.to_dict(), "game_over": True}

    board.apply_action(req.action.upper())
    return {
        "state": board.to_dict(),
    }


@app.post("/api/decide")
async def generic_decide_api(req: DecideRequest):
    """General decision endpoint."""
    global router_instance
    if router_instance is None:
        raise HTTPException(status_code=500, detail="Router not initialized.")

    placement_q = req.questions.get("placement") or req.questions.get("place", {})
    criteria = placement_q.get("criteria", {})
    instructions = placement_q.get("instructions", "Which placement is best? Pick one.")

    options = []
    opt_key_map = {}
    for k, v in criteria.items():
        desc = f"{k}: {v}"
        options.append(desc)
        opt_key_map[desc] = k

    if not options:
        options = ["p0: column 1, leaves 0 hole(s)"]
        opt_key_map[options[0]] = "p0"

    res = router_instance.decide(
        context=req.state,
        question=instructions,
        options=options,
        question_type="choice",
    )

    chosen_key = opt_key_map.get(res.selected_option, "p0")
    total_chars = len(req.state) + len(instructions) + sum(len(o) for o in options)
    tokens = max(1, total_chars // 4)

    return {
        "answers": {
            "placement": {
                "choice": chosen_key,
                "confidence": float(res.confidence),
            }
        },
        "usage": {
            "input_tokens": tokens,
        },
    }
