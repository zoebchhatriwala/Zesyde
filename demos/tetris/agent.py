"""Autonomous Tetris Agent powered by Zesyde 150M-class Foundation Router.

Evaluates 10x18 board state, ranks legal candidate placements,
formats the decision context, and dispatches to Zesyde ONNX Runtime.
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Any
from zesyde.router import ZesydeRouter
from demos.tetris.engine import TetrisBoard, TETROMINOES


class ZesydeTetrisAgent:
    """Tetris Agent powered directly by Zesyde 150M-class Foundation Router."""

    def __init__(
        self,
        router: ZesydeRouter,
        escalation_threshold: float = 0.15,
    ):
        self.router = router
        self.escalation_threshold = escalation_threshold

    def format_board_ascii(self, board: TetrisBoard) -> str:
        """Renders 10x18 board into '#' filled and '.' empty cells."""
        lines = []
        for row in board.grid:
            lines.append("".join("#" if cell != 0 else "." for cell in row))
        return "\n".join(lines)

    def decide_piece(self, board: TetrisBoard) -> Dict[str, Any]:
        """Runs one real Zesyde neural decision forward pass for the active piece."""
        if board.game_over:
            return {"game_over": True, "target_x": 3, "target_rot": 0}

        p = board.current_piece
        if not p:
            return {"game_over": True, "target_x": 3, "target_rot": 0}

        shortlisted = board.get_shortlisted_placements(count=6)
        if not shortlisted:
            return {"game_over": True, "target_x": 3, "target_rot": 0}

        board_ascii = self.format_board_ascii(board)
        state = (
            'Tetris board (10 wide). "#" filled, "." empty.\n'
            f"{board_ascii}\n"
            f"Current piece: {p}. Pick the best placement — clear lines, keep the stack low and flat, avoid holes."
        )

        question = "Which placement is best? Pick one."

        options = []
        option_map = {}
        for i, c in enumerate(shortlisted):
            key = f"p{i}"
            col_1based = c["x"] + 1
            if c["cleared"] > 0:
                desc = f"{key}: column {col_1based}, clears {c['cleared']} line(s), leaves {c['holes']} hole(s)"
            else:
                desc = f"{key}: column {col_1based}, leaves {c['holes']} hole(s)"
            options.append(desc)
            option_map[key] = c
            option_map[desc] = c

        total_chars = len(state) + len(question) + sum(len(o) for o in options)
        approx_tokens = max(1, total_chars // 4)
        cost_usd = (approx_tokens / 1_000_000.0) * 0.042

        t0 = time.perf_counter()
        result = self.router.decide(
            context=state,
            question=question,
            options=options,
            question_type="choice",
            escalation_threshold=self.escalation_threshold,
        )
        latency = (time.perf_counter() - t0) * 1000.0

        chosen = shortlisted[0]
        selected_text = result.selected_option
        match = re.search(r"p(\d+)", selected_text)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(shortlisted):
                chosen = shortlisted[idx]
        elif selected_text in option_map:
            chosen = option_map[selected_text]

        return {
            "target_x": chosen["x"],
            "target_rot": chosen["rot"],
            "landing_y": chosen["landing_y"],
            "cleared_projection": chosen["cleared"],
            "holes_projection": chosen["holes"],
            "max_h_projection": chosen["max_h"],
            "confidence": round(float(result.confidence), 4),
            "probabilities": {opt: round(float(p_val), 4) for opt, p_val in result.probabilities.items()},
            "should_escalate": bool(result.should_escalate),
            "escalate_probability": round(float(result.escalate_probability), 4),
            "latency_ms": round(float(latency), 1),
            "input_tokens": approx_tokens,
            "cost_usd": cost_usd,
            "context_text": state,
            "question_text": question,
            "options_text": options,
            "selected_option_text": selected_text,
        }
