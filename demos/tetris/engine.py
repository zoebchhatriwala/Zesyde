"""Core Tetris Game Mechanics Engine.

Provides 10x18 grid physics, 7-bag randomizer, rotations,
line-clear cascading, ghost piece projection, and candidate placement simulation.
"""

import random
from typing import Dict, List, Optional, Tuple, Any

TETROMINOES = {
    "I": {
        "color": "#6be0ff",
        "id": 1,
        "rotations": [
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 0), (0, 1), (0, 2), (0, 3)],
        ],
    },
    "O": {
        "color": "#edc22e",
        "id": 2,
        "rotations": [
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            [(0, 0), (1, 0), (0, 1), (1, 1)],
        ],
    },
    "T": {
        "color": "#c77dff",
        "id": 3,
        "rotations": [
            [(0, 0), (1, 0), (2, 0), (1, 1)],
            [(1, 0), (0, 1), (1, 1), (1, 2)],
            [(1, 0), (0, 1), (1, 1), (2, 1)],
            [(0, 0), (0, 1), (1, 1), (0, 2)],
        ],
    },
    "S": {
        "color": "#7ff0a8",
        "id": 4,
        "rotations": [
            [(1, 0), (2, 0), (0, 1), (1, 1)],
            [(0, 0), (0, 1), (1, 1), (1, 2)],
            [(1, 0), (2, 0), (0, 1), (1, 1)],
            [(0, 0), (0, 1), (1, 1), (1, 2)],
        ],
    },
    "Z": {
        "color": "#ff6b6b",
        "id": 5,
        "rotations": [
            [(0, 0), (1, 0), (1, 1), (2, 1)],
            [(1, 0), (0, 1), (1, 1), (0, 2)],
            [(0, 0), (1, 0), (1, 1), (2, 1)],
            [(1, 0), (0, 1), (1, 1), (0, 2)],
        ],
    },
    "J": {
        "color": "#7c8cff",
        "id": 6,
        "rotations": [
            [(0, 0), (0, 1), (1, 1), (2, 1)],
            [(0, 0), (1, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 0), (2, 0), (2, 1)],
            [(1, 0), (1, 1), (0, 2), (1, 2)],
        ],
    },
    "L": {
        "color": "#f59563",
        "id": 7,
        "rotations": [
            [(2, 0), (0, 1), (1, 1), (2, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 2)],
            [(0, 0), (1, 0), (2, 0), (0, 1)],
            [(0, 0), (1, 0), (1, 1), (1, 2)],
        ],
    },
}

WIDTH = 10
HEIGHT = 18


class TetrisBoard:
    """10x18 Tetris grid with real physical collision, lines clearing, and metric calculations."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.bag: List[str] = []
        self.current_piece: Optional[str] = None
        self.current_x: int = 3
        self.current_y: int = 0
        self.current_rot: int = 0
        self.next_piece: Optional[str] = None
        self.score: int = 0
        self.lines_cleared: int = 0
        self.level: int = 1
        self.game_over: bool = False
        self.piece_count: int = 0
        self.last_cleared_lines: int = 0

        self._refill_bag()
        self.next_piece = self._pop_piece()
        self._spawn_piece()

    def _refill_bag(self):
        pieces = list(TETROMINOES.keys())
        random.shuffle(pieces)
        self.bag.extend(pieces)

    def _pop_piece(self) -> str:
        if not self.bag:
            self._refill_bag()
        return self.bag.pop(0)

    def _spawn_piece(self) -> bool:
        self.current_piece = self.next_piece
        self.next_piece = self._pop_piece()
        self.current_rot = 0
        self.current_x = 3
        self.current_y = 0
        self.piece_count += 1

        if self._check_collision(self.current_x, self.current_y, self.current_rot):
            self.game_over = True
            return False
        return True

    def _get_blocks(self, piece: str, rot: int, x: int, y: int) -> List[Tuple[int, int]]:
        coords = TETROMINOES[piece]["rotations"][rot % 4]
        return [(x + dx, y + dy) for dx, dy in coords]

    def _check_collision(self, x: int, y: int, rot: int, piece: Optional[str] = None) -> bool:
        p = piece or self.current_piece
        if not p:
            return False
        blocks = self._get_blocks(p, rot, x, y)
        for bx, by in blocks:
            if bx < 0 or bx >= self.width:
                return True
            if by >= self.height:
                return True
            if by >= 0 and self.grid[by][bx] != 0:
                return True
        return False

    def get_column_heights(self) -> List[int]:
        """Calculates column heights from 0 to 18."""
        heights = []
        for c in range(self.width):
            h = 0
            for r in range(self.height):
                if self.grid[r][c] != 0:
                    h = self.height - r
                    break
            heights.append(h)
        return heights

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates aggregate height, holes, bumpiness, and max height."""
        heights = self.get_column_heights()
        holes = 0
        for c in range(self.width):
            seen = False
            for r in range(self.height):
                if self.grid[r][c] != 0:
                    seen = True
                elif seen and self.grid[r][c] == 0:
                    holes += 1

        max_h = max(heights) if heights else 0
        agg_h = sum(heights)
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(self.width - 1))

        return {
            "heights": heights,
            "max_height": max_h,
            "agg_height": agg_h,
            "holes": holes,
            "bumpiness": bumpiness,
        }

    def get_ghost_y(self, x: int, rot: int) -> int:
        """Finds the lowest valid y-coordinate for the current piece."""
        y = self.current_y
        while not self._check_collision(x, y + 1, rot):
            y += 1
        return y

    def simulate_placement(self, target_x: int, rot: int) -> Optional[Dict[str, Any]]:
        """Simulates dropping piece at (target_x, rot) and computes topology metrics."""
        p = self.current_piece
        if not p:
            return None

        if self._check_collision(target_x, 0, rot):
            return None

        y = 0
        while not self._check_collision(target_x, y + 1, rot):
            y += 1

        blocks = self._get_blocks(p, rot, target_x, y)
        temp_grid = [row[:] for row in self.grid]
        color_id = TETROMINOES[p]["id"]

        for bx, by in blocks:
            if 0 <= by < self.height and 0 <= bx < self.width:
                temp_grid[by][bx] = color_id

        # Check line clears
        cleared = 0
        cleaned_grid = []
        for r in range(self.height):
            if all(cell != 0 for cell in temp_grid[r]):
                cleared += 1
            else:
                cleaned_grid.append(temp_grid[r])

        for _ in range(cleared):
            cleaned_grid.insert(0, [0 for _ in range(self.width)])

        # Calculate metrics on prospective board
        heights = []
        for c in range(self.width):
            h = 0
            for r in range(self.height):
                if cleaned_grid[r][c] != 0:
                    h = self.height - r
                    break
            heights.append(h)

        holes = 0
        for c in range(self.width):
            seen = False
            for r in range(self.height):
                if cleaned_grid[r][c] != 0:
                    seen = True
                elif seen and cleaned_grid[r][c] == 0:
                    holes += 1

        max_h = max(heights) if heights else 0
        agg_h = sum(heights)
        bump = sum(abs(heights[i] - heights[i + 1]) for i in range(self.width - 1))

        # Standard board stability score
        s = 3.4 * cleared - 7.5 * holes - 0.5 * agg_h - 0.35 * bump - 0.2 * max_h

        return {
            "x": target_x,
            "rot": rot,
            "landing_y": y,
            "cleared": cleared,
            "holes": holes,
            "max_h": max_h,
            "agg_h": agg_h,
            "bumpiness": bump,
            "score": s,
        }

    def get_shortlisted_placements(self, count: int = 6) -> List[Dict[str, Any]]:
        """Returns top candidate placements shortlisted by stability score."""
        p = self.current_piece
        if not p:
            return []

        num_rots = len(TETROMINOES[p]["rotations"])
        if p == "O":
            num_rots = 1
        elif p in ("I", "S", "Z"):
            num_rots = 2

        candidates = []
        for r in range(num_rots):
            for x in range(self.width):
                sim = self.simulate_placement(x, r)
                if sim is not None:
                    candidates.append(sim)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:count]

    def apply_placement(self, target_x: int, target_rot: int) -> Dict[str, Any]:
        """Locks piece at (target_x, target_rot), clears lines, and spawns next piece."""
        if self.game_over or not self.current_piece:
            return {"cleared": 0, "game_over": True}

        self.current_x = target_x
        self.current_rot = target_rot % 4
        self.current_y = self.get_ghost_y(self.current_x, self.current_rot)

        p = self.current_piece
        blocks = self._get_blocks(p, self.current_rot, self.current_x, self.current_y)
        color_id = TETROMINOES[p]["id"]

        for bx, by in blocks:
            if 0 <= by < self.height and 0 <= bx < self.width:
                self.grid[by][bx] = color_id

        # Clear complete rows
        cleared = 0
        new_grid = []
        for r in range(self.height):
            if all(cell != 0 for cell in self.grid[r]):
                cleared += 1
            else:
                new_grid.append(self.grid[r])

        for _ in range(cleared):
            new_grid.insert(0, [0 for _ in range(self.width)])

        self.grid = new_grid
        self.lines_cleared += cleared
        self.last_cleared_lines = cleared

        line_pts = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += line_pts.get(cleared, 1000) * self.level
        self.level = (self.lines_cleared // 10) + 1

        spawn_ok = self._spawn_piece()

        return {
            "cleared": cleared,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "game_over": not spawn_ok,
        }

    def apply_action(self, action: str) -> bool:
        """Manual step action (for arrow key controls)."""
        if self.game_over or not self.current_piece:
            return False

        if action == "MOVE_LEFT":
            if not self._check_collision(self.current_x - 1, self.current_y, self.current_rot):
                self.current_x -= 1
        elif action == "MOVE_RIGHT":
            if not self._check_collision(self.current_x + 1, self.current_y, self.current_rot):
                self.current_x += 1
        elif action == "ROTATE":
            next_rot = (self.current_rot + 1) % 4
            if not self._check_collision(self.current_x, self.current_y, next_rot):
                self.current_rot = next_rot
            elif not self._check_collision(self.current_x - 1, self.current_y, next_rot):
                self.current_x -= 1
                self.current_rot = next_rot
            elif not self._check_collision(self.current_x + 1, self.current_y, next_rot):
                self.current_x += 1
                self.current_rot = next_rot
        elif action == "DROP":
            self.current_y = self.get_ghost_y(self.current_x, self.current_rot)
            self.apply_placement(self.current_x, self.current_rot)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes current board state."""
        return {
            "grid": self.grid,
            "width": self.width,
            "height": self.height,
            "current_piece": self.current_piece,
            "current_x": self.current_x,
            "current_y": self.current_y,
            "current_rot": self.current_rot,
            "ghost_y": self.get_ghost_y(self.current_x, self.current_rot),
            "next_piece": self.next_piece,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "level": self.level,
            "game_over": self.game_over,
            "piece_count": self.piece_count,
            "last_cleared_lines": self.last_cleared_lines,
            "metrics": self.get_metrics(),
        }
