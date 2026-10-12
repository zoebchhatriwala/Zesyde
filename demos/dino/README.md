# Zesyde Dino Showcase (`demos/dino`)

Authentic Google Chrome T-Rex runner powered directly by the **Zesyde 150M-class Foundation Router** on CPU with sub-30ms real-time inference.

---

## Architecture & Mechanics

- **Authentic Canvas & Sprites**: Official Chromium sprite sheet (`100-offline-sprite.png`) and exact Chrome physics parameters (jump velocity, gravity, acceleration).
- **Zero Synthetic Hacks**: 100% genuine zero-shot neural routing over canonical tactical actions (`JUMP`, `DUCK`, `RUN`).
- **Low-Latency Telemetry**: Bidirectional WebSocket connection (`/ws`) with HTTP POST fallback (`/api/game-decision`).
- **Multi-lingual Support**: Prompts & telemetry localized across 7+ languages directly on the foundation model.

---

## Quickstart

```bash
# Run server with default INT8 precision on port 8088
python demos/dino/run.py

# Or specify custom port
python demos/dino/run.py --port 8088 --int8
```

Open **`http://127.0.0.1:8088`** in your browser:
- Press **Space** or click the dinosaur to start running.
- Press **`A`** to toggle between Autopilot mode (AI-controlled) and Manual mode.
- Press **`M`** to toggle sound effects.
