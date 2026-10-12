# Zesyde Tetris Demo (`demos/tetris`)

Autonomous Tetris demo powered directly by the **Zesyde 150M-class Foundation Router** (`models/zesyde_prod_fp32.onnx`).

---

## Architecture (100% Transparent)

To remain completely honest and avoid misleading claims:
1. **Candidate Physics Simulation (`engine.py`)**:
   - Drops the active piece across all legal columns and rotations.
   - Evaluates prospective board stability (holes, aggregate height, bumpiness, line clears).
   - Generates the top candidate placements.
2. **Context Formatting & Decision Dispatch (`agent.py`)**:
   - Formats the 10×18 ASCII board matrix, active falling tetromino, and candidate placement options.
   - Dispatches the prompt to ONNX Runtime via `ZesydeRouter`.
3. **Execution & Web UI (`server.py` & `index.html`)**:
   - Executes the winning placement chosen by the model.
   - Displays real-time decision metrics: latency (~100ms), decision counts, and live neural softmax distributions.

---

## Quickstart

### 1. Launch the Server
```bash
python demos/tetris/run.py --port 8000
```

To run with INT8 quantized weights (faster on CPU):
```bash
python demos/tetris/run.py --port 8000 --int8
```

### 2. Open in Browser
Visit **`http://localhost:8000`** and click **▶ Watch Zesyde play**.
