# Zesyde Menu Routing & Escalation Evaluation (`demos/bad_menu`)

Interactive evaluation suite testing Zesyde on challenging or ill-formed menus:
- **Missing Option**: Evaluates whether the model selects *"None of these"* or escalates when the correct destination is absent.
- **Overlapping Options**: Evaluates whether the model identifies ambiguity and prompts for clarification when options conflict.
- **Vague Input**: Evaluates whether the model detects low confidence and abstains when input lacks context.
- **Clear Match Baseline**: Evaluates control routing accuracy on straightforward inputs.

---

## Quickstart

```bash
python demos/bad_menu/run.py --port 8082
```

Open **`http://127.0.0.1:8082`** in your browser to test live with Zesyde's calibrated INT8 ONNX CPU router.
