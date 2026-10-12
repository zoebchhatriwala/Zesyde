# Zesyde Adversarial Menu & Escalation Playground (`demos/bad_menu`)

Interactive stress-test playground testing Zesyde on intentionally flawed menus:
- **The Missing Destination**: When the correct choice is absent, can it select *"None of these"* or escalate instead of hallucinating?
- **The Overlapping Twins**: When options overlap ambiguously, can it detect high epistemic entropy and prompt for user clarification?
- **The Underspecified Input**: When queries lack context, does the model abstain?
- **Clean Match Control**: Authoritative, low-latency, calibrated routing.

---

## Quickstart

```bash
python demos/bad_menu/run.py --port 8082
```

Open **`http://127.0.0.1:8082`** in your browser to test live with Zesyde's calibrated INT8 ONNX CPU router.
