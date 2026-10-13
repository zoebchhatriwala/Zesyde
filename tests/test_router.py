import json
from unittest.mock import MagicMock

import pytest
import zesyde.router as router_module
from zesyde.router import ZesydeRouter

def test_router_initialization():
    # Ensure router can be instantiated (mocking dependencies if needed)
    # This is a placeholder for actual ZesydeRouter logic
    pass


def _fake_checkpoint_dir(tmp_path):
    (tmp_path / "zesyde_prod_fp32.onnx").write_bytes(b"")
    (tmp_path / "zesyde_prod_int8.onnx").write_bytes(b"")
    (tmp_path / "zesyde_calibration.json").write_text(
        json.dumps(
            {
                "backbone_name": "jhu-clsp/mmBERT-base",
                "temperatures": {"*|0": 1.0},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_from_pretrained_forwards_prefer_fp32(tmp_path, monkeypatch):
    """Regression test: from_pretrained must pass prefer_fp32 to the constructor.

    Previously the flag was accepted but dropped, so prefer_fp32=False callers
    (e.g. the demos' --int8 flags) silently ran FP32 anyway.
    """
    _fake_checkpoint_dir(tmp_path)
    monkeypatch.setattr(router_module.ort, "InferenceSession", MagicMock())
    monkeypatch.setattr(
        router_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=MagicMock()),
    )

    assert ZesydeRouter.from_pretrained(str(tmp_path), prefer_fp32=False).prefer_fp32 is False
    assert ZesydeRouter.from_pretrained(str(tmp_path), prefer_fp32=True).prefer_fp32 is True
    assert ZesydeRouter.from_pretrained(str(tmp_path)).prefer_fp32 is True
