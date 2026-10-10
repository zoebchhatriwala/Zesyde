import torch
import pytest
from zesyde.model import ZesydeModel
from zesyde.model_multilingual import ZesydeMultilingualModel
from zesyde.onnx_export import export_to_onnx
import tempfile
from pathlib import Path

def test_zesyde_model_peft():
    # Use tiny random bert model for fast unit testing
    tiny_model = "hf-internal-testing/tiny-random-BertModel"
    
    # 1. Standard model without PEFT
    base_model = ZesydeModel(tiny_model, use_peft=False)
    base_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    
    # 2. Model with PEFT (LoRA) - default is use_peft=True
    peft_model = ZesydeModel(tiny_model, lora_r=4, lora_alpha=8)
    peft_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    
    # PEFT trainable params must be significantly fewer than full fine-tuning
    assert peft_trainable < base_trainable
    assert peft_model.use_peft is True

    # 3. Forward pass verification
    input_ids = torch.randint(0, 100, (2, 32))
    attention_mask = torch.ones((2, 32))
    marker_indices = [[5, 10], [5, 10, 15]]
    
    peft_model.eval()
    with torch.no_grad():
        out_before = peft_model(input_ids, attention_mask, marker_indices)
        
    assert "logits" in out_before
    assert len(out_before["logits"]) == 2
    assert out_before["logits"][0].shape == (2,)
    assert out_before["logits"][1].shape == (3,)
    assert "act_logits" in out_before
    assert out_before["act_logits"].shape == (2,)

    # 4. Merge PEFT weights
    peft_model.merge_peft()
    assert peft_model.use_peft is False

    with torch.no_grad():
        out_after = peft_model(input_ids, attention_mask, marker_indices)

    # Predictions before and after merge should match numerically
    torch.testing.assert_close(out_before["logits"][0], out_after["logits"][0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_before["logits"][1], out_after["logits"][1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_before["act_logits"], out_after["act_logits"], atol=1e-4, rtol=1e-4)


def test_peft_onnx_export():
    tiny_model = "hf-internal-testing/tiny-random-BertModel"
    model = ZesydeModel(tiny_model, lora_r=4, lora_alpha=8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "test_peft.onnx"
        # export_to_onnx should automatically merge PEFT weights and export cleanly
        export_to_onnx(model, onnx_file)
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 0
        assert model.use_peft is False


def test_zesyde_multilingual_model_peft_default():
    tiny_model = "hf-internal-testing/tiny-random-BertModel"
    model = ZesydeMultilingualModel(tiny_model, lora_r=4, lora_alpha=8)
    assert model.use_peft is True
    input_ids = torch.randint(0, 100, (1, 32))
    attention_mask = torch.ones((1, 32))
    marker_indices = [[5, 10]]
    out = model(input_ids, attention_mask, marker_indices)
    assert len(out["logits"]) == 1
    model.merge_peft()
    assert model.use_peft is False
