import torch
import pytest
from zesyde.model import ZesydeModel

def test_model_forward():
    # Use a tiny model for fast tests
    model = ZesydeModel('hf-internal-testing/tiny-random-BertModel')
    input_ids = torch.randint(0, 100, (2, 32))
    attention_mask = torch.ones((2, 32))
    marker_indices = [[5, 10], [5, 10, 15]]
    
    out = model(input_ids, attention_mask, marker_indices)
    assert isinstance(out, dict)
    assert 'logits' in out
    assert len(out['logits']) == 2
    assert out['logits'][0].shape == (2,)
    assert out['logits'][1].shape == (3,)


def test_context_fusion_backward():
    model = ZesydeModel('hf-internal-testing/tiny-random-BertModel', use_context_fusion=True)
    input_ids = torch.randint(0, 100, (1, 32))
    attention_mask = torch.ones((1, 32))
    marker_indices = [[5, 12]]
    
    out = model(input_ids, attention_mask, marker_indices)
    loss = out['logits'][0].sum() + out['act_logits'].sum()
    loss.backward()
    
    # Verify gradients flow into the option head
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.option_head.parameters())
    assert has_grad


def test_semantic_transformer_head():
    model = ZesydeModel('hf-internal-testing/tiny-random-BertModel', head_num_heads=4)
    input_ids = torch.randint(0, 100, (1, 32))
    attention_mask = torch.ones((1, 32))
    marker_indices = [[4, 10, 16, 22]]
    
    out = model(input_ids, attention_mask, marker_indices)
    assert len(out['logits']) == 1
    assert out['logits'][0].shape == (4,)


