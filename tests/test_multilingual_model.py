import torch
import pytest
from zesyde.model_multilingual import ZesydeMultilingualModel

def test_multilingual_model_forward():
    model = ZesydeMultilingualModel('hf-internal-testing/tiny-random-BertModel')
    input_ids = torch.randint(0, 100, (1, 32))
    attention_mask = torch.ones((1, 32))
    marker_indices = [[5, 10]]
    
    out = model(input_ids, attention_mask, marker_indices)
    assert isinstance(out, dict)
    assert 'logits' in out
    assert len(out['logits']) == 1
    assert out['logits'][0].shape == (2,)

