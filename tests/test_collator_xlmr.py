import pytest
from transformers import AutoTokenizer
from zesyde.collator import ZesydeCollator
from zesyde.schema import DecisionExample, QuestionType

def test_collator():
    tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-BertModel')
    collator = ZesydeCollator(tokenizer, max_length=64)
    
    ex = DecisionExample(
        id='1', context='ctx', question='q?', question_type=QuestionType.CHOICE,
        options=['Opt1', 'Opt2'], target=0
    )
    batch = collator([ex, ex])
    
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'marker_indices' in batch
    assert 'targets' in batch
    assert batch['input_ids'].shape[0] == 2
    assert len(batch['marker_indices']) == 2

