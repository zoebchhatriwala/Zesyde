import pytest
import os
from pathlib import Path
from zesyde.data import load_frozen_dataset, freeze_dataset
from zesyde.schema import DecisionExample, QuestionType

def test_freeze_and_load(tmp_path):
    ex = DecisionExample(
        id='1', context='ctx', question='q?', question_type=QuestionType.CHOICE,
        options=['A', 'B'], target=0
    )
    file_path = tmp_path / 'dataset.jsonl'
    freeze_dataset([ex], file_path)
    
    loaded = load_frozen_dataset(file_path)
    assert len(loaded) == 1
    assert loaded[0].id == '1'

def test_multilingual_eval_suite_exists():
    from zesyde.pipeline_utils import MULTILINGUAL_EVAL_FILES
    for name, path in MULTILINGUAL_EVAL_FILES.items():
        p = Path(path)
        assert p.exists(), f"Multilingual eval file missing: {path}"
        exs = load_frozen_dataset(p)
        assert len(exs) > 0, f"Empty eval dataset: {path}"
        assert all(isinstance(e, DecisionExample) for e in exs)

