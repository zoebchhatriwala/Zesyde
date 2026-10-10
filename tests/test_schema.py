import pytest
from zesyde.schema import DecisionExample, QuestionType

def test_decision_example_valid():
    ex = DecisionExample(
        id='1', context='ctx', question='q', question_type=QuestionType.CHOICE,
        options=['A', 'B'], target=0
    )
    assert ex.id == '1'
    assert ex.options[ex.target] == 'A'

def test_decision_example_invalid_options():
    with pytest.raises(ValueError):
        DecisionExample(
            id='2', context='ctx', question='q', question_type=QuestionType.CHOICE,
            options=['A'], target=0
        )

def test_version():
    import zesyde
    assert zesyde.__version__ == "1.0.0"

