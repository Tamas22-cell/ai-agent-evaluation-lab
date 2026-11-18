import pytest
from src.evaluator import evaluate_accuracy


@pytest.mark.parametrize(
    "response, expected, score",
    [
        ("6000", "6000", 100),
        ("5000", "6000", 0),
        ("Rome", "rome", 100),
        (" 1024 ", "1024", 100),
        ("", "Rome", 0),
        ("80", "8", 0),
        ("The answer is 8", "8", 0),
    ]
)
def test_accuracy(response, expected, score):
    result = evaluate_accuracy(response, expected)
    assert result["accuracy_score"] == score