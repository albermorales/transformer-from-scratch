from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.metrics.TranslationMetrics import TokenAccuracy


def sequence(token_ids):
    token_ids = torch.tensor([token_ids], dtype=torch.long)
    return {
        "token_ids": token_ids,
        "padding_mask": token_ids != 0,
    }


def recorded_accuracy(predictions, targets):
    batches = [
        {
            "en": {
                "token_ids": target["token_ids"],
                "padding_mask": target["padding_mask"],
            },
            "target": target,
        }
        for target in targets
    ]
    model = SimpleNamespace(generate=Mock(side_effect=predictions))
    writer = Mock()
    trainer = SimpleNamespace(
        model=model,
        writer=writer,
        valid_dl=batches,
        device=torch.device("cpu"),
        current_epoch=7,
    )

    TokenAccuracy(trainer)

    assert model.generate.call_count == len(batches)
    writer.add_scalar.assert_called_once()
    tag, value, epoch = writer.add_scalar.call_args.args
    assert tag == "eval/TokenAccuracy"
    assert epoch == trainer.current_epoch
    return value


@pytest.mark.parametrize(
    "prediction, target, expected",
    [
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 5], [1, 2, 4, 5], 0.5),
        ([1, 2, 3, 4, 5], [1, 2, 5], 0.4),
    ],
    ids=["perfect-generation", "short-generation", "long-generation"],
)
def test_token_accuracy_scores_generated_tokens(prediction, target, expected):
    value = recorded_accuracy([sequence(prediction)], [sequence(target)])
    assert value == pytest.approx(expected)


def test_token_accuracy_excludes_padding():
    prediction = sequence([1, 2, 0, 0])
    target = sequence([1, 2, 0, 0])
    # Values in positions masked out on both sides must not affect accuracy.
    prediction["token_ids"][0, 2:] = torch.tensor([8, 9])
    target["token_ids"][0, 2:] = torch.tensor([6, 7])

    assert recorded_accuracy([prediction], [target]) == pytest.approx(1.0)


def test_token_accuracy_accumulates_tokens_across_batches():
    # One correct token in the first batch and three errors in the second:
    # global accuracy is 1/4, rather than the mean of batch accuracies (1/2).
    predictions = [sequence([1]), sequence([7, 8, 9])]
    targets = [sequence([1]), sequence([2, 3, 4])]

    assert recorded_accuracy(predictions, targets) == pytest.approx(0.25)


def test_token_accuracy_handles_all_padding():
    assert recorded_accuracy(
        [sequence([0, 0])], [sequence([0, 0])]
    ) == pytest.approx(0.0)


def test_token_accuracy_handles_no_batches():
    assert recorded_accuracy([], []) == pytest.approx(0.0)
