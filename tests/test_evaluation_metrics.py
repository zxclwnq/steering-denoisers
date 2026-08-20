"""Hand-derived tests for the clean Phase B metric instruments."""

from __future__ import annotations

import math

import pytest
import torch


def test_text_metrics_are_hand_calculated() -> None:
    from interp.evaluation_metrics import text_metrics

    got = text_metrics("Police police policing cat cat", ("polic*", "cat"))

    assert got.n_words == 5
    assert got.lexicon_score == 1.0
    assert got.dist_1 == 3 / 5
    assert got.dist_2 == 1.0
    assert got.dist_3 == 1.0
    assert got.rep_3 == 0.0
    assert got.repetition_rate == 2 / 4


def test_short_text_metrics_do_not_invent_ngrams() -> None:
    from interp.evaluation_metrics import text_metrics

    empty = text_metrics("", ("word",))
    one = text_metrics("word", ("word",))

    assert empty.n_words == 0
    assert empty.lexicon_score == 0.0
    assert empty.dist_1 == empty.dist_2 == empty.dist_3 == 0.0
    assert empty.rep_3 == 1.0
    assert empty.repetition_rate == 0.0
    assert one.n_words == 1
    assert one.lexicon_score == 1.0
    assert one.dist_1 == 1.0
    assert one.dist_2 == one.dist_3 == 0.0


def test_masked_continuation_nll_counts_only_selected_targets() -> None:
    from interp.evaluation_metrics import masked_continuation_nll

    # Four predictions over two tokens. Selected targets have p=0.8 and p=0.6.
    probabilities = torch.tensor(
        [[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.6, 0.4]]], dtype=torch.float64
    )
    logits = probabilities.log()
    targets = torch.tensor([[0, 1, 0, 1]])
    mask = torch.tensor([[False, True, True, False]])

    got = masked_continuation_nll(logits, targets, mask)

    assert got.tolist() == pytest.approx([-math.log(0.8 * 0.6) / 2])


def test_masked_continuation_nll_rejects_empty_or_nonfinite_rows() -> None:
    from interp.evaluation_metrics import masked_continuation_nll

    logits = torch.zeros(1, 2, 3)
    targets = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="continuation token"):
        masked_continuation_nll(logits, targets, torch.zeros(1, 2, dtype=torch.bool))
    logits[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        masked_continuation_nll(logits, targets, torch.ones(1, 2, dtype=torch.bool))


def test_masked_feature_means_exclude_bos_and_padding() -> None:
    from interp.evaluation_metrics import masked_feature_means

    features = torch.tensor(
        [
            [[100.0, 100.0], [1.0, 2.0], [3.0, 6.0]],
            [[200.0, 200.0], [300.0, 300.0], [4.0, 8.0]],
        ]
    )
    valid = torch.tensor([[False, True, True], [False, False, True]])

    got = masked_feature_means(features, valid)

    assert got.tolist() == [[2.0, 4.0], [4.0, 8.0]]


def test_masked_feature_means_reject_rows_without_valid_tokens() -> None:
    from interp.evaluation_metrics import masked_feature_means

    with pytest.raises(ValueError, match="valid continuation"):
        masked_feature_means(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool))


def test_score_continuations_batches_only_equal_width_rows_without_padding() -> None:
    from interp.evaluation_metrics import score_continuations

    class Tokenizer:
        bos_token_id = 0

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [len(word) for word in text.split()]

    class Model:
        tokenizer = Tokenizer()
        cfg = type("Config", (), {"device": "cpu"})()

        def __init__(self) -> None:
            self.widths: list[int] = []

        def __call__(self, tokens: torch.Tensor, *, return_type: str) -> torch.Tensor:
            assert return_type == "logits"
            self.widths.append(tokens.shape[1])
            return torch.zeros(*tokens.shape, 16)

        def run_with_cache(self, tokens: torch.Tensor, *, names_filter: list[str]):
            self.widths.append(tokens.shape[1])
            h = torch.stack((tokens.float(), 2 * tokens.float()), dim=-1)
            return None, {names_filter[0]: h}

    class SAE:
        def encode_features(self, h: torch.Tensor, feature_ids: list[int]) -> torch.Tensor:
            assert feature_ids == [0, 1]
            return h

    model = Model()
    got = score_continuations(
        model,
        SAE(),
        "hook",
        ["aa b", "cccc"],
        ["ddd e", "ff"],
        [0, 1],
    )

    assert [item.nll for item in got] == pytest.approx([math.log(16), math.log(16)])
    assert got[0].continuation_token_ids == (3, 1)
    assert got[0].sae_feature_means == pytest.approx((2.0, 4.0))
    assert got[1].continuation_token_ids == (2,)
    assert got[1].sae_feature_means == pytest.approx((2.0, 4.0))
    assert model.widths == [3, 5, 2, 3]
