"""Synthetic tests for the per-sequence functional LM-loss instrument."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from interp.functional import sequence_delta_lm, sequence_lm_losses


class TinyHookedLM(nn.Module):
    """A deterministic hook-capable LM; no external model or mock behavior is needed."""

    def __init__(self, *, nonfinite: bool = False) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(device="cpu")
        self.embedding = nn.Embedding(7, 3)
        self.unembed = nn.Linear(3, 7, bias=False)
        self._hooks = []
        self.nonfinite = nonfinite
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                    ]
                )
            )
            self.unembed.weight.copy_(self.embedding.weight)

    @contextmanager
    def hooks(self, *, fwd_hooks):  # noqa: ANN001
        previous = self._hooks
        self._hooks = list(fwd_hooks)
        try:
            yield self
        finally:
            self._hooks = previous

    def forward(self, tokens: torch.Tensor, *, return_type: str) -> torch.Tensor:
        assert return_type == "logits"
        activation = self.embedding(tokens)
        for _, hook_fn in self._hooks:
            activation = hook_fn(activation, None)
        logits = self.unembed(activation)
        if self.nonfinite:
            logits = logits.clone()
            logits[0, 0, 0] = float("nan")
        return logits


TOKENS = torch.tensor(
    [
        [0, 1, 2, 3, 4],
        [0, 2, 3, 4, 5],
        [0, 3, 4, 5, 6],
    ],
    dtype=torch.long,
)


def test_identity_transform_has_exact_zero_per_sequence_delta() -> None:
    result = sequence_delta_lm(TinyHookedLM(), TOKENS, lambda activation: activation)

    assert result["delta"].dtype == torch.float64
    assert result["delta"].device.type == "cpu"
    assert torch.equal(result["delta"], torch.zeros(3, dtype=torch.float64))


def test_skip_bos_only_exposes_non_bos_positions_to_transform() -> None:
    seen_shapes = []

    def record(activation: torch.Tensor) -> torch.Tensor:
        seen_shapes.append(tuple(activation.shape))
        return activation

    sequence_lm_losses(TinyHookedLM(), TOKENS, record, batch_size=2, skip_bos=True)

    assert seen_shapes == [(2, 4, 3), (1, 4, 3)]


def test_per_sequence_losses_do_not_depend_on_batch_partition() -> None:
    model = TinyHookedLM()

    together = sequence_lm_losses(model, TOKENS, batch_size=3)
    split = sequence_lm_losses(model, TOKENS, batch_size=2)

    assert together.shape == (3,)
    assert torch.allclose(together, split, rtol=0.0, atol=1e-12)


def test_wrong_transform_shape_fails_loudly() -> None:
    def wrong_shape(activation: torch.Tensor) -> torch.Tensor:
        return activation[..., :2]

    with pytest.raises(ValueError, match="shape"):
        sequence_lm_losses(TinyHookedLM(), TOKENS, wrong_shape)


@pytest.mark.parametrize(
    "tokens",
    [torch.ones(3), torch.ones((0, 5), dtype=torch.long), torch.ones((2, 1), dtype=torch.long)],
)
def test_malformed_token_batches_are_rejected(tokens: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="tokens"):
        sequence_lm_losses(TinyHookedLM(), tokens)


def test_nonfinite_language_model_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        sequence_lm_losses(TinyHookedLM(nonfinite=True), TOKENS)
