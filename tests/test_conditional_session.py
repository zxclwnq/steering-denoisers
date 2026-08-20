"""The conditional generation session must inherit frozen mechanics unchanged."""

from __future__ import annotations

import pytest
import torch

from interp.conditional_steering import ConditionalFlowGenerationSession
from interp.flow_core import ActivationNormalizer
from interp.flow_steering import FlowGenerationSession, FlowNoiseCell

D_MODEL = 8


class StubConditional:
    def __init__(self) -> None:
        self.normalizer = ActivationNormalizer(
            torch.zeros(D_MODEL), torch.ones(D_MODEL), eps=0.0
        )

    def velocity_field(self, v_x: torch.Tensor, c_x: torch.Tensor):  # noqa: ANN201, ARG002
        def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            return torch.full_like(x, 0.1)

        return velocity


class StubUnconditional:
    def __init__(self) -> None:
        self.normalizer = ActivationNormalizer(
            torch.zeros(D_MODEL), torch.ones(D_MODEL), eps=0.0
        )

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return torch.full_like(x, 0.1)


def _direction() -> torch.Tensor:
    direction = torch.zeros(D_MODEL)
    direction[1] = 1.0
    return direction


def _kwargs(alpha: float = 2.0, t_start: float = 0.75) -> dict:
    return {
        "alpha": alpha,
        "t_start": t_start,
        "nfe": 3,
        "cells": (FlowNoiseCell("allegations", alpha, 0, 0),),
        "prompt_width": 3,
        "max_new_tokens": 2,
        "off_distribution_norm": 600.0,
    }


def test_matched_noise_is_identical_to_the_unconditional_session() -> None:
    """Epsilon must not depend on the method label (protocol section 6)."""

    h = torch.randn(1, 3, D_MODEL, generator=torch.Generator().manual_seed(7))

    conditional = ConditionalFlowGenerationSession(
        StubConditional(), _direction(), **_kwargs()
    )
    unconditional = FlowGenerationSession(StubUnconditional(), _direction(), **_kwargs())
    conditional.apply(h.clone())
    unconditional.apply(h.clone())

    assert (
        conditional.cell_receipts()[0]["epsilon_sha256"]
        == unconditional.cell_receipts()[0]["epsilon_sha256"]
    )


def test_conditional_output_differs_from_the_unconditional_arm() -> None:
    """Same noise and grid, different method: the activations must not coincide."""

    h = torch.randn(1, 3, D_MODEL, generator=torch.Generator().manual_seed(11))

    conditional = ConditionalFlowGenerationSession(
        StubConditional(), _direction(), **_kwargs()
    )
    unconditional = FlowGenerationSession(StubUnconditional(), _direction(), **_kwargs())

    assert not torch.allclose(conditional.apply(h.clone()), unconditional.apply(h.clone()))


def test_guard_falls_back_to_the_additive_steer() -> None:
    """A guarded position must match the additive arm exactly, as in frozen Phase B."""

    h = torch.zeros(1, 3, D_MODEL)
    h[..., 0] = 1000.0
    session = ConditionalFlowGenerationSession(
        StubConditional(), _direction(), **_kwargs()
    )

    out = session.apply(h.clone())

    assert torch.allclose(out, h + 2.0 * _direction(), atol=1e-4)
    assert session.receipt()["guarded_positions"] == 3


def test_coordinate_receipt_reports_the_c1_diagnostic() -> None:
    h = torch.randn(1, 3, D_MODEL, generator=torch.Generator().manual_seed(3))
    session = ConditionalFlowGenerationSession(
        StubConditional(), _direction(), **_kwargs()
    )

    session.apply(h)
    receipt = session.coordinate_receipt()

    assert receipt["seed_mode"] == "clean"
    assert receipt["positions"] == 3
    assert receipt["coordinate_abs_error_mean"] >= 0.0


def test_unknown_seed_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="seed_mode"):
        ConditionalFlowGenerationSession(
            StubConditional(), _direction(), seed_mode="additive", **_kwargs()
        )


def test_unconditional_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="ConditionalFlowMatcher"):
        ConditionalFlowGenerationSession(
            StubUnconditional(), _direction(), **_kwargs()
        )
