"""Mathematical and generation-state tests for clean Phase B flow steering."""

from __future__ import annotations

import inspect
import re

import pytest
import torch

from interp.flow_core import ActivationNormalizer
from interp.flow_steering import (
    FlowGenerationSession,
    FlowNoiseCell,
    apply_flow_steering,
    matched_flow_noise,
    steering_geometry,
)


class ConstantFlow:
    def __init__(self, value: torch.Tensor, *, mean=None, std=None) -> None:  # noqa: ANN001
        self.value = value
        self.normalizer = ActivationNormalizer(
            torch.zeros_like(value) if mean is None else mean,
            torch.ones_like(value) if std is None else std,
            eps=0.0,
        )
        self.times: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.times.append(t.detach().clone())
        return self.value.expand_as(x)


def _cells() -> tuple[FlowNoiseCell, ...]:
    return (
        FlowNoiseCell("allegations", 1.0, 3, 0),
        FlowNoiseCell("allegations", 1.0, 9, 0),
    )


def test_noise_is_matched_across_arms_and_keyed_by_absolute_position() -> None:
    positions = torch.tensor([[-2, -1, 0], [-2, -1, 0]])

    first = matched_flow_noise(_cells(), positions, d_model=4, device="cpu")
    second = matched_flow_noise(_cells(), positions, d_model=4, device="cpu")

    assert torch.equal(first, second)
    assert first.shape == (2, 3, 4)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert not torch.equal(first[0, 0], first[0, 1])
    assert set(inspect.signature(matched_flow_noise).parameters) == {
        "cells",
        "token_positions",
        "d_model",
        "device",
        "dtype",
        "namespace",
    }


def test_noise_follows_prompt_identity_under_batch_reordering() -> None:
    positions = torch.tensor([[-2, -1], [-2, -1]])
    ordered = matched_flow_noise(_cells(), positions, d_model=3, device="cpu")
    permuted = matched_flow_noise(
        tuple(reversed(_cells())), positions.flip(0), d_model=3, device="cpu"
    )

    assert torch.equal(permuted[0], ordered[1])
    assert torch.equal(permuted[1], ordered[0])


def test_noise_uses_exact_alpha_not_a_rounded_display_value() -> None:
    cell = FlowNoiseCell("allegations", 0.1, 0, 0)
    adjacent = FlowNoiseCell(
        "allegations",
        torch.nextafter(torch.tensor(0.1), torch.tensor(1.0)).item(),
        0,
        0,
    )
    positions = torch.tensor([[0]])

    first = matched_flow_noise((cell,), positions, d_model=3, device="cpu")
    second = matched_flow_noise((adjacent,), positions, d_model=3, device="cpu")

    assert not torch.equal(first, second)


def test_flow_steering_applies_add_standardize_noise_integrate_denormalize_order() -> None:
    model = ConstantFlow(
        torch.tensor([2.0, 4.0]),
        mean=torch.tensor([10.0, -4.0]),
        std=torch.tensor([2.0, 0.5]),
    )
    h = torch.tensor([[[12.0, -3.0]]])
    direction = torch.tensor([1.0, 0.0])
    noise = torch.tensor([[[6.0, -2.0]]])

    result = apply_flow_steering(
        model,
        h,
        direction,
        alpha=2.0,
        noise=noise,
        t_start=0.5,
        nfe=1,
        off_distribution_norm=600.0,
    )

    assert torch.equal(result.activation, torch.tensor([[[16.0, -5.0]]]))
    assert result.network_evaluations == 1
    assert len(model.times) == 1
    assert not result.guarded.any()


@pytest.mark.parametrize("nfe", [1, 3, 5])
def test_flow_steering_performs_exactly_the_requested_nfe(nfe: int) -> None:
    model = ConstantFlow(torch.tensor([0.25, -0.5]))

    result = apply_flow_steering(
        model,
        torch.zeros(1, 2, 2),
        torch.tensor([1.0, 0.0]),
        alpha=0.0,
        noise=torch.ones(1, 2, 2),
        t_start=0.25,
        nfe=nfe,
        off_distribution_norm=600.0,
    )

    assert result.network_evaluations == nfe
    assert len(model.times) == nfe
    assert not torch.equal(result.activation, torch.zeros_like(result.activation))


def test_zero_time_is_exact_additive_with_zero_flow_evaluations() -> None:
    model = ConstantFlow(torch.tensor([9.0, 9.0]))
    h = torch.tensor([[[1.0, 2.0]]])

    result = apply_flow_steering(
        model,
        h,
        torch.tensor([0.0, 1.0]),
        alpha=3.0,
        noise=torch.full_like(h, -100.0),
        t_start=0.0,
        nfe=5,
        off_distribution_norm=600.0,
    )

    assert torch.equal(result.activation, torch.tensor([[[1.0, 5.0]]]))
    assert result.network_evaluations == 0
    assert model.times == []


def test_guarded_positions_keep_additive_output_after_noise_was_supplied() -> None:
    model = ConstantFlow(torch.tensor([10.0, 10.0]))
    h = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]])
    direction = torch.tensor([1.0, 0.0])

    result = apply_flow_steering(
        model,
        h,
        direction,
        alpha=1.0,
        noise=torch.tensor([[[7.0, 8.0], [9.0, 10.0]]]),
        t_start=0.5,
        nfe=1,
        off_distribution_norm=2.5,
    )

    assert torch.equal(result.guarded, torch.tensor([[True, False]]))
    assert torch.equal(result.activation[0, 0], torch.tensor([3.0, 0.0]))
    assert len(model.times) == 1


@pytest.mark.parametrize("mutation", ["nonunit", "bad_noise", "nonfinite", "bad_rank"])
def test_flow_steering_rejects_ambiguous_or_nonfinite_inputs(mutation: str) -> None:
    model = ConstantFlow(torch.tensor([0.0, 0.0]))
    h = torch.zeros(1, 2, 2)
    direction = torch.tensor([1.0, 0.0])
    noise = torch.zeros_like(h)
    if mutation == "nonunit":
        direction = torch.tensor([2.0, 0.0])
    elif mutation == "bad_noise":
        noise = torch.zeros(1, 1, 2)
    elif mutation == "nonfinite":
        h[0, 0, 0] = float("nan")
    elif mutation == "bad_rank":
        h = torch.zeros(2)
        noise = torch.zeros_like(h)

    with pytest.raises(ValueError):
        apply_flow_steering(
            model,
            h,
            direction,
            alpha=1.0,
            noise=noise,
            t_start=0.5,
            nfe=1,
            off_distribution_norm=600.0,
        )


def test_geometry_matches_hand_calculated_parallel_and_orthogonal_correction() -> None:
    clean = torch.zeros(2, 2)
    additive = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    corrected = torch.tensor([[1.0, 3.0], [4.0, 0.0]])

    result = steering_geometry(
        clean, additive, corrected, torch.tensor([1.0, 0.0]), alpha=2.0
    )

    assert result["n_positions"] == 2
    assert result["realized_projection_mean"] == pytest.approx(2.5)
    assert result["retained_fraction_mean"] == pytest.approx(1.25)
    assert result["correction_norm_mean"] == pytest.approx((10**0.5 + 2.0) / 2.0)
    assert result["parallel_correction_norm_mean"] == pytest.approx(1.5)
    assert result["orthogonal_correction_norm_mean"] == pytest.approx(1.5)
    assert result["correction_cosine_mean"] == pytest.approx((-1 / 10**0.5 + 1.0) / 2.0)
    assert result["correction_cosine_defined"] == 2


def test_geometry_marks_alpha_zero_retention_and_zero_correction_cosine_undefined() -> None:
    clean = torch.zeros(1, 2)

    result = steering_geometry(
        clean, clean, clean, torch.tensor([1.0, 0.0]), alpha=0.0
    )

    assert result["retained_fraction_mean"] is None
    assert result["correction_cosine_mean"] is None
    assert result["correction_cosine_defined"] == 0


def _session(model: ConstantFlow, *, cells=None) -> FlowGenerationSession:  # noqa: ANN001
    return FlowGenerationSession(
        model,
        torch.tensor([1.0, 0.0]),
        alpha=1.0,
        t_start=0.25,
        nfe=1,
        cells=_cells() if cells is None else cells,
        prompt_width=3,
        max_new_tokens=2,
        off_distribution_norm=600.0,
    )


def test_generation_session_is_invariant_to_cached_or_full_context_hook_grouping() -> None:
    initial = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10.0
    generated = torch.tensor([[[2.0, 3.0]], [[4.0, 5.0]]])
    cached = _session(ConstantFlow(torch.tensor([0.5, -0.25])))
    full = _session(ConstantFlow(torch.tensor([0.5, -0.25])))

    cached_initial = cached.apply(initial)
    cached_generated = cached.apply(generated)
    full.apply(initial)
    full_all = full.apply(torch.cat((initial, generated), dim=1))

    assert torch.equal(full_all[:, :3], cached_initial)
    assert torch.equal(full_all[:, 3:], cached_generated)
    assert cached.receipt()["hook_calls"] == 2
    assert cached.receipt()["flow_network_evaluations"] == 2
    assert full.receipt()["hook_calls"] == 2
    assert full.receipt()["flow_network_evaluations"] == 2


def test_generation_session_follows_cell_identity_under_batch_reordering() -> None:
    h = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10.0
    ordered = _session(ConstantFlow(torch.tensor([0.5, -0.25])))
    permuted = _session(
        ConstantFlow(torch.tensor([0.5, -0.25])), cells=tuple(reversed(_cells()))
    )

    first = ordered.apply(h)
    second = permuted.apply(h.flip(0))

    assert torch.equal(second[0], first[1])
    assert torch.equal(second[1], first[0])


def test_generation_session_reports_geometry_and_noise_per_prompt_cell() -> None:
    session = _session(ConstantFlow(torch.tensor([0.5, -0.25])))
    initial = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10.0
    generated = torch.tensor([[[2.0, 3.0]], [[4.0, 5.0]]])

    session.apply(initial)
    session.apply(generated)
    receipts = session.cell_receipts()

    assert tuple(item["prompt_id"] for item in receipts) == (3, 9)
    assert all(item["positions_evaluated"] == 4 for item in receipts)
    assert all(item["padding_positions_evaluated"] == 0 for item in receipts)
    assert all(item["flow_network_evaluations"] == 2 for item in receipts)
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["epsilon_sha256"]) for item in receipts)
    assert all(item["geometry"]["token_variance_along_v"] >= 0 for item in receipts)


def test_cell_receipt_epsilon_is_independent_of_batch_order_and_arm() -> None:
    def run(cells, *, t_start: float, nfe: int):  # noqa: ANN001
        session = FlowGenerationSession(
            ConstantFlow(torch.tensor([0.5, -0.25])),
            torch.tensor([1.0, 0.0]),
            alpha=1.0,
            t_start=t_start,
            nfe=nfe,
            cells=cells,
            prompt_width=3,
            max_new_tokens=1,
            off_distribution_norm=600.0,
        )
        session.apply(torch.zeros(2, 3, 2))
        return {item["prompt_id"]: item["epsilon_sha256"] for item in session.cell_receipts()}

    first = run(_cells(), t_start=0.10, nfe=1)
    second = run(tuple(reversed(_cells())), t_start=0.50, nfe=5)

    assert first == second


def test_generation_session_excludes_left_padding_from_flow_noise_and_geometry() -> None:
    cells = _cells()
    mask = torch.tensor([[True, True, True], [False, True, True]])
    session = FlowGenerationSession(
        ConstantFlow(torch.tensor([0.5, -0.25])),
        torch.tensor([1.0, 0.0]),
        alpha=1.0,
        t_start=0.25,
        nfe=1,
        cells=cells,
        prompt_width=3,
        prompt_attention_mask=mask,
        max_new_tokens=1,
        off_distribution_norm=600.0,
    )
    initial = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10.0

    corrected = session.apply(initial)
    receipts = session.cell_receipts()

    assert torch.equal(corrected[1, 0], initial[1, 0])
    assert receipts[0]["positions_evaluated"] == 3
    assert receipts[1]["positions_evaluated"] == 2
    assert receipts[0]["epsilon_positions"] == 3
    assert receipts[1]["epsilon_positions"] == 2
    assert all(item["padding_positions_evaluated"] == 0 for item in receipts)


def test_generation_session_fails_loudly_on_ambiguous_hook_shape() -> None:
    session = _session(ConstantFlow(torch.tensor([0.5, -0.25])))
    session.apply(torch.zeros(2, 3, 2))

    with pytest.raises(ValueError, match="hook shape"):
        session.apply(torch.zeros(2, 2, 2))
