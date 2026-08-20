"""Synthetic evidence that the tangent objective is learnable and specializes.

Two questions, both answered locally on CPU with a tiny model and no real data:

1. can a tiny tangent model drive the constrained loss to the overfit floor on a
   fixed batch (the objective is representable at all);
2. on a structured clean distribution, does a tangent-trained model actually
   reconstruct the *orthogonal* state, rather than trivially copying the
   coordinate that the geometry already pins for free.

Question 2 is the one with teeth. Coordinate preservation is free by
construction, so a model that only preserves the coordinate must fail here.

Section 9 of the task specification asks for a matched-objective sanity check:
an isotropically trained model and a tangent-trained model of identical size and
budget, both evaluated on tangent-corrupted data. That is an implementation
check that the matched objective can specialize to the intended geometry. It is
not a scientific result and must never be reported as one.
"""

from __future__ import annotations

import pytest
import torch

from interp.conditional_flow import (
    ConditionalFlowMatcher,
    ConditionEncoderConfig,
    DirectionPoolProvenance,
    TrainingDirectionPool,
    _pool_digest,
    clamp_seed,
    sample_conditional_flow_batch,
    standardized_hyperplane,
)
from interp.flow_core import ActivationNormalizer, FlowModelConfig, flow_matching_loss
from interp.tangent_flow import (
    clamp_then_tangent_flow,
    coordinate,
    sample_tangent_flow_batch,
    tangent_flow_states,
    tangent_project,
)

D = 8
RANK = 2
N_DIRECTIONS = 4
IDENTITY = ActivationNormalizer(torch.zeros(D), torch.ones(D), eps=0.0)


def _directions(seed: int = 20260816) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(N_DIRECTIONS, D, generator=generator)
    return v / v.norm(dim=-1, keepdim=True)


def _pool(directions: torch.Tensor) -> TrainingDirectionPool:
    ranks = tuple(range(300, 300 + directions.shape[0]))
    fields = {
        "split": "training_only",
        "source": "synthetic_tangent_smoke",
        "selection": "unit_test",
        "selection_seed": 0,
        "min_rank": 256,
        "excluded_splits": ["dev", "held_out"],
        "n_directions": len(ranks),
    }
    return TrainingDirectionPool(
        directions,
        ranks,
        DirectionPoolProvenance(
            split="training_only",
            source="synthetic_tangent_smoke",
            selection="unit_test",
            selection_seed=0,
            min_rank=256,
            excluded_splits=("dev", "held_out"),
            n_directions=len(ranks),
            digest=_pool_digest(directions, ranks, fields),
        ),
    )


def _tiny_model(seed: int = 0) -> ConditionalFlowMatcher:
    torch.manual_seed(seed)
    cfg = FlowModelConfig(
        d_model=64,
        d_mlp=128,
        n_blocks=2,
        time_dim=16,
        time_hidden=32,
        max_period=10000.0,
        activation_dim=D,
    )
    return ConditionalFlowMatcher(cfg, ConditionEncoderConfig(cond_hidden=32), IDENTITY)


def _clean_batch(
    rows: int,
    directions: torch.Tensor,
    generator: torch.Generator,
    *,
    basis: torch.Tensor,
    c_scale: float = 1.0,
    c_shift: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x0 = c v + P_perp(z)`` with ``z`` in a fixed rank-2 subspace.

    The orthogonal part is low-rank, so genuine reconstruction is possible and a
    model that ignores it is measurably worse than one that does not. The
    coordinate is exactly ``c`` by construction.
    """

    index = torch.randint(len(directions), (rows,), generator=generator)
    v = directions[index]
    weights = torch.randn(rows, RANK, generator=generator)
    z = weights @ basis
    z_perp = z - (z * v).sum(-1, keepdim=True) * v
    c = torch.randn(rows, 1, generator=generator) * c_scale + c_shift
    return c * v + z_perp, v


def _basis(seed: int = 99) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(RANK, D, generator=generator)
    return basis / basis.norm(dim=-1, keepdim=True) * 2.0


def _train(
    model: ConditionalFlowMatcher,
    directions: torch.Tensor,
    *,
    objective: str,
    steps: int,
    batch: int = 256,
    seed: int = 4,
) -> list[float]:
    """Train the tiny model on one corruption geometry; return the loss trace."""

    pool = _pool(directions)
    basis = _basis()
    data_rng = torch.Generator().manual_seed(seed)
    flow_rng = torch.Generator().manual_seed(seed + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    trace: list[float] = []
    model.train()
    for _ in range(steps):
        h, _ = _clean_batch(batch, directions, data_rng, basis=basis)
        if objective == "tangent":
            sampled = sample_tangent_flow_batch(
                h, normalizer=IDENTITY, pool=pool, generator=flow_rng
            )
            prediction = tangent_project(
                model(sampled.x_t, sampled.t, sampled.v_x, sampled.c_x), sampled.v_x
            )
        else:
            sampled = sample_conditional_flow_batch(
                h, normalizer=IDENTITY, pool=pool, generator=flow_rng
            )
            prediction = model(sampled.x_t, sampled.t, sampled.v_x, sampled.c_x)
        loss = flow_matching_loss(prediction, sampled.target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach()))
    model.eval()
    return trace


@torch.no_grad()
def _tangent_reconstruction(
    model: ConditionalFlowMatcher,
    h: torch.Tensor,
    v: torch.Tensor,
    noise: torch.Tensor,
    *,
    t_start: float,
    nfe: int,
) -> dict[str, float]:
    """Tangent-corrupt a clean activation, reconstruct it, and score the orthogonal part."""

    c = (h * v).sum(-1, keepdim=True)
    v_x, c_x = standardized_hyperplane(IDENTITY, v, c)
    _, corrupted, _ = tangent_flow_states(h, v_x, c_x, noise, t_start)
    out = clamp_then_tangent_flow(
        model, h, v, c, noise=noise, t_start=t_start, nfe=nfe
    )

    def orthogonal_error(state: torch.Tensor) -> float:
        delta = state - h
        return float((delta - (delta * v).sum(-1, keepdim=True) * v).norm(dim=-1).mean())

    corrupted_error = orthogonal_error(corrupted)
    reconstructed_error = orthogonal_error(out.activation)
    return {
        "corrupted_orthogonal_error": corrupted_error,
        "reconstructed_orthogonal_error": reconstructed_error,
        "recovered_fraction": 1.0 - reconstructed_error / corrupted_error,
        "max_coordinate_drift": float(out.diagnostics["max_coordinate_drift"]),
        "coordinate_abs_error_max": float(out.diagnostics["coordinate_abs_error_max"]),
        "raw_parallel_velocity_norm_mean": float(
            out.diagnostics["raw_parallel_velocity_norm_mean"]
        ),
        "network_evaluations": out.network_evaluations,
    }


# --------------------------------------------------------------------------
# 1. the objective is representable: overfit a fixed batch to the floor
# --------------------------------------------------------------------------


def test_tiny_model_overfits_a_fixed_tangent_batch_to_the_floor() -> None:
    """The overfit floor of a memorizable fixed batch is zero; get close to it."""

    directions = _directions()
    generator = torch.Generator().manual_seed(1)
    h, _ = _clean_batch(64, directions, generator, basis=_basis())
    batch = sample_tangent_flow_batch(
        h, normalizer=IDENTITY, pool=_pool(directions),
        generator=torch.Generator().manual_seed(2),
    )
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    zero_predictor = float(batch.target_velocity.square().mean())
    first = None
    for _ in range(1500):
        prediction = tangent_project(
            model(batch.x_t, batch.t, batch.v_x, batch.c_x), batch.v_x
        )
        loss = flow_matching_loss(prediction, batch.target_velocity)
        first = float(loss.detach()) if first is None else first
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final = float(loss.detach())
    assert final < 0.02 * zero_predictor, (final, zero_predictor)
    assert final < 0.05 * first, (final, first)


# --------------------------------------------------------------------------
# 2. the model reconstructs the orthogonal state, not just the coordinate
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained() -> tuple[ConditionalFlowMatcher, torch.Tensor, list[float]]:
    directions = _directions()
    model = _tiny_model(seed=0)
    trace = _train(model, directions, objective="tangent", steps=3000)
    return model, directions, trace


def test_tangent_flow_loss_decreases_strongly(trained) -> None:  # noqa: ANN001
    _, _, trace = trained
    early = sum(trace[:50]) / 50
    late = sum(trace[-50:]) / 50
    # The flow-matching loss has a large irreducible floor (eps_perp is
    # unpredictable), so the meaningful evidence of learning is the
    # *reconstruction* tests below. This only rules out a flat trace.
    assert late < 0.65 * early, (early, late)


def test_orthogonal_reconstruction_improves_and_the_coordinate_never_moves(
    trained,  # noqa: ANN001
) -> None:
    model, directions, _ = trained
    generator = torch.Generator().manual_seed(77)
    h, v = _clean_batch(512, directions, generator, basis=_basis())
    noise = torch.randn(h.shape, generator=generator)

    scores = _tangent_reconstruction(model, h, v, noise, t_start=0.5, nfe=1)
    assert scores["recovered_fraction"] > 0.20, scores
    assert scores["max_coordinate_drift"] < 1e-4, scores
    assert scores["coordinate_abs_error_max"] < 1e-3, scores
    assert scores["network_evaluations"] == 1
    # The network still emits a parallel component; the projection is what
    # removes it, which is exactly the invariant this branch relies on.
    assert scores["raw_parallel_velocity_norm_mean"] > 0.0


def test_reconstruction_holds_for_every_direction_in_the_pool(trained) -> None:  # noqa: ANN001
    """Changing v must change the tangent subspace, not break reconstruction."""

    model, directions, _ = trained
    basis = _basis()
    for index in range(len(directions)):
        generator = torch.Generator().manual_seed(100 + index)
        one = directions[index : index + 1]
        h, v = _clean_batch(256, one, generator, basis=basis)
        noise = torch.randn(h.shape, generator=generator)
        scores = _tangent_reconstruction(model, h, v, noise, t_start=0.5, nfe=1)
        assert scores["recovered_fraction"] > 0.12, (index, scores)
        assert scores["max_coordinate_drift"] < 1e-4, (index, scores)


def test_shifting_the_coordinate_moves_the_parallel_part_only(trained) -> None:  # noqa: ANN001
    """Changing c relocates the parallel component without breaking the tangent task."""

    model, directions, _ = trained
    generator = torch.Generator().manual_seed(303)
    h, v = _clean_batch(256, directions, generator, basis=_basis())
    noise = torch.randn(h.shape, generator=generator)

    base = (h * v).sum(-1, keepdim=True)
    shifted_target = base + 3.0
    shifted = clamp_seed(h, v, shifted_target)
    # the shift is purely parallel
    delta = shifted - h
    assert torch.allclose(delta, (delta * v).sum(-1, keepdim=True) * v, atol=1e-5)
    assert torch.allclose((shifted * v).sum(-1, keepdim=True), shifted_target, atol=1e-4)

    scores = _tangent_reconstruction(model, shifted, v, noise, t_start=0.5, nfe=1)
    assert scores["recovered_fraction"] > 0.08, scores
    assert scores["max_coordinate_drift"] < 1e-4, scores


def test_predicted_velocity_is_exactly_tangent_after_projection(trained) -> None:  # noqa: ANN001
    model, directions, _ = trained
    generator = torch.Generator().manual_seed(404)
    h, _ = _clean_batch(128, directions, generator, basis=_basis())
    batch = sample_tangent_flow_batch(
        h, normalizer=IDENTITY, pool=_pool(directions),
        generator=torch.Generator().manual_seed(405),
    )
    raw = model(batch.x_t, batch.t, batch.v_x, batch.c_x)
    assert coordinate(raw, batch.v_x).abs().max() > 1e-4
    assert coordinate(tangent_project(raw, batch.v_x), batch.v_x).abs().max() < 1e-5


# --------------------------------------------------------------------------
# 3. matched-objective sanity check (implementation check, not a result)
# --------------------------------------------------------------------------


def test_tangent_objective_specializes_to_tangent_corruption(trained) -> None:  # noqa: ANN001
    """Same architecture, same budget, different training corruption geometry.

    The assertion is the specialization claim and nothing more: a model trained
    on tangent corruption fits the tangent velocity target better than an
    identically sized model trained on isotropic corruption. That is what the
    training path is supposed to do.

    Deliberately NOT asserted: that the tangent model reconstructs better. On
    this toy the two geometries are nearly equivalent and the reconstruction
    numbers come out level (~0.27 recovered at t=0.5 for both). Recording that
    honestly is the point; turning the synthetic problem until the gap looks
    large would be fishing, and this fixture is not evidence about T1 either way.
    """

    tangent_model, directions, _ = trained
    isotropic_model = _tiny_model(seed=0)
    _train(isotropic_model, directions, objective="isotropic", steps=3000)

    generator = torch.Generator().manual_seed(909)
    h, _ = _clean_batch(2048, directions, generator, basis=_basis())
    batch = sample_tangent_flow_batch(
        h, normalizer=IDENTITY, pool=_pool(directions),
        generator=torch.Generator().manual_seed(910),
    )

    def tangent_objective_mse(model: ConditionalFlowMatcher) -> float:
        with torch.no_grad():
            prediction = tangent_project(
                model(batch.x_t, batch.t, batch.v_x, batch.c_x), batch.v_x
            )
        return float(flow_matching_loss(prediction, batch.target_velocity))

    tangent_mse = tangent_objective_mse(tangent_model)
    isotropic_mse = tangent_objective_mse(isotropic_model)
    zero_predictor = float(batch.target_velocity.square().mean())

    assert tangent_mse < isotropic_mse, (tangent_mse, isotropic_mse)
    assert tangent_mse < 0.75 * zero_predictor, (tangent_mse, zero_predictor)
