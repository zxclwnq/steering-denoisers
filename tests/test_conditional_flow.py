"""Tests for the conditional (steering-aware) flow prior.

The scientifically important test here is
``test_synthetic_conditioning_is_actually_used``: an ambiguous toy problem where
``x_t`` carries no information about the conditioned coordinate, so the model can
only fit it by using ``(v_x, c_x)``.  A model that learns to ignore the condition
fails that test rather than passing on shapes alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from interp.conditional_flow import (
    MIN_TRAINING_RANK,
    ConditionalFlowMatcher,
    ConditionEncoderConfig,
    DirectionPoolProvenance,
    TrainingDirectionPool,
    canonicalize_hyperplane,
    clamp_seed,
    condition_parameter_count,
    conditional_clamp_steer,
    conditional_parameter_count,
    implied_x0,
    load_conditional_flow_config,
    load_training_direction_pool,
    sample_conditional_flow_batch,
    save_direction_pool,
    standardized_hyperplane,
    target_coordinate,
)
from interp.flow_core import (
    ActivationNormalizer,
    FlowMatcher,
    FlowModelConfig,
    flow_matching_loss,
    flow_parameter_count,
)
from interp.train_flow import load_flow_checkpoint, save_flow_checkpoint

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "flow_core_conditional_60m_v1.yaml"


def _tiny_config(width: int = 6) -> FlowModelConfig:
    return FlowModelConfig(
        d_model=16,
        d_mlp=32,
        n_blocks=2,
        time_dim=8,
        time_hidden=8,
        max_period=10000.0,
        activation_dim=width,
    )


def _normalizer(width: int = 6, *, trivial: bool = False) -> ActivationNormalizer:
    if trivial:
        return ActivationNormalizer(torch.zeros(width), torch.ones(width), eps=0.0)
    generator = torch.Generator().manual_seed(11)
    mean = torch.randn(width, generator=generator)
    std = torch.rand(width, generator=generator) + 0.5
    return ActivationNormalizer(mean, std, eps=1e-5)


def _tiny_model(
    width: int = 6, *, seed: int = 0, trivial_norm: bool = False
) -> ConditionalFlowMatcher:
    torch.manual_seed(seed)
    return ConditionalFlowMatcher(
        _tiny_config(width),
        ConditionEncoderConfig(cond_hidden=12),
        _normalizer(width, trivial=trivial_norm),
    )


def _unit(rows: int, width: int, generator: torch.Generator) -> torch.Tensor:
    vectors = torch.randn(rows, width, generator=generator)
    return vectors / vectors.norm(dim=-1, keepdim=True)


def synthetic_pool(
    tmp_path: Path, rows: int = 5, width: int = 6, *, seed: int = 31, name: str = "pool.pt"
):
    """A training-only pool manifest with a valid provenance digest."""

    generator = torch.Generator().manual_seed(seed)
    directions = _unit(rows, width, generator)
    ranks = tuple(range(MIN_TRAINING_RANK + 7, MIN_TRAINING_RANK + 7 + rows))
    path = tmp_path / name
    save_direction_pool(
        path,
        directions,
        ranks,
        source="synthetic_test_fixture",
        selection="blake2b_priority_rank",
        selection_seed=20260807,
    )
    return load_training_direction_pool(path)


# -------------------------------------------------------------- A: hyperplane


def test_raw_and_standardized_hyperplanes_are_equivalent() -> None:
    generator = torch.Generator().manual_seed(3)
    width = 9
    normalizer = ActivationNormalizer(
        torch.randn(width, generator=generator),
        torch.rand(width, generator=generator) + 0.25,
        eps=1e-5,
    )
    v = _unit(4, width, generator)
    h = torch.randn(4, width, generator=generator)
    c = (h * v).sum(dim=-1, keepdim=True)

    v_x, c_x = standardized_hyperplane(normalizer, v, c)
    x = normalizer.normalize(h)

    assert torch.allclose((x * v_x).sum(dim=-1, keepdim=True), c_x, atol=1e-5)
    assert torch.allclose(v_x.norm(dim=-1), torch.ones(4), atol=1e-6)


def test_hyperplane_membership_transfers_for_off_manifold_points() -> None:
    generator = torch.Generator().manual_seed(4)
    width = 7
    normalizer = _normalizer(width)
    v = _unit(1, width, generator)
    h = torch.randn(3, width, generator=generator)
    target = torch.tensor([[2.5], [-1.0], [0.0]])
    seeded = clamp_seed(h, v, target)

    v_x, c_x = standardized_hyperplane(normalizer, v, target)
    x = normalizer.normalize(seeded)

    assert torch.allclose((x * v_x).sum(dim=-1, keepdim=True), c_x, atol=1e-5)


def test_degenerate_standardized_direction_is_rejected() -> None:
    width = 4
    normalizer = ActivationNormalizer(torch.zeros(width), torch.full((width,), 1e-12), eps=0.0)
    v = torch.zeros(width)
    v[0] = 1.0
    with pytest.raises(ValueError, match="degenerate"):
        standardized_hyperplane(normalizer, v, 1.0, min_norm=1e-8)


# ------------------------------------------------------- B, C: seed conversion


def test_clamp_seed_lands_exactly_on_the_requested_coordinate() -> None:
    generator = torch.Generator().manual_seed(5)
    v = _unit(1, 8, generator)
    h = torch.randn(5, 8, generator=generator)
    target = torch.tensor([1.0, -3.0, 0.0, 7.5, 2.0])

    seed = clamp_seed(h, v, target)

    assert torch.allclose((seed * v).sum(dim=-1), target, atol=1e-5)


def test_additive_mode_reproduces_h_plus_alpha_v() -> None:
    generator = torch.Generator().manual_seed(6)
    v = _unit(1, 8, generator)
    h = torch.randn(4, 8, generator=generator)
    alpha = 2.75

    target = target_coordinate(h, v, alpha, mode="additive")
    seed = clamp_seed(h, v, target)

    assert torch.allclose(seed, h + alpha * v, atol=1e-5)


def test_absolute_mode_removes_the_existing_component() -> None:
    generator = torch.Generator().manual_seed(7)
    v = _unit(1, 8, generator)
    h = torch.randn(4, 8, generator=generator)
    alpha = -1.5

    target = target_coordinate(h, v, alpha, mode="absolute")
    seed = clamp_seed(h, v, target)

    expected = h - (h * v).sum(dim=-1, keepdim=True) * v + alpha * v
    assert torch.allclose(seed, expected, atol=1e-5)
    assert torch.allclose((seed * v).sum(dim=-1), torch.full((4,), alpha), atol=1e-5)


def test_unknown_coordinate_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown coordinate mode"):
        target_coordinate(torch.zeros(2, 3), torch.tensor([1.0, 0.0, 0.0]), 1.0, mode="clamp")


# ------------------------------------------------------ D: sign canonicalization


def test_opposite_hyperplane_signs_canonicalize_identically() -> None:
    generator = torch.Generator().manual_seed(8)
    width = 11
    normalizer = _normalizer(width)
    v = _unit(3, width, generator)
    c = torch.tensor([[1.25], [-0.5], [3.0]])

    forward = standardized_hyperplane(normalizer, v, c)
    reversed_ = standardized_hyperplane(normalizer, -v, -c)

    assert torch.allclose(forward[0], reversed_[0], atol=1e-6)
    assert torch.allclose(forward[1], reversed_[1], atol=1e-6)


def test_canonicalization_makes_the_largest_component_positive() -> None:
    v_x = torch.tensor([[0.3, -0.9, 0.1], [0.8, 0.2, -0.1]])
    c_x = torch.tensor([[2.0], [-1.0]])

    canonical_v, canonical_c = canonicalize_hyperplane(v_x, c_x)

    pivot = canonical_v.abs().argmax(dim=-1, keepdim=True)
    assert bool((torch.gather(canonical_v, -1, pivot) > 0).all())
    assert torch.allclose(canonical_v[0], -v_x[0])
    assert torch.allclose(canonical_c[0], -c_x[0])
    assert torch.allclose(canonical_v[1], v_x[1])


# ----------------------------------------------------------- E: batch semantics


def test_per_row_directions_and_targets_are_independent() -> None:
    model = _tiny_model()
    generator = torch.Generator().manual_seed(9)
    h = torch.randn(4, 6, generator=generator)
    directions = _unit(4, 6, generator)
    alphas = torch.tensor([1.0, -2.0, 0.5, 3.0])
    noise = torch.randn(4, 6, generator=generator)

    batched = conditional_clamp_steer(
        model, h, directions, alpha=alphas, noise=noise, t_start=0.4, nfe=3
    )
    for row in range(4):
        single = conditional_clamp_steer(
            model,
            h[row : row + 1],
            directions[row],
            alpha=float(alphas[row]),
            noise=noise[row : row + 1],
            t_start=0.4,
            nfe=3,
        )
        assert torch.allclose(batched.activation[row], single.activation[0], atol=1e-5)


# ------------------------------------------------- F: padding/masking behaviour


def test_masked_positions_do_not_change_valid_row_results() -> None:
    model = _tiny_model()
    generator = torch.Generator().manual_seed(10)
    h = torch.randn(2, 5, 6, generator=generator)
    noise = torch.randn(2, 5, 6, generator=generator)
    valid = torch.tensor(
        [[False, True, True, True, True], [False, False, True, True, True]]
    )
    v = _unit(1, 6, generator)

    selected = conditional_clamp_steer(
        model, h[valid], v, alpha=1.5, noise=noise[valid], t_start=0.4, nfe=2
    )
    padded_h = h.clone()
    padded_h[~valid] = 123.0  # arbitrary padding content
    reselected = conditional_clamp_steer(
        model, padded_h[valid], v, alpha=1.5, noise=noise[valid], t_start=0.4, nfe=2
    )

    assert torch.equal(selected.activation, reselected.activation)


# -------------------------------------------- G, H: finite forward and gradients


def test_forward_backward_is_finite_and_condition_receives_gradient(tmp_path: Path) -> None:
    model = _tiny_model()
    generator = torch.Generator().manual_seed(12)
    h = torch.randn(8, 6, generator=generator)
    pool = synthetic_pool(tmp_path)
    batch = sample_conditional_flow_batch(
        h, normalizer=model.normalizer, pool=pool, generator=generator
    )

    prediction = model(batch.x_t, batch.t, batch.v_x, batch.c_x)
    loss = flow_matching_loss(prediction, batch.target_velocity)
    loss.backward()

    assert torch.isfinite(prediction).all() and torch.isfinite(loss)
    for name, parameter in model.condition.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0.0, name


# ---------------------------------------------------------------- I: save/load


def test_checkpoint_round_trip_reproduces_outputs(tmp_path: Path) -> None:
    model = _tiny_model(seed=3).eval()
    generator = torch.Generator().manual_seed(13)
    x_t = torch.randn(3, 6, generator=generator)
    v_x = _unit(3, 6, generator)
    c_x = torch.tensor([[0.5], [-1.0], [2.0]])
    t = torch.full((3, 1), 0.3)

    with torch.no_grad():
        before = model(x_t, t, v_x, c_x)
    path = tmp_path / "conditional.pt"
    save_flow_checkpoint(model, path, metadata={"experiment_id": "test"})
    restored, metadata, _ = load_flow_checkpoint(path)
    with torch.no_grad():
        after = restored(x_t, t, v_x, c_x)

    assert torch.equal(before, after)
    assert metadata == {"experiment_id": "test"}
    assert restored.normalizer.eps == model.normalizer.eps
    assert torch.equal(restored.normalizer.mean, model.normalizer.mean)


# ------------------------------------------------------ K, L: sampler semantics


def test_nfe_accounting_matches_the_unconditional_convention() -> None:
    model = _tiny_model()
    generator = torch.Generator().manual_seed(14)
    h = torch.randn(3, 6, generator=generator)
    noise = torch.randn(3, 6, generator=generator)
    v = _unit(1, 6, generator)
    calls = 0
    inner = model.forward

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return inner(*args, **kwargs)

    model.forward = counting  # type: ignore[method-assign]
    for nfe in (1, 3, 5):
        calls = 0
        output = conditional_clamp_steer(
            model, h, v, alpha=1.0, noise=noise, t_start=0.6, nfe=nfe
        )
        assert calls == nfe
        assert output.network_evaluations == nfe


def test_t_start_zero_is_exact_seed_identity_with_no_evaluations() -> None:
    model = _tiny_model()
    generator = torch.Generator().manual_seed(15)
    h = torch.randn(4, 6, generator=generator)
    noise = torch.randn(4, 6, generator=generator)
    v = _unit(1, 6, generator)
    calls = 0
    inner = model.forward

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return inner(*args, **kwargs)

    model.forward = counting  # type: ignore[method-assign]
    output = conditional_clamp_steer(
        model, h, v, alpha=2.0, noise=noise, t_start=0.0, nfe=3
    )

    assert calls == 0
    assert output.network_evaluations == 0
    assert torch.equal(output.activation, h + 2.0 * v)
    assert torch.allclose(output.coordinate_error, torch.zeros(4), atol=1e-5)


def test_final_projection_is_off_by_default_and_exact_when_requested() -> None:
    model = _tiny_model()
    generator = torch.Generator().manual_seed(16)
    h = torch.randn(6, 6, generator=generator)
    noise = torch.randn(6, 6, generator=generator)
    v = _unit(1, 6, generator)

    free = conditional_clamp_steer(model, h, v, alpha=2.0, noise=noise, t_start=0.5, nfe=2)
    projected = conditional_clamp_steer(
        model, h, v, alpha=2.0, noise=noise, t_start=0.5, nfe=2, final_projection=True
    )

    assert free.final_projection is False
    assert projected.final_projection is True
    # An untrained model does not land on the coordinate by itself; the optional
    # projection does, and the two paths must be distinguishable.
    assert float(free.coordinate_error.abs().max()) > 1e-4
    assert torch.allclose(projected.coordinate_error, torch.zeros(6), atol=1e-4)
    assert not torch.equal(free.activation, projected.activation)


def test_condition_is_held_fixed_across_the_trajectory() -> None:
    model = _tiny_model()
    generator = torch.Generator().manual_seed(17)
    h = torch.randn(2, 6, generator=generator)
    noise = torch.randn(2, 6, generator=generator)
    v = _unit(1, 6, generator)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    inner = model.forward

    def recording(x_t, t, v_x, c_x):
        seen.append((v_x.clone(), torch.as_tensor(c_x).clone()))
        return inner(x_t, t, v_x, c_x)

    model.forward = recording  # type: ignore[method-assign]
    conditional_clamp_steer(model, h, v, alpha=1.0, noise=noise, t_start=0.7, nfe=4)

    assert len(seen) == 4
    for direction, coordinate in seen[1:]:
        assert torch.equal(direction, seen[0][0])
        assert torch.equal(coordinate, seen[0][1])


# ------------------------------------------------- training-only pool isolation


def test_pool_rejects_ranks_below_the_training_only_floor(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(18)
    with pytest.raises(ValueError, match="below the declared floor"):
        save_direction_pool(
            tmp_path / "low.pt",
            _unit(2, 6, generator),
            ranks=(3, MIN_TRAINING_RANK + 1),
            source="synthetic",
            selection="blake2b_priority_rank",
            selection_seed=1,
        )
    assert not (tmp_path / "low.pt").exists()


def test_pool_rejects_a_floor_below_the_governance_minimum(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(23)
    with pytest.raises(ValueError, match="training-only floor"):
        save_direction_pool(
            tmp_path / "shallow.pt",
            _unit(2, 6, generator),
            ranks=(64, 65),
            source="synthetic",
            selection="blake2b_priority_rank",
            selection_seed=1,
            min_rank=64,
        )


def test_pool_rejects_non_training_provenance() -> None:
    generator = torch.Generator().manual_seed(19)
    directions = _unit(2, 6, generator)
    ranks = (MIN_TRAINING_RANK, MIN_TRAINING_RANK + 1)
    for split in ("dev", "held_out", "all"):
        provenance = DirectionPoolProvenance(
            split=split,
            source="synthetic",
            selection="blake2b_priority_rank",
            selection_seed=1,
            min_rank=MIN_TRAINING_RANK,
            excluded_splits=("dev", "held_out"),
            n_directions=2,
            digest="0" * 64,
        )
        with pytest.raises(ValueError, match="split must be"):
            TrainingDirectionPool(directions, ranks, provenance)


def test_pool_requires_both_evaluation_splits_to_be_excluded() -> None:
    generator = torch.Generator().manual_seed(24)
    provenance = DirectionPoolProvenance(
        split="training_only",
        source="synthetic",
        selection="blake2b_priority_rank",
        selection_seed=1,
        min_rank=MIN_TRAINING_RANK,
        excluded_splits=("dev",),  # held_out not excluded
        n_directions=2,
        digest="0" * 64,
    )
    with pytest.raises(ValueError, match="missing \\['held_out'\\]"):
        TrainingDirectionPool(
            _unit(2, 6, generator),
            (MIN_TRAINING_RANK, MIN_TRAINING_RANK + 1),
            provenance,
        )


def test_pool_requires_a_provenance_record_not_a_bare_string() -> None:
    generator = torch.Generator().manual_seed(25)
    with pytest.raises(TypeError, match="DirectionPoolProvenance"):
        TrainingDirectionPool(
            _unit(2, 6, generator),
            (MIN_TRAINING_RANK, MIN_TRAINING_RANK + 1),
            "training_only",  # type: ignore[arg-type]
        )


def test_tampered_manifest_fails_the_provenance_digest(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(26)
    path = tmp_path / "pool.pt"
    save_direction_pool(
        path,
        _unit(3, 6, generator),
        ranks=(300, 301, 302),
        source="synthetic",
        selection="blake2b_priority_rank",
        selection_seed=20260807,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    swapped = _unit(3, 6, torch.Generator().manual_seed(999))
    torch.save({**payload, "directions": swapped}, path)

    with pytest.raises(ValueError, match="digest mismatch"):
        load_training_direction_pool(path)


def test_tampered_provenance_claims_fail_the_digest(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(27)
    path = tmp_path / "pool.pt"
    save_direction_pool(
        path,
        _unit(3, 6, generator),
        ranks=(300, 301, 302),
        source="synthetic",
        selection="blake2b_priority_rank",
        selection_seed=20260807,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    record = {**payload["provenance"], "source": "a_different_sae"}
    torch.save({**payload, "provenance": record}, path)

    with pytest.raises(ValueError, match="digest mismatch"):
        load_training_direction_pool(path)


def test_direction_manifest_under_protected_path_is_refused(tmp_path: Path) -> None:
    protected = tmp_path / "configs" / "protected"
    protected.mkdir(parents=True)
    manifest = protected / "directions.pt"
    manifest.write_bytes(b"never read")

    with pytest.raises(PermissionError, match="protected path"):
        load_training_direction_pool(manifest)


def test_pool_round_trips_with_its_provenance_record(tmp_path: Path) -> None:
    pool = synthetic_pool(tmp_path, rows=4)

    assert len(pool) == 4
    identity = pool.identity()
    assert identity["split"] == "training_only"
    assert identity["excluded_splits"] == ["dev", "held_out"]
    assert identity["selection"] == "blake2b_priority_rank"
    assert identity["min_rank"] == MIN_TRAINING_RANK
    assert identity["observed_min_rank"] >= MIN_TRAINING_RANK
    assert len(identity["digest"]) == 64
    sampled = pool.sample(3, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(sampled.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_training_batch_conditions_on_the_natural_coordinate(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(21)
    width = 6
    normalizer = _normalizer(width)
    h = torch.randn(7, width, generator=generator)
    pool = synthetic_pool(tmp_path, rows=3, width=width)

    batch = sample_conditional_flow_batch(
        h, normalizer=normalizer, pool=pool, generator=generator
    )

    x0 = normalizer.normalize(h)
    # c_x must be the standardized coordinate of the clean activation itself.
    assert torch.allclose((x0 * batch.v_x).sum(dim=-1, keepdim=True), batch.c_x, atol=1e-4)
    # x_t = (1 - t) x_0 + t eps with eps = x_0 + u, i.e. the unconditional path.
    epsilon = x0 + batch.target_velocity
    assert torch.allclose(batch.x_t, (1 - batch.t) * x0 + batch.t * epsilon, atol=1e-5)


# ------------------------------------------------------ config and parameters


def test_conditional_config_declares_the_conditioning_contract() -> None:
    cfg = load_conditional_flow_config(CONFIG)

    assert cfg.coordinate == "linear_projection"
    assert cfg.feature_id_conditioning is False
    assert cfg.model.activation_width == 768
    assert cfg.model.n_blocks == 3
    assert condition_parameter_count(cfg.model, cfg.condition) == 592_640
    assert conditional_parameter_count(cfg.model, cfg.condition) == 61_000_448


def test_parameter_overhead_matches_the_built_module() -> None:
    cfg = _tiny_config()
    cond = ConditionEncoderConfig(cond_hidden=12)
    model = ConditionalFlowMatcher(cfg, cond, _normalizer())
    base = FlowMatcher(cfg, _normalizer())

    built = sum(p.numel() for p in model.parameters())
    assert built == conditional_parameter_count(cfg, cond)
    assert built - sum(p.numel() for p in base.parameters()) == condition_parameter_count(cfg, cond)
    assert flow_parameter_count(cfg) == sum(p.numel() for p in base.parameters())


# ------------------------------------------------ the synthetic conditioning smoke


def _toy_problem(
    rows: int, width: int, directions: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clean states whose coordinate along a chosen direction is the condition.

    ``x_0`` is Gaussian noise with its ``v`` component replaced by ``c``, so
    ``E[x_0 | v, c] = c v``.  Everything else about ``x_0`` is unpredictable.
    """

    index = torch.randint(directions.shape[0], (rows,), generator=generator)
    v = directions[index]
    c = torch.empty(rows, 1).uniform_(-3.0, 3.0, generator=generator)
    z = torch.randn(rows, width, generator=generator)
    x0 = z + (c - (z * v).sum(dim=-1, keepdim=True)) * v
    return x0, v, c


def test_synthetic_conditioning_is_actually_used() -> None:
    """Overfit a tiny conditional model on a problem x_t alone cannot solve.

    At ``t = 1`` the state is pure noise and carries zero information about the
    clean coordinate, so any loss reduction below the zero-predictor level must
    come from ``(v_x, c_x)``.
    """

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    width = 8
    rows = 256
    directions = _unit(3, width, generator)
    model = ConditionalFlowMatcher(
        FlowModelConfig(
            d_model=64, d_mlp=128, n_blocks=2, time_dim=16, time_hidden=32, max_period=10000.0,
            activation_dim=width,
        ),
        ConditionEncoderConfig(cond_hidden=32),
        ActivationNormalizer(torch.zeros(width), torch.ones(width), eps=0.0),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    time = torch.ones(rows, 1)

    def coordinate_error(seed: int) -> float:
        probe = torch.Generator().manual_seed(seed)
        x0, v, c = _toy_problem(rows, width, directions, probe)
        noise = torch.randn(rows, width, generator=probe)
        with torch.no_grad():
            velocity = model(noise, time, v, c)
            reconstructed = implied_x0(noise, time, velocity)
        return float(((reconstructed * v).sum(dim=-1, keepdim=True) - c).abs().mean())

    def step() -> float:
        x0, v, c = _toy_problem(rows, width, directions, generator)
        noise = torch.randn(rows, width, generator=generator)
        # t = 1: x_t is exactly the noise endpoint, u = eps - x_0.
        loss = flow_matching_loss(model(noise, time, v, c), noise - x0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    initial_error = coordinate_error(seed=99)
    initial_loss = sum(step() for _ in range(10)) / 10
    for _ in range(600):
        step()
    final_loss = sum(step() for _ in range(10)) / 10
    final_error = coordinate_error(seed=99)

    # 1. the flow loss falls substantially
    assert final_loss < 0.5 * initial_loss, (initial_loss, final_loss)
    # 4. the reconstructed coordinate tracks c_target far better after overfit
    assert final_error < 0.25 * initial_error, (initial_error, final_error)
    assert final_error < 0.35, final_error

    probe = torch.Generator().manual_seed(5)
    x_t = torch.randn(4, width, generator=probe)
    v = directions[0].expand(4, width)
    other = directions[1].expand(4, width)
    c_low = torch.full((4, 1), -2.0)
    c_high = torch.full((4, 1), 2.0)
    with torch.no_grad():
        low = model(x_t, torch.ones(4, 1), v, c_low)
        high = model(x_t, torch.ones(4, 1), v, c_high)
        swapped = model(x_t, torch.ones(4, 1), other, c_high)
        low_x0 = implied_x0(x_t, torch.ones(4, 1), low)
        high_x0 = implied_x0(x_t, torch.ones(4, 1), high)
        swapped_x0 = implied_x0(x_t, torch.ones(4, 1), swapped)

    # 2 & 5. same x_t, different c -> different prediction, in the requested direction
    assert not torch.allclose(low, high, atol=1e-3)
    low_coordinate = (low_x0 * v).sum(dim=-1)
    high_coordinate = (high_x0 * v).sum(dim=-1)
    assert bool((high_coordinate > low_coordinate + 2.0).all()), (
        low_coordinate.tolist(),
        high_coordinate.tolist(),
    )
    # 3. implied x_0 moves along v, tracking the requested coordinate
    assert torch.allclose(high_coordinate, torch.full((4,), 2.0), atol=0.5)
    assert torch.allclose(low_coordinate, torch.full((4,), -2.0), atol=0.5)
    # 6. same x_t and c, different v -> different conditioned reconstruction
    assert not torch.allclose(high, swapped, atol=1e-3)
    assert torch.allclose(
        (swapped_x0 * other).sum(dim=-1), torch.full((4,), 2.0), atol=0.5
    )
