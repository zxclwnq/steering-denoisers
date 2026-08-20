"""Experiment D must not silently stop being a positive control.

The failure modes worth testing here are not type errors. They are: the direction
drifting away from the published one, the intervention conventions being
"tidied up" into something weaker, the token position sliding onto padding, and
the causal gate passing when only one half of the effect reproduced.

CPU, tiny, synthetic. No gemma download, no GPU, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from interp.refusal_control import (
    DIRECTION_NORM,
    DIRECTION_SHA256,
    GEMMA_CHAT_TEMPLATE,
    REFUSAL_CONTROL_SPEC,
    REFUSAL_SUBSTRINGS,
    _ablate,
    _addition_pre_hook,
    causal_validation_verdict,
    end_of_instruction_length,
    format_instruction,
    is_refusal,
    load_refusal_direction,
    refusal_rate,
    refusal_score,
    tokenize_instructions,
)

REPO = Path(__file__).resolve().parents[1]
DIRECTION = REPO / "data" / "refusal_direction" / "gemma-2b-it_direction.pt"
METADATA = REPO / "data" / "refusal_direction" / "gemma-2b-it_direction_metadata.json"


# --------------------------------------------------------------------------
# the direction is the published one, or the experiment does not run
# --------------------------------------------------------------------------


def test_the_published_direction_loads_with_its_recorded_identity() -> None:
    direction = load_refusal_direction(DIRECTION, METADATA)
    assert direction.sha256 == DIRECTION_SHA256
    assert direction.layer == 10
    assert direction.position == -2
    assert float(direction.raw.norm()) == pytest.approx(DIRECTION_NORM, abs=1e-9)
    assert float(direction.unit.norm()) == pytest.approx(1.0, abs=1e-12)
    assert direction.raw.shape == (2048,)


def test_the_raw_scale_is_preserved_because_it_is_the_intervention_magnitude() -> None:
    """Normalizing the stored vector would silently change the published arm."""

    direction = load_refusal_direction(DIRECTION, METADATA)
    assert float(direction.raw.norm()) > 10.0
    assert not torch.allclose(direction.raw, direction.unit)


def test_a_direction_that_is_not_the_published_one_is_refused(tmp_path: Path) -> None:
    impostor = tmp_path / "direction.pt"
    torch.save(torch.randn(2048, dtype=torch.float64), impostor)
    with pytest.raises(ValueError, match="not the published direction"):
        load_refusal_direction(impostor)


def test_metadata_disagreeing_with_the_frozen_spec_is_refused(tmp_path: Path) -> None:
    """A re-selected layer would quietly turn D into a different experiment."""

    meta = tmp_path / "meta.json"
    meta.write_text('{"pos": -1, "layer": 14}')
    with pytest.raises(ValueError, match="frozen spec is layer 10 pos -2"):
        load_refusal_direction(DIRECTION, meta)


# --------------------------------------------------------------------------
# prompt formatting and position
# --------------------------------------------------------------------------


def test_the_chat_template_is_the_published_one() -> None:
    formatted = format_instruction("Tell me a joke")
    assert formatted.startswith("<start_of_turn>user\n")
    assert "Tell me a joke<end_of_turn>" in formatted
    assert formatted.endswith("<start_of_turn>model\n")
    # trailing whitespace is part of the published convention
    assert GEMMA_CHAT_TEMPLATE.endswith("\n")


class _StubTokenizer:
    """Just enough tokenizer to exercise padding side and suffix length."""

    padding_side = "left"

    def __call__(self, prompts, padding, truncation, return_tensors):
        lengths = [len(p.split()) for p in prompts]
        width = max(lengths)
        ids = torch.zeros(len(prompts), width, dtype=torch.long)
        mask = torch.zeros(len(prompts), width, dtype=torch.long)
        for row, length in enumerate(lengths):
            ids[row, width - length:] = torch.arange(1, length + 1)
            mask[row, width - length:] = 1
        return type("Batch", (), {"input_ids": ids, "attention_mask": mask})()

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def test_right_padding_is_refused_because_it_would_move_the_position() -> None:
    """The direction lives at a position counted from the end of the prompt."""

    tokenizer = _StubTokenizer()
    tokenizer.padding_side = "right"
    with pytest.raises(ValueError, match="pads on the left"):
        tokenize_instructions(tokenizer, ["a b", "c"])


def test_left_padding_keeps_the_final_positions_aligned() -> None:
    batch = tokenize_instructions(_StubTokenizer(), ["a b c", "d"])
    assert batch.attention_mask[:, -1].tolist() == [1, 1]
    assert batch.attention_mask[1, 0].item() == 0


def test_the_end_of_instruction_span_is_read_from_the_template() -> None:
    assert end_of_instruction_length(_StubTokenizer()) >= 2


# --------------------------------------------------------------------------
# interventions
# --------------------------------------------------------------------------


def test_ablation_removes_exactly_the_component_along_the_direction() -> None:
    torch.manual_seed(0)
    activation = torch.randn(4, 7, 16)
    direction = torch.randn(16)
    out = _ablate(activation, direction)
    unit = direction / direction.norm()
    assert torch.allclose(out @ unit, torch.zeros(4, 7), atol=1e-5)
    # and it changes nothing orthogonal to the direction
    residual = activation - (activation @ unit).unsqueeze(-1) * unit
    assert torch.allclose(out, residual, atol=1e-5)


def test_ablation_is_scale_invariant_in_the_supplied_direction() -> None:
    """Ablation normalizes internally, so the raw norm must not matter here."""

    torch.manual_seed(1)
    activation = torch.randn(2, 3, 8)
    direction = torch.randn(8)
    assert torch.allclose(
        _ablate(activation, direction), _ablate(activation, 10.0 * direction), atol=1e-5
    )


def test_addition_is_not_scale_invariant_and_uses_the_vector_as_given() -> None:
    """Addition does NOT normalize: the published magnitude is the raw norm."""

    activation = torch.zeros(1, 1, 4)
    vector = torch.tensor([3.0, 0.0, 0.0, 0.0])
    hook = _addition_pre_hook(vector, 1.0)
    out = hook(None, (activation,))[0]
    assert out[0, 0, 0].item() == pytest.approx(3.0)
    doubled = _addition_pre_hook(2.0 * vector, 1.0)(None, (activation,))[0]
    assert doubled[0, 0, 0].item() == pytest.approx(6.0)


def test_the_spec_records_the_published_ablation_scope() -> None:
    """Narrowing the scope to one layer would not reproduce the paper's effect."""

    assert REFUSAL_CONTROL_SPEC.ablation_scope == (
        "all_layers_blocks_attn_mlp_all_positions"
    )
    assert REFUSAL_CONTROL_SPEC.addition_coefficient == 1.0


# --------------------------------------------------------------------------
# refusal metrics
# --------------------------------------------------------------------------


def test_the_refusal_score_is_a_log_odds_at_the_last_position() -> None:
    logits = torch.full((1, 3, 300000), -20.0, dtype=torch.float32)
    logits[0, -1, 235285] = 20.0
    score = refusal_score(logits)
    assert score.shape == (1,)
    assert score.item() > 10.0
    logits[0, -1, 235285] = -40.0
    assert refusal_score(logits).item() < 0.0


def test_the_refusal_score_reads_the_last_position_not_an_earlier_one() -> None:
    logits = torch.full((1, 3, 300000), -20.0, dtype=torch.float32)
    logits[0, 0, 235285] = 40.0  # a refusal signal in the wrong place
    assert refusal_score(logits).item() < 0.0


def test_the_substring_judge_matches_the_published_prefix_list() -> None:
    assert len(REFUSAL_SUBSTRINGS) == 12
    for substring in REFUSAL_SUBSTRINGS:
        assert is_refusal(f"prefix {substring} suffix")
    assert not is_refusal("Sure! Here is a detailed answer.")
    # case-insensitive, as upstream
    assert is_refusal("i cannot do that")


def test_the_refusal_rate_is_a_percentage() -> None:
    assert refusal_rate(["I cannot", "sure", "I apologize", "ok"]) == 50.0
    with pytest.raises(ValueError, match="no completions"):
        refusal_rate([])


# --------------------------------------------------------------------------
# the causal gate
# --------------------------------------------------------------------------


def _verdict(**overrides):
    base = {
        "harmful_baseline_rate": 95.0,
        "harmful_ablated_rate": 10.0,
        "harmless_baseline_rate": 2.0,
        "harmless_addition_rate": 90.0,
        "harmful_baseline_score": 3.0,
        "harmful_ablated_score": -2.0,
        "harmless_baseline_score": -3.0,
        "harmless_addition_score": 2.0,
    }
    base.update(overrides)
    return causal_validation_verdict(**base)


def test_both_halves_of_the_effect_reproducing_is_a_pass() -> None:
    verdict = _verdict()
    assert verdict["verdict"] == "CAUSAL_CONTROL_PASS"
    assert verdict["ablation_effect_reproduced"] is True
    assert verdict["addition_effect_reproduced"] is True
    assert verdict["ablation_refusal_drop_pp"] == pytest.approx(85.0)


def test_removing_refusal_without_inducing_it_is_not_a_pass() -> None:
    """A big enough perturbation breaks a model; that is not mediation."""

    verdict = _verdict(harmless_addition_rate=3.0, harmless_addition_score=-2.9)
    assert verdict["verdict"] == "CAUSAL_CONTROL_FAIL"
    assert verdict["ablation_effect_reproduced"] is True
    assert verdict["addition_effect_reproduced"] is False
    assert "stops" in verdict["consequence"]


def test_inducing_refusal_without_removing_it_is_not_a_pass() -> None:
    verdict = _verdict(harmful_ablated_rate=94.0, harmful_ablated_score=2.9)
    assert verdict["verdict"] == "CAUSAL_CONTROL_FAIL"
    assert verdict["addition_effect_reproduced"] is True


def test_a_rate_change_with_the_wrong_score_direction_does_not_pass() -> None:
    """The two metrics must agree, or the rate change is not about refusal."""

    verdict = _verdict(harmful_ablated_score=3.5)
    assert verdict["ablation_effect_reproduced"] is False
    assert verdict["verdict"] == "CAUSAL_CONTROL_FAIL"


def test_the_threshold_is_the_frozen_one_and_a_near_miss_fails() -> None:
    assert REFUSAL_CONTROL_SPEC.min_ablation_refusal_drop == 50.0
    verdict = _verdict(harmful_ablated_rate=95.0 - 49.9)
    assert verdict["ablation_effect_reproduced"] is False


def test_the_geometry_never_influences_the_gate() -> None:
    """The gate reads behaviour only: no geometry term may appear in it."""

    verdict = _verdict()
    forbidden = ("cos", "secant", "curvature", "shortfall", "alignment")
    assert not any(word in key for key in verdict for word in forbidden)


# --------------------------------------------------------------------------
# the four preregistered outcomes
# --------------------------------------------------------------------------


def _null(above: bool = False, below: bool = False, usable: bool = True) -> dict:
    return {
        "usable": usable,
        "above_random_interval": above,
        "below_random_interval": below,
    }


def _contrast(mean: float, upper: float, usable: bool = True) -> dict:
    return {"usable": usable, "mean": mean, "ci_lower": mean - 0.05, "ci_upper": upper}


def test_alignment_above_random_everywhere_is_the_aligned_outcome() -> None:
    from interp.refusal_control import geometry_interpretation

    result = geometry_interpretation(
        pooled_alignment_vs_random=_null(above=True),
        central_alignment_vs_random=_null(above=True),
        tail_minus_central=_contrast(0.02, 0.06),
        curvature_vs_random=_null(),
        shortfall_vs_random=_null(),
    )
    assert result["outcome"] == "CAUSAL_DIRECTION_ALIGNED"
    assert "Positive control only" in result["why"]


def test_alignment_that_peaks_centrally_and_falls_is_the_local_outcome() -> None:
    """Checked before the global one: it is the more specific claim."""

    from interp.refusal_control import geometry_interpretation

    result = geometry_interpretation(
        pooled_alignment_vs_random=_null(above=True),
        central_alignment_vs_random=_null(above=True),
        tail_minus_central=_contrast(-0.20, -0.10),
        curvature_vs_random=_null(),
        shortfall_vs_random=_null(),
    )
    assert result["outcome"] == "LOCAL_ALIGNMENT_ONLY"
    assert result["alignment_falls_away_from_the_centre"] is True


def test_a_causal_direction_with_ordinary_alignment_is_the_negative_outcome() -> None:
    """The result that weakens our own geometric story, and must stay reachable."""

    from interp.refusal_control import geometry_interpretation

    result = geometry_interpretation(
        pooled_alignment_vs_random=_null(),
        central_alignment_vs_random=_null(),
        tail_minus_central=_contrast(0.01, 0.05),
        curvature_vs_random=_null(below=True),
        shortfall_vs_random=_null(above=True),
    )
    assert result["outcome"] == "CAUSAL_BUT_NOT_ALIGNED"
    assert "not a necessary condition" in result["why"]
    assert "merely correlational" in result["why"]


def test_geometry_matching_random_on_everything_is_the_uninformative_outcome() -> None:
    from interp.refusal_control import geometry_interpretation

    result = geometry_interpretation(
        pooled_alignment_vs_random=_null(),
        central_alignment_vs_random=_null(),
        tail_minus_central=_contrast(0.0, 0.04),
        curvature_vs_random=_null(),
        shortfall_vs_random=_null(),
    )
    assert result["outcome"] == "GEOMETRY_NOT_INFORMATIVE_FOR_CAUSALITY"


def test_an_unusable_comparison_never_counts_as_a_difference() -> None:
    """A NaN estimate must not be promoted into evidence of anything."""

    from interp.refusal_control import geometry_interpretation

    result = geometry_interpretation(
        pooled_alignment_vs_random=_null(above=True, usable=False),
        central_alignment_vs_random=_null(above=True, usable=False),
        tail_minus_central=_contrast(-0.2, -0.1, usable=False),
        curvature_vs_random=_null(above=True, usable=False),
        shortfall_vs_random=_null(usable=False),
    )
    assert result["outcome"] == "GEOMETRY_NOT_INFORMATIVE_FOR_CAUSALITY"
    assert result["any_statistic_differs_from_random"] is False


def test_every_outcome_carries_the_prohibited_claims() -> None:
    from interp.refusal_control import geometry_interpretation

    result = geometry_interpretation(
        _null(), _null(), _contrast(0.0, 0.1), _null(), _null()
    )
    assert "SAE features are merely correlational" in result["prohibited_claims"]
    assert "any cross-model causal taxonomy" in result["prohibited_claims"]


def test_direction_and_spec_agree_with_the_frozen_protocol() -> None:
    assert REFUSAL_CONTROL_SPEC.model_path == "google/gemma-2b-it"
    assert REFUSAL_CONTROL_SPEC.layer == 10
    assert REFUSAL_CONTROL_SPEC.position == -2
    assert np.isclose(REFUSAL_CONTROL_SPEC.addition_coefficient, 1.0)
