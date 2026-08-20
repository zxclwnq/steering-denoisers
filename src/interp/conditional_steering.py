"""Phase B-C generation session for the conditional flow prior.

Implements docs/PHASE_B_CONDITIONAL_PROTOCOL.md section 4: the requested
coordinate reaches the model only through the condition, so the prior is seeded
with clean ``h`` and never with ``h + alpha*v``.

Everything else -- position bookkeeping, matched Gaussian noise, correction
geometry, per-cell receipts -- is inherited unchanged from the frozen
unconditional session, so the two methods stay comparable by construction.
"""

from __future__ import annotations

import torch

from .conditional_flow import ConditionalFlowMatcher, conditional_clamp_steer
from .flow_steering import FlowGenerationSession, FlowSteeringOutput

SEED_MODES = ("clean", "clamp")


class ConditionalFlowGenerationSession(FlowGenerationSession):
    """Steer by requesting a coordinate through the conditional prior.

    The off-distribution guard keeps the frozen criterion deliberately: a
    position is guarded when the *additive* steer would exceed
    ``off_distribution_norm``, and a guarded position falls back to the additive
    steer. Using the same criterion and the same fallback as the unconditional
    and additive arms means the arms guard the identical set of positions, which
    is what makes their matched comparison valid.
    """

    def __init__(
        self,
        model: ConditionalFlowMatcher,
        direction: torch.Tensor,
        *,
        seed_mode: str = "clean",
        coordinate_mode: str = "additive",
        **kwargs: object,
    ) -> None:
        if seed_mode not in SEED_MODES:
            raise ValueError(f"unknown seed_mode {seed_mode!r}, expected one of {SEED_MODES}")
        if not hasattr(model, "velocity_field"):
            raise ValueError("conditional steering requires a ConditionalFlowMatcher")
        super().__init__(model, direction, **kwargs)  # type: ignore[arg-type]
        self.seed_mode = seed_mode
        self.coordinate_mode = coordinate_mode
        self._coordinate_abs_error: list[tuple[int, float]] = []

    @torch.no_grad()
    def _steer(self, valid_h: torch.Tensor, valid_noise: torch.Tensor) -> FlowSteeringOutput:
        vector = self.direction.to(device=valid_h.device, dtype=valid_h.dtype)
        result = conditional_clamp_steer(
            self.model,
            valid_h,
            vector,
            alpha=self.alpha,
            mode=self.coordinate_mode,
            noise=valid_noise,
            t_start=self.t_start,
            nfe=self.nfe,
            seed_mode=self.seed_mode,
        )
        additive = valid_h + self.alpha * vector
        guarded = additive.norm(dim=-1) > self.off_distribution_norm
        activation = torch.where(guarded.unsqueeze(-1), additive, result.activation)
        if not torch.isfinite(activation).all():
            raise ValueError("conditional flow steering produced a non-finite activation")
        self._coordinate_abs_error.append(
            (int(valid_h.shape[0]), float(result.coordinate_error.abs().double().mean()))
        )
        return FlowSteeringOutput(
            activation=activation,
            guarded=guarded,
            network_evaluations=0 if self.t_start == 0.0 else self.nfe,
        )

    def coordinate_receipt(self) -> dict[str, float | int | str]:
        """Mean |realized - requested| coordinate error, the C1 diagnostic."""

        if not self._coordinate_abs_error:
            raise ValueError("generation session has not observed a hook call")
        total = sum(count for count, _ in self._coordinate_abs_error)
        weighted = sum(count * value for count, value in self._coordinate_abs_error)
        return {
            "seed_mode": self.seed_mode,
            "coordinate_mode": self.coordinate_mode,
            "positions": total,
            "coordinate_abs_error_mean": weighted / total,
        }
