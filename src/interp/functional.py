"""Per-sequence functional activation metric based on next-token LM loss."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from .model import STEERING_HOOK

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

Transform = Callable[[torch.Tensor], torch.Tensor]


def _substitution_hook(transform: Transform, *, skip_bos: bool):
    def apply(activation: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001, ARG001
        selected = activation[:, 1:, :] if skip_bos else activation
        transformed = transform(selected)
        if transformed.shape != selected.shape:
            raise ValueError(
                f"transform shape {tuple(transformed.shape)} != activation shape "
                f"{tuple(selected.shape)}"
            )
        if not transformed.is_floating_point() or not torch.isfinite(transformed).all():
            raise ValueError("transformed activation must be finite and floating point")
        if not skip_bos:
            return transformed
        output = activation.clone()
        output[:, 1:, :] = transformed
        return output

    return apply


@torch.no_grad()
def sequence_lm_losses(
    model: HookedTransformer,
    tokens: torch.Tensor,
    transform: Transform | None = None,
    *,
    hook: str = STEERING_HOOK,
    skip_bos: bool = True,
    batch_size: int = 8,
) -> torch.Tensor:
    """Return one mean next-token cross-entropy value per input sequence."""

    if (
        tokens.ndim != 2
        or tokens.shape[0] == 0
        or tokens.shape[1] < 2
        or tokens.is_floating_point()
    ):
        raise ValueError("tokens must be a nonempty integer [sequence, position>=2] tensor")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    hooks = [] if transform is None else [(hook, _substitution_hook(transform, skip_bos=skip_bos))]
    losses: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], batch_size):
        token_batch = tokens[start : start + batch_size].to(model.cfg.device)
        with model.hooks(fwd_hooks=hooks):
            logits = model(token_batch, return_type="logits")
        expected_shape = (token_batch.shape[0], token_batch.shape[1])
        if logits.ndim != 3 or logits.shape[:2] != expected_shape:
            raise ValueError(
                f"LM logits shape {tuple(logits.shape)} is incompatible with tokens "
                f"{tuple(token_batch.shape)}"
            )
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError("LM logits must be finite and floating point")
        token_losses = F.cross_entropy(
            logits[:, :-1, :].transpose(1, 2),
            token_batch[:, 1:],
            reduction="none",
        )
        sequence_losses = token_losses.mean(dim=1)
        if not torch.isfinite(sequence_losses).all():
            raise ValueError("per-sequence LM losses must be finite")
        losses.append(sequence_losses.detach().to(device="cpu", dtype=torch.float64))
    return torch.cat(losses)


@torch.no_grad()
def sequence_delta_lm(
    model: HookedTransformer,
    tokens: torch.Tensor,
    transform: Transform,
    *,
    hook: str = STEERING_HOOK,
    skip_bos: bool = True,
    batch_size: int = 8,
    clean: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return paired clean, transformed, and transformed-minus-clean sequence losses."""

    kwargs = {"hook": hook, "skip_bos": skip_bos, "batch_size": batch_size}
    baseline = sequence_lm_losses(model, tokens, **kwargs) if clean is None else clean
    if baseline.shape != (tokens.shape[0],) or not torch.isfinite(baseline).all():
        raise ValueError("clean losses must be one finite value per input sequence")
    transformed = sequence_lm_losses(model, tokens, transform, **kwargs)
    return {
        "clean": baseline.detach().to(device="cpu", dtype=torch.float64),
        "transformed": transformed,
        "delta": transformed - baseline,
    }
