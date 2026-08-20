"""Independent clean metrics for Phase B continuations."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .phase_b import FrozenSAE

_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class TextMetrics:
    lexicon_score: float
    dist_1: float
    dist_2: float
    dist_3: float
    rep_3: float
    repetition_rate: float
    n_words: int


@dataclass(frozen=True)
class ModelMetrics:
    nll: float
    continuation_token_ids: tuple[int, ...]
    prompt_token_count: int
    sae_feature_means: tuple[float, ...]


def _distinct(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def text_metrics(text: str, lexicon: tuple[str, ...]) -> TextMetrics:
    """Compute the frozen word metrics without importing any legacy evaluator."""

    if not isinstance(text, str):
        raise ValueError("metric input must be text")
    words = _WORD_RE.findall(text.lower())
    entries = tuple(item.lower() for item in lexicon)
    exact = {item for item in entries if not item.endswith("*")}
    prefixes = tuple(item[:-1] for item in entries if item.endswith("*"))
    hits = sum(
        word in exact or any(word.startswith(prefix) for prefix in prefixes) for word in words
    )
    dist_1 = _distinct(words, 1)
    dist_2 = _distinct(words, 2)
    dist_3 = _distinct(words, 3)
    repetitions = sum(left == right for left, right in zip(words, words[1:], strict=False))
    return TextMetrics(
        lexicon_score=hits / len(words) if words else 0.0,
        dist_1=dist_1,
        dist_2=dist_2,
        dist_3=dist_3,
        rep_3=1.0 - dist_3,
        repetition_rate=repetitions / (len(words) - 1) if len(words) > 1 else 0.0,
        n_words=len(words),
    )


def masked_continuation_nll(
    logits: torch.Tensor, targets: torch.Tensor, continuation_mask: torch.Tensor
) -> torch.Tensor:
    """Mean NLL for each row over exactly the selected continuation targets."""

    if (
        logits.ndim != 3
        or targets.shape != logits.shape[:2]
        or continuation_mask.shape != targets.shape
        or targets.dtype != torch.long
        or continuation_mask.dtype != torch.bool
    ):
        raise ValueError("malformed logits, targets, or continuation mask")
    if not torch.isfinite(logits).all():
        raise ValueError("NLL logits must be finite")
    counts = continuation_mask.sum(dim=1)
    if not bool((counts > 0).all()):
        raise ValueError("each row needs at least one continuation token")
    token_nll = -torch.log_softmax(logits.double(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    return (token_nll * continuation_mask).sum(dim=1) / counts


def masked_feature_means(features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean SAE features over valid continuation positions only."""

    if features.ndim != 3 or valid_mask.shape != features.shape[:2]:
        raise ValueError("malformed SAE features or valid-position mask")
    if not features.is_floating_point() or valid_mask.dtype != torch.bool:
        raise ValueError("SAE features must be floating point and mask boolean")
    if not torch.isfinite(features).all():
        raise ValueError("SAE features must be finite")
    counts = valid_mask.sum(dim=1)
    if not bool((counts > 0).all()):
        raise ValueError("each row needs at least one valid continuation token")
    return (features.double() * valid_mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


def _groups_by_width(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[len(row)].append(index)
    return tuple(tuple(groups[width]) for width in sorted(groups))


@torch.no_grad()
def score_continuations(
    model,  # noqa: ANN001
    sae: FrozenSAE,
    hook: str,
    prompts: list[str],
    continuations: list[str],
    feature_ids: list[int],
    batch_size: int = 16,
) -> tuple[ModelMetrics, ...]:
    """Score decoded text with clean GPT-2/SAE in exact-width, padding-free batches."""

    if not prompts or len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must be nonempty and paired")
    if not feature_ids or len(set(feature_ids)) != len(feature_ids):
        raise ValueError("metric feature IDs must be nonempty and unique")
    if batch_size < 1:
        raise ValueError("metric batch_size must be positive")
    tokenizer = model.tokenizer
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("clean GPT-2 tokenizer has no BOS token")
    prompt_ids = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
    continuation_ids = [
        tokenizer.encode(text, add_special_tokens=False) for text in continuations
    ]
    if any(not row for row in continuation_ids):
        raise ValueError("every decoded continuation must contain at least one metric token")

    sequences = [
        [bos, *prompt, *continuation]
        for prompt, continuation in zip(prompt_ids, continuation_ids, strict=True)
    ]
    nlls = [float("nan")] * len(sequences)
    for group in _groups_by_width(sequences):
        for start in range(0, len(group), batch_size):
            indices = group[start : start + batch_size]
            tokens = torch.tensor(
                [sequences[index] for index in indices],
                dtype=torch.long,
                device=model.cfg.device,
            )
            logits = model(tokens, return_type="logits")
            mask = torch.zeros(
                tokens.shape[0], tokens.shape[1] - 1, dtype=torch.bool, device=tokens.device
            )
            for batch_index, original_index in enumerate(indices):
                start_prediction = len(prompt_ids[original_index])
                mask[batch_index, start_prediction:] = True
            values = masked_continuation_nll(logits[:, :-1], tokens[:, 1:], mask)
            for original_index, value in zip(indices, values.tolist(), strict=True):
                nlls[original_index] = float(value)

    sae_means: list[tuple[float, ...] | None] = [None] * len(sequences)
    sae_sequences = [[bos, *row] for row in continuation_ids]
    for group in _groups_by_width(sae_sequences):
        for start in range(0, len(group), batch_size):
            indices = group[start : start + batch_size]
            tokens = torch.tensor(
                [sae_sequences[index] for index in indices],
                dtype=torch.long,
                device=model.cfg.device,
            )
            _, cache = model.run_with_cache(tokens, names_filter=[hook])
            features = sae.encode_features(cache[hook][:, 1:, :], feature_ids)
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
            values = masked_feature_means(features, mask).cpu()
            for original_index, value in zip(indices, values, strict=True):
                sae_means[original_index] = tuple(float(item) for item in value)

    if any(item is None for item in sae_means) or not all(torch.isfinite(torch.tensor(nlls))):
        raise ValueError("clean continuation scoring produced incomplete or non-finite metrics")
    return tuple(
        ModelMetrics(
            nll=nlls[index],
            continuation_token_ids=tuple(continuation_ids[index]),
            prompt_token_count=len(prompt_ids[index]) + 1,
            sae_feature_means=sae_means[index],  # type: ignore[arg-type]
        )
        for index in range(len(sequences))
    )
