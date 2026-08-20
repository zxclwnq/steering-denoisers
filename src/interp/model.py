"""Canonical GPT-2 loading and activation hook constants."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

MODEL_NAME = "gpt2-small"
MODEL_RESOLVED_NAME = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MID_LAYER = 6
STEERING_HOOK = f"blocks.{MID_LAYER + 1}.hook_resid_pre"


def resolve_device(device: str | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def load_model(
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    model_name: str = MODEL_NAME,
) -> HookedTransformer:
    """Load the frozen model interface in evaluation mode with gradients disabled."""

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_RESOLVED_NAME,
        revision=MODEL_REVISION,
    )
    model = HookedTransformer.from_pretrained(
        model_name,
        device=str(resolve_device(device)),
        dtype=dtype,
        revision=MODEL_REVISION,
        tokenizer=tokenizer,
    )
    model.eval()
    model.requires_grad_(False)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    if model.cfg.n_layers <= MID_LAYER + 1:
        raise ValueError(f"model has too few layers for hook {STEERING_HOOK}")
    if model_name == MODEL_NAME and model.cfg.model_name != MODEL_RESOLVED_NAME:
        raise ValueError(
            f"canonical model resolved as {model.cfg.model_name!r}, "
            f"expected {MODEL_RESOLVED_NAME!r}"
        )
    if STEERING_HOOK not in model.hook_dict:
        raise ValueError(f"hook {STEERING_HOOK} is absent from the loaded model")
    return model
