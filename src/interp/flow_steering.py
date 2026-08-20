"""Clean Phase B SDEdit steering with position-stable paired Gaussian noise."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from .flow_sampling import NormalizedVelocityModel, flow_correct

DEFAULT_NOISE_NAMESPACE = "flow_noise_v2"


@dataclass(frozen=True)
class FlowNoiseCell:
    vector: str
    alpha: float
    prompt_id: int
    generation_seed: int

    def __post_init__(self) -> None:
        if not self.vector:
            raise ValueError("flow-noise vector identity must be nonempty")
        if not math.isfinite(self.alpha):
            raise ValueError("flow-noise alpha must be finite")
        for name in ("prompt_id", "generation_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def _position_seed(cell: FlowNoiseCell, position: int, namespace: str) -> int:
    key = "|".join(
        (
            namespace,
            cell.vector,
            float(cell.alpha).hex(),
            str(cell.prompt_id),
            str(cell.generation_seed),
            str(position),
        )
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


def matched_flow_noise(
    cells: tuple[FlowNoiseCell, ...] | list[FlowNoiseCell],
    token_positions: torch.Tensor,
    d_model: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    namespace: str = DEFAULT_NOISE_NAMESPACE,
) -> torch.Tensor:
    """Return epsilon keyed only by immutable cell and absolute token position."""

    if not cells or not all(isinstance(cell, FlowNoiseCell) for cell in cells):
        raise ValueError("cells must be a nonempty sequence of FlowNoiseCell values")
    if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model < 1:
        raise ValueError("d_model must be a positive integer")
    if not dtype.is_floating_point:
        raise ValueError("flow noise dtype must be floating point")
    if not namespace:
        raise ValueError("flow-noise namespace must be nonempty")
    if token_positions.ndim == 1:
        positions = token_positions[None, :].expand(len(cells), -1)
    elif token_positions.ndim == 2 and token_positions.shape[0] == len(cells):
        positions = token_positions
    else:
        raise ValueError("token_positions must be [position] or [cell, position]")
    if positions.shape[1] == 0 or positions.dtype == torch.bool or positions.is_floating_point():
        raise ValueError("token_positions must be a nonempty integer tensor")
    position_rows = positions.detach().cpu().tolist()
    if any(len(set(row)) != len(row) for row in position_rows):
        raise ValueError("token positions must be unique within each cell")

    resolved_device = torch.device(device)
    rows = []
    for cell, row in zip(cells, position_rows, strict=True):
        samples = []
        for position in row:
            generator = torch.Generator(device=resolved_device)
            generator.manual_seed(_position_seed(cell, int(position), namespace))
            samples.append(
                torch.randn(
                    d_model,
                    generator=generator,
                    device=resolved_device,
                    dtype=dtype,
                )
            )
        rows.append(torch.stack(samples))
    return torch.stack(rows)


@dataclass(frozen=True)
class FlowSteeringOutput:
    activation: torch.Tensor
    guarded: torch.Tensor
    network_evaluations: int


def _unit_direction(direction: torch.Tensor, d_model: int) -> torch.Tensor:
    if direction.ndim != 1 or direction.shape != (d_model,) or not direction.is_floating_point():
        raise ValueError(f"direction must be floating point with shape [{d_model}]")
    if not torch.isfinite(direction).all():
        raise ValueError("direction must be finite")
    norm = float(direction.double().norm())
    if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError(f"direction must be unit-normalized, got norm {norm}")
    return direction


@torch.no_grad()
def apply_flow_steering(
    model: NormalizedVelocityModel,
    h: torch.Tensor,
    direction: torch.Tensor,
    *,
    alpha: float,
    noise: torch.Tensor,
    t_start: float,
    nfe: int,
    off_distribution_norm: float,
) -> FlowSteeringOutput:
    """Apply additive steering first, then correct through the frozen flow."""

    if h.ndim not in (2, 3) or h.numel() == 0 or not h.is_floating_point():
        raise ValueError(
            "h must be a nonempty floating-point [position, d] or [batch, position, d]"
        )
    if not torch.isfinite(h).all():
        raise ValueError("h must be finite")
    if noise.shape != h.shape or not noise.is_floating_point() or not torch.isfinite(noise).all():
        raise ValueError("noise must be a same-shaped finite floating-point tensor")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not math.isfinite(off_distribution_norm) or off_distribution_norm <= 0:
        raise ValueError("off_distribution_norm must be finite and positive")
    vector = _unit_direction(direction, h.shape[-1]).to(device=h.device, dtype=h.dtype)

    additive = h + alpha * vector
    flat = additive.reshape(-1, additive.shape[-1])
    flat_noise = noise.to(device=h.device, dtype=h.dtype).reshape_as(flat)
    corrected = flow_correct(
        model,
        flat,
        noise=flat_noise,
        t_start=t_start,
        nfe=nfe,
    ).reshape_as(additive)
    guarded = additive.norm(dim=-1) > off_distribution_norm
    output = torch.where(guarded.unsqueeze(-1), additive, corrected)
    if not torch.isfinite(output).all():
        raise ValueError("flow-steered activation became non-finite")
    return FlowSteeringOutput(
        activation=output,
        guarded=guarded,
        network_evaluations=0 if t_start == 0.0 else nfe,
    )


@torch.no_grad()
def steering_geometry(
    clean: torch.Tensor,
    additive: torch.Tensor,
    corrected: torch.Tensor,
    direction: torch.Tensor,
    *,
    alpha: float,
) -> dict[str, float | int | None]:
    """Aggregate the frozen realized-steering and correction-geometry quantities."""

    if (
        clean.shape != additive.shape
        or clean.shape != corrected.shape
        or clean.ndim not in (2, 3)
        or clean.numel() == 0
        or not clean.is_floating_point()
        or not additive.is_floating_point()
        or not corrected.is_floating_point()
    ):
        raise ValueError("geometry requires same-shaped nonempty floating-point activations")
    if (
        not torch.isfinite(clean).all()
        or not torch.isfinite(additive).all()
        or not torch.isfinite(corrected).all()
    ):
        raise ValueError("geometry activations must be finite")
    if not math.isfinite(alpha):
        raise ValueError("geometry alpha must be finite")
    vector = _unit_direction(direction, clean.shape[-1]).to(device=clean.device, dtype=clean.dtype)
    clean_flat = clean.reshape(-1, clean.shape[-1])
    additive_flat = additive.reshape_as(clean_flat)
    corrected_flat = corrected.reshape_as(clean_flat)

    displacement = corrected_flat - clean_flat
    correction = corrected_flat - additive_flat
    realized = displacement @ vector
    parallel_signed = correction @ vector
    parallel_norm = parallel_signed.abs()
    orthogonal = correction - parallel_signed[:, None] * vector
    correction_norm = correction.norm(dim=-1)
    orthogonal_norm = orthogonal.norm(dim=-1)
    defined = correction_norm > 0
    cosine = parallel_signed[defined] / correction_norm[defined]
    result: dict[str, float | int | None] = {
        "n_positions": int(clean_flat.shape[0]),
        "realized_projection_mean": float(realized.double().mean()),
        "retained_fraction_mean": (
            None if alpha == 0.0 else float((realized / alpha).double().mean())
        ),
        "correction_norm_mean": float(correction_norm.double().mean()),
        "parallel_correction_norm_mean": float(parallel_norm.double().mean()),
        "orthogonal_correction_norm_mean": float(orthogonal_norm.double().mean()),
        "correction_cosine_mean": (
            None if not bool(defined.any()) else float(cosine.double().mean())
        ),
        "correction_cosine_defined": int(defined.sum()),
    }
    numeric = [value for value in result.values() if isinstance(value, float)]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("steering geometry became non-finite")
    return result


class FlowGenerationSession:
    """Resolve hook calls to stable prompt/generated positions for one generation cell."""

    def __init__(
        self,
        model: NormalizedVelocityModel,
        direction: torch.Tensor,
        *,
        alpha: float,
        t_start: float,
        nfe: int,
        cells: tuple[FlowNoiseCell, ...],
        prompt_width: int,
        max_new_tokens: int,
        off_distribution_norm: float,
        prompt_attention_mask: torch.Tensor | None = None,
        noise_namespace: str = DEFAULT_NOISE_NAMESPACE,
    ) -> None:
        if not cells or len({cell.prompt_id for cell in cells}) != len(cells):
            raise ValueError("generation cells must have unique prompt IDs")
        if (
            len({cell.vector for cell in cells}) != 1
            or len({cell.generation_seed for cell in cells}) != 1
        ):
            raise ValueError("one session must share vector and generation seed")
        if any(cell.alpha != alpha for cell in cells):
            raise ValueError("noise-cell alpha must exactly match steering alpha")
        if prompt_width < 1 or max_new_tokens < 1:
            raise ValueError("prompt_width and max_new_tokens must be positive")
        _unit_direction(direction, direction.numel())
        self.model = model
        self.direction = direction
        self.alpha = float(alpha)
        self.t_start = float(t_start)
        self.nfe = int(nfe)
        self.cells = cells
        self.prompt_width = int(prompt_width)
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones(
                len(cells), prompt_width, dtype=torch.bool
            )
        if (
            prompt_attention_mask.shape != (len(cells), prompt_width)
            or prompt_attention_mask.dtype != torch.bool
            or not bool(prompt_attention_mask.any(dim=1).all())
        ):
            raise ValueError("prompt_attention_mask must select tokens in every prompt row")
        self.prompt_attention_mask = prompt_attention_mask.detach().cpu()
        self.max_new_tokens = int(max_new_tokens)
        self.off_distribution_norm = float(off_distribution_norm)
        self.noise_namespace = noise_namespace
        self._hook_calls = 0
        self._flow_evaluations = 0
        self._guarded = 0
        self._geometry: list[dict[str, float | int | None]] = []
        self._cell_geometry: list[list[dict[str, float | int | None]]] = [
            [] for _ in cells
        ]
        self._cell_guarded = [0 for _ in cells]
        self._cell_noise: list[dict[int, bytes]] = [{} for _ in cells]
        self._cell_realized: list[list[torch.Tensor]] = [[] for _ in cells]

    def _positions_and_mask(
        self, sequence_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._hook_calls >= self.max_new_tokens:
            raise ValueError("more hook calls than the frozen generation length permits")
        if self._hook_calls == 0:
            if sequence_length != self.prompt_width:
                raise ValueError("initial hook shape does not match the frozen prompt width")
            valid = self.prompt_attention_mask.to(device)
            positions = torch.zeros_like(valid, dtype=torch.long)
            for row in range(len(self.cells)):
                length = int(valid[row].sum())
                positions[row, valid[row]] = torch.arange(-length, 0, device=device)
            return positions, valid
        if sequence_length == 1:
            positions = torch.full(
                (len(self.cells), 1), self._hook_calls - 1, device=device, dtype=torch.long
            )
            return positions, torch.ones_like(positions, dtype=torch.bool)
        expected_full = self.prompt_width + self._hook_calls
        if sequence_length == expected_full:
            prompt_positions, prompt_valid = self._positions_for_prompt(device)
            generated = torch.arange(self._hook_calls, device=device)[None, :].expand(
                len(self.cells), -1
            )
            positions = torch.cat((prompt_positions, generated), dim=1)
            valid = torch.cat(
                (
                    prompt_valid,
                    torch.ones(len(self.cells), self._hook_calls, dtype=torch.bool, device=device),
                ),
                dim=1,
            )
            return positions, valid
        raise ValueError("ambiguous generation hook shape")

    def _positions_for_prompt(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        valid = self.prompt_attention_mask.to(device)
        positions = torch.zeros_like(valid, dtype=torch.long)
        for row in range(len(self.cells)):
            length = int(valid[row].sum())
            positions[row, valid[row]] = torch.arange(-length, 0, device=device)
        return positions, valid

    @torch.no_grad()
    def _steer(self, valid_h: torch.Tensor, valid_noise: torch.Tensor) -> FlowSteeringOutput:
        """The frozen unconditional SDEdit transform. Subclasses may replace it."""

        return apply_flow_steering(
            self.model,
            valid_h,
            self.direction,
            alpha=self.alpha,
            noise=valid_noise,
            t_start=self.t_start,
            nfe=self.nfe,
            off_distribution_norm=self.off_distribution_norm,
        )

    @torch.no_grad()
    def apply(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3 or h.shape[0] != len(self.cells):
            raise ValueError("generation hook shape must be [cell, position, d_model]")
        positions, valid = self._positions_and_mask(h.shape[1], h.device)
        noise_rows = [
            matched_flow_noise(
                (cell,),
                positions[index, valid[index]],
                h.shape[-1],
                h.device,
                h.dtype,
                self.noise_namespace,
            )[0]
            for index, cell in enumerate(self.cells)
        ]
        valid_h = h[valid]
        valid_noise = torch.cat(noise_rows, dim=0)
        result = self._steer(valid_h, valid_noise)
        vector = self.direction.to(device=h.device, dtype=h.dtype)
        output = h.clone()
        output[valid] = result.activation
        geometry = steering_geometry(
            valid_h,
            valid_h + self.alpha * vector,
            result.activation,
            vector,
            alpha=self.alpha,
        )
        self._geometry.append(geometry)
        offset = 0
        for index, (cell, row_positions) in enumerate(zip(self.cells, positions, strict=True)):
            count = int(valid[index].sum())
            row_activation = result.activation[offset : offset + count]
            row_guarded = result.guarded[offset : offset + count]
            row_noise = noise_rows[index]
            offset += count
            row_geometry = steering_geometry(
                h[index, valid[index]],
                h[index, valid[index]] + self.alpha * vector,
                row_activation,
                vector,
                alpha=self.alpha,
            )
            self._cell_geometry[index].append(row_geometry)
            self._cell_guarded[index] += int(row_guarded.sum())
            realized = (row_activation - h[index, valid[index]]) @ vector
            self._cell_realized[index].append(realized.detach().double().cpu())
            valid_positions = row_positions[valid[index]].tolist()
            for position, sample in zip(valid_positions, row_noise, strict=True):
                payload = sample.detach().float().cpu().contiguous().numpy().tobytes()
                previous = self._cell_noise[index].setdefault(int(position), payload)
                if previous != payload:
                    raise ValueError(
                        f"epsilon changed within flow cell {cell.prompt_id} at {position}"
                    )
        self._hook_calls += 1
        self._flow_evaluations += result.network_evaluations
        self._guarded += int(result.guarded.sum())
        return output

    @staticmethod
    def _aggregate_geometry(
        items: list[dict[str, float | int | None]],
    ) -> dict[str, float | int | None]:
        n_positions = sum(int(item["n_positions"]) for item in items)
        mean_keys = (
            "realized_projection_mean",
            "correction_norm_mean",
            "parallel_correction_norm_mean",
            "orthogonal_correction_norm_mean",
        )
        geometry: dict[str, float | int | None] = {"n_positions": n_positions}
        for key in mean_keys:
            geometry[key] = sum(
                float(item[key]) * int(item["n_positions"]) for item in items
            ) / n_positions
        retained = [item["retained_fraction_mean"] for item in items]
        geometry["retained_fraction_mean"] = (
            None
            if any(value is None for value in retained)
            else sum(
                float(value) * int(item["n_positions"])
                for value, item in zip(retained, items, strict=True)
            )
            / n_positions
        )
        cosine_defined = sum(int(item["correction_cosine_defined"]) for item in items)
        geometry["correction_cosine_defined"] = cosine_defined
        geometry["correction_cosine_mean"] = (
            None
            if cosine_defined == 0
            else sum(
                float(item["correction_cosine_mean"])
                * int(item["correction_cosine_defined"])
                for item in items
                if item["correction_cosine_mean"] is not None
            )
            / cosine_defined
        )
        return geometry

    def cell_receipts(self) -> tuple[dict[str, object], ...]:
        """Return one geometry/noise receipt per unpadded prompt cell."""

        if not self._geometry:
            raise ValueError("generation session has not observed a hook call")
        receipts = []
        for index, cell in enumerate(self.cells):
            geometry = self._aggregate_geometry(self._cell_geometry[index])
            realized = torch.cat(self._cell_realized[index])
            geometry["token_variance_along_v"] = float(realized.var(correction=0))
            digest = hashlib.sha256()
            for position, payload in sorted(self._cell_noise[index].items()):
                digest.update(int(position).to_bytes(8, "little", signed=True))
                digest.update(payload)
            receipts.append(
                {
                    "prompt_id": cell.prompt_id,
                    "hook_calls": self._hook_calls,
                    "flow_network_evaluations": self._flow_evaluations,
                    "positions_evaluated": int(geometry["n_positions"]),
                    "guarded_positions": self._cell_guarded[index],
                    "padding_positions_evaluated": 0,
                    "epsilon_positions": len(self._cell_noise[index]),
                    "epsilon_sha256": digest.hexdigest(),
                    "geometry": geometry,
                }
            )
        return tuple(receipts)

    def receipt(self) -> dict[str, int | dict[str, float | int | None]]:
        if not self._geometry:
            raise ValueError("generation session has not observed a hook call")
        geometry = self._aggregate_geometry(self._geometry)
        n_positions = int(geometry["n_positions"])
        return {
            "hook_calls": self._hook_calls,
            "flow_network_evaluations": self._flow_evaluations,
            "positions_evaluated": n_positions,
            "guarded_positions": self._guarded,
            "geometry": geometry,
        }
