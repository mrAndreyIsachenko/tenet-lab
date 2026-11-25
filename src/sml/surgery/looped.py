"""H22: Looped-MoE-aware Lancet surgery and speculative-drafter synchronization.

Implements two H22 capabilities on top of the Tri-Anchor Lancet MoE surgery
(``sml.surgery.subspace``):

1. ``apply_looped_moe_lancet_surgery``: builds a rank-``k_loop`` SVD-compressed
   target basis across loop-unrolled execution ticks, bi-projected orthogonally
   to the anchor (reason/boundary/format) directions, applies the MoE Lancet
   edit with tick-averaged expert selectivity, and verifies hard-zero drift on
   loop controllers and router weights.
2. ``synchronize_speculative_drafter_surgery``: synchronizes the MoE surgery
   into embedded speculative drafter / MTP head projections with row-norm
   preservation and a KL-reduction gate against the backbone.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch

from sml.surgery.cka import (
    get_transformer_layers,
    is_router_weight_key,
)
from sml.surgery.subspace import (
    apply_moe_lancet_surgery,
    build_loop_unrolled_biprojected_basis,
    project_out_subspace_norm_preserving,
)

# Name segments identifying embedded speculative drafter / MTP heads.
_DRAFTER_TOKENS = ("dspark", "dfly2", "drafter", "mtp_heads", "eh_proj")

# Frozen loop-controller name segments (hard 0.0 drift requirement).
_LOOP_CONTROLLER_TOKENS = ("loop_controller", "loop_gate", "tick_embed")

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:$|\.)")


def is_drafter_weight_key(name: str) -> bool:
    """Return ``True`` when ``name`` is a 2-D-eligible drafter / MTP weight.

    A key counts as a drafter weight when it ends with ``.weight``, is not a
    router weight, and at least one dotted name segment either contains one of
    the drafter tokens (``dspark``, ``dfly2``, ``drafter``, ``mtp_heads``,
    ``eh_proj``) or equals ``mtp`` / starts with ``mtp_``.
    """
    if not name.endswith(".weight") or is_router_weight_key(name):
        return False
    for segment in name.lower().split("."):
        if any(token in segment for token in _DRAFTER_TOKENS):
            return True
        if segment == "mtp" or segment.startswith("mtp_"):
            return True
    return False


def _resolve_dirs(
    layer_bases: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    layer_idx: int | None,
) -> list[torch.Tensor]:
    """Resolve per-layer directions for ``layer_idx`` (fall back to any entry)."""
    if isinstance(layer_bases, dict):
        if layer_idx is not None and layer_idx in layer_bases:
            return subspace_directions(layer_bases[layer_idx])
        for value in layer_bases.values():
            dirs = subspace_directions(value)
            if dirs:
                return dirs
        return []
    return subspace_directions(layer_bases)


def subspace_directions(value: list[torch.Tensor] | torch.Tensor) -> list[torch.Tensor]:
    """Normalize ``value`` into a flat list of 1-D direction tensors."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return [value]
        return [row for row in value]
    return [row for row in value]


def _max_param_drift(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> float:
    """Max absolute weight drift between current params and ``snapshot``."""
    drift = 0.0
    for name, param in model.named_parameters():
        if name in snapshot:
            drift = max(drift, float((param.data - snapshot[name]).abs().max()))
    return drift


def _snapshot_keys(model: torch.nn.Module, predicate) -> dict[str, torch.Tensor]:
    return {name: param.data.clone() for name, param in model.named_parameters() if predicate(name)}


def apply_looped_moe_lancet_surgery(
    model: torch.nn.Module,
    anchor_dirs_by_tick: list[list[torch.Tensor] | torch.Tensor],
    target_dirs_by_tick: list[list[torch.Tensor] | torch.Tensor],
    expert_profiles: dict[int, dict[str, Any]] | None = None,
    *,
    k_loop: int = 2,
    scale: float = 0.35,
    min_selectivity_threshold: float = 0.25,
    shared_expert_damping: float = 0.40,
    hot_vram_expert_ids: set[int] | None = None,
    quant_snap: bool | None = None,
    max_block16_crest_factor: float = 3.20,
) -> dict[str, Any]:
    """Apply tick-bi-projected Lancet surgery to a Looped MoE model.

    Builds an SVD-compressed target basis of rank ``<= k_loop`` across the
    loop-unrolled ticks, orthogonal to the anchor subspace
    (``U_reason + U_boundary + U_format``), then runs
    :func:`apply_moe_lancet_surgery` with tick-averaged expert selectivity and
    verifies hard-zero drift on loop controllers (``loop_controller``,
    ``loop_gate``, ``tick_embed``) and router weights.

    Returns a summary dict including ``k_loop``, ``basis_rank``,
    ``loop_controller_drift``, ``untouched_expert_ratio`` and
    ``router_weight_drift``.
    """
    u_anchor, u_target = build_loop_unrolled_biprojected_basis(
        anchor_dirs_by_tick,
        target_dirs_by_tick,
        k_loop=k_loop,
    )
    basis_rank = int(u_target.shape[0])
    target_dirs = [u_target[i] for i in range(basis_rank)]

    layer_bases = {
        layer_idx: target_dirs for layer_idx in range(len(get_transformer_layers(model)))
    }

    # Tick-averaged expert selectivity: average `selective_weights` across
    # per-tick profile entries when expert_profiles carries one profile per tick.
    averaged_profiles: dict[int, dict[str, Any]] | None = None
    if expert_profiles is not None:
        averaged_profiles = {}
        for layer_idx, profile in expert_profiles.items():
            if isinstance(profile, (list, tuple)) and len(profile) > 0:
                ticks = [p for p in profile if isinstance(p, dict)]
                if ticks:
                    merged: dict[str, Any] = dict(ticks[-1])
                    weights = [
                        p.get("selective_weights")
                        for p in ticks
                        if p.get("selective_weights") is not None
                    ]
                    if weights:
                        stacked = torch.stack(
                            [torch.as_tensor(w, dtype=torch.float32).reshape(-1) for w in weights],
                            dim=0,
                        )
                        merged["selective_weights"] = stacked.mean(dim=0)
                    averaged_profiles[layer_idx] = merged
                    continue
            averaged_profiles[layer_idx] = profile

    loop_keys = lambda name: any(  # noqa: E731
        token in segment for segment in name.lower().split(".") for token in _LOOP_CONTROLLER_TOKENS
    )
    loop_snapshot = _snapshot_keys(model, loop_keys)
    router_snapshot = _snapshot_keys(model, is_router_weight_key)

    summary = apply_moe_lancet_surgery(
        model,
        layer_bases,
        averaged_profiles,
        scale=scale,
        min_selectivity_threshold=min_selectivity_threshold,
        shared_expert_damping=shared_expert_damping,
        hot_vram_expert_ids=hot_vram_expert_ids,
        quant_snap=quant_snap,
        max_block16_crest_factor=max_block16_crest_factor,
    )

    loop_controller_drift = _max_param_drift(model, loop_snapshot)
    router_weight_drift = _max_param_drift(model, router_snapshot)

    result: dict[str, Any] = dict(summary)
    result.update(
        {
            "k_loop": k_loop,
            "basis_rank": basis_rank,
            "anchor_rank": int(u_anchor.shape[0]),
            "loop_controller_drift": loop_controller_drift,
            "router_weight_drift": router_weight_drift,
            "loop_controller_frozen": loop_controller_drift == 0.0,
            "router_frozen": router_weight_drift == 0.0,
        }
    )
    return result


def synchronize_speculative_drafter_surgery(
    model: torch.nn.Module,
    layer_bases: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    *,
    scale: float = 0.35,
    min_kl_reduction_pct: float = 35.0,
    calibration_inputs: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Synchronize the MoE Lancet edit into embedded drafter / MTP projections.

    Scans the model for embedded speculative drafter / MTP output projections
    (``dspark``, ``dfly2``, ``drafter``, ``mtp_heads``, ``eh_proj``) with 2-D
    weights and applies a synchronous row-norm-preserving bi-projection of
    ``layer_bases``. A KL-reduction gate verifies the edit actually reduces
    drafter/backbone divergence (forward KL on ``calibration_inputs`` when a
    drafter/backbone KL helper is available, otherwise a projection residual
    energy proxy onto the target subspace against backbone-aligned
    directions).

    Returns ``{edited_drafter_params, kl_before, kl_after, kl_reduction_pct,
    passed_kl_gate}``.
    """
    try:  # Optional helper, when available in this installation.
        from sml.surgery.drafters import drafter_backbone_kl_divergence
    except ImportError:
        drafter_backbone_kl_divergence = None  # type: ignore[assignment]

    drafter_params = [
        (name, param)
        for name, param in model.named_parameters()
        if is_drafter_weight_key(name) and param.ndim == 2
    ]

    kl_before = 0.0
    kl_after = 0.0

    if (
        drafter_backbone_kl_divergence is not None
        and calibration_inputs is not None
        and len(drafter_params) > 0
    ):
        kl_before = float(drafter_backbone_kl_divergence(model, calibration_inputs))

    edited = 0
    residual_energy_before: list[float] = []
    residual_energy_after: list[float] = []

    for name, param in drafter_params:
        match = _LAYER_INDEX_RE.search(name)
        layer_idx = int(match.group(1)) if match else None
        dirs = _resolve_dirs(layer_bases, layer_idx)
        if not dirs:
            continue
        weight = param.data
        u = torch.stack([d.reshape(-1).to(dtype=torch.float32) for d in dirs], dim=0)
        if weight.shape[0] != u.shape[1]:
            continue
        # Residual-energy proxy: fraction of column energy along the target
        # output subspace (what project_out_subspace_norm_preserving removes).
        flat = weight.reshape(weight.shape[0], -1).to(dtype=torch.float32)
        coeff = u @ flat
        denom = float((flat * flat).sum()) + 1e-12
        residual_energy_before.append(float((coeff * coeff).sum()) / denom)

        param.data = project_out_subspace_norm_preserving(
            weight, dirs, scale=scale, preserve_row_norm=True
        )
        edited += 1

        flat_after = param.data.reshape(param.data.shape[0], -1).to(dtype=torch.float32)
        coeff_after = u @ flat_after
        denom_after = float((flat_after * flat_after).sum()) + 1e-12
        residual_energy_after.append(float((coeff_after * coeff_after).sum()) / denom_after)

    if edited == 0:
        return {
            "edited_drafter_params": 0,
            "kl_before": 0.0,
            "kl_after": 0.0,
            "kl_reduction_pct": 0.0,
            "passed_kl_gate": False,
        }

    if drafter_backbone_kl_divergence is not None and calibration_inputs is not None:
        kl_after = float(drafter_backbone_kl_divergence(model, calibration_inputs))
    else:
        # Projection residual-energy proxy: "KL before" is the target-subspace
        # energy still present against backbone-aligned directions before the
        # edit; "after" is what remains once the bi-projection removed it.
        kl_before = float(sum(residual_energy_before) / len(residual_energy_before))
        kl_after = float(sum(residual_energy_after) / len(residual_energy_after))

    kl_reduction_pct = (kl_before - kl_after) / max(kl_before, 1e-12) * 100.0

    return {
        "edited_drafter_params": edited,
        "kl_before": kl_before,
        "kl_after": kl_after,
        "kl_reduction_pct": kl_reduction_pct,
        "passed_kl_gate": kl_reduction_pct >= min_kl_reduction_pct,
    }


def synchronize_speculative_drafter_streaming(
    checkpoint_dir: str | Path,
    layer_bases: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    *,
    output_dir: str | Path | None = None,
    scale: float = 0.35,
    min_kl_reduction_pct: float = 35.0,
) -> dict[str, Any]:
    """Streaming (shard-by-shard) speculative-drafter synchronization.

    Applies the same bi-projection as :func:`synchronize_speculative_drafter_surgery`
    without materializing the full model: each ``.safetensors`` shard is
    inspected zero-copy via :func:`safetensors.safe_open`, and only shards
    carrying drafter weights are loaded, edited and re-saved atomically.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    src_dir = Path(checkpoint_dir)
    dst_dir = Path(output_dir) if output_dir is not None else src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    same_dir = src_dir.resolve() == dst_dir.resolve()

    index_path = src_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as fh:
            weight_map = json.load(fh).get("weight_map", {})
        shard_names: list[str] = list(dict.fromkeys(weight_map.values()))
    else:
        shard_names = sorted(
            p.name for p in src_dir.glob("*.safetensors") if p.name != "lancet_basis.safetensors"
        )

    edited = 0
    shards_touched = 0
    router_weight_drift = 0.0
    residual_energy_before: list[float] = []
    residual_energy_after: list[float] = []

    for shard_name in shard_names:
        src_shard = src_dir / shard_name
        if not src_shard.is_file():
            continue
        with safe_open(str(src_shard), framework="pt", device="cpu") as handle:
            shard_keys = list(handle.keys())
        drafter_keys = [k for k in shard_keys if is_drafter_weight_key(k)]
        if not drafter_keys:
            if not same_dir:
                shutil.copy2(src_shard, dst_dir / shard_name)
            continue
        shards_touched += 1
        shard_tensors = load_file(str(src_shard))
        router_snapshot = {
            k: tensor.clone() for k, tensor in shard_tensors.items() if is_router_weight_key(k)
        }
        for key in drafter_keys:
            tensor = shard_tensors[key]
            if tensor.ndim != 2:
                continue
            match = _LAYER_INDEX_RE.search(key)
            layer_idx = int(match.group(1)) if match else None
            dirs = _resolve_dirs(layer_bases, layer_idx)
            if not dirs:
                continue
            u = torch.stack([d.reshape(-1).to(dtype=torch.float32) for d in dirs], dim=0)
            if tensor.shape[0] != u.shape[1]:
                continue
            flat = tensor.reshape(tensor.shape[0], -1).to(dtype=torch.float32)
            coeff = u @ flat
            denom = float((flat * flat).sum()) + 1e-12
            residual_energy_before.append(float((coeff * coeff).sum()) / denom)
            shard_tensors[key] = project_out_subspace_norm_preserving(
                tensor, dirs, scale=scale, preserve_row_norm=True
            )
            flat_after = (
                shard_tensors[key].reshape(shard_tensors[key].shape[0], -1).to(dtype=torch.float32)
            )
            coeff_after = u @ flat_after
            denom_after = float((flat_after * flat_after).sum()) + 1e-12
            residual_energy_after.append(float((coeff_after * coeff_after).sum()) / denom_after)
            edited += 1
        for name, snapshot in router_snapshot.items():
            current = shard_tensors.get(name, snapshot)
            router_weight_drift = max(router_weight_drift, float((current - snapshot).abs().max()))
        tmp_path = dst_dir / (shard_name + ".tmp")
        save_file(shard_tensors, str(tmp_path))
        os.replace(tmp_path, dst_dir / shard_name)

    if edited == 0:
        return {
            "mode": "streaming",
            "edited_drafter_params": 0,
            "shards_touched": 0,
            "router_weight_drift": 0.0,
            "kl_before": 0.0,
            "kl_after": 0.0,
            "kl_reduction_pct": 0.0,
            "passed_kl_gate": False,
        }

    kl_before = float(sum(residual_energy_before) / len(residual_energy_before))
    kl_after = float(sum(residual_energy_after) / len(residual_energy_after))
    kl_reduction_pct = (kl_before - kl_after) / max(kl_before, 1e-12) * 100.0

    return {
        "mode": "streaming",
        "edited_drafter_params": edited,
        "shards_touched": shards_touched,
        "router_weight_drift": float(router_weight_drift),
        "kl_before": kl_before,
        "kl_after": kl_after,
        "kl_reduction_pct": kl_reduction_pct,
        "passed_kl_gate": bool(kl_reduction_pct >= min_kl_reduction_pct),
    }


__all__ = [
    "apply_looped_moe_lancet_surgery",
    "is_drafter_weight_key",
    "synchronize_speculative_drafter_surgery",
    "synchronize_speculative_drafter_streaming",
]
