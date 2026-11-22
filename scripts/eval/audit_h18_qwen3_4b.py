"""H18 Quant-Snap Lancet audit on Qwen/Qwen3-4B.

Real shard-streaming audit that compares three subspace-edit variants
(``naive_ablation``, ``standard_lancet_h15``, ``h18_quant_snap_lancet``)
under four storage formats (bf16 / NVFP4-b16 / INT4-b32 / FP8 per-channel),
measuring micro-block crest factor, relative L2 drift and re-leakage of the
ablated target-direction energy.

Pure functions (``quantize_int4_symmetric``, ``load_lancet_basis``,
``audit_layer_variants``) are importable so unit tests can exercise them with
synthetic tensors and no network access.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safetensors.torch import load_file as _safetensors_load_file  # noqa: E402

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"
DEFAULT_BASIS_PATH = "results/tenet_releases/Tenet-4B-Qwen3-Unslop-v2/lancet_basis.safetensors"
DEFAULT_OUTPUT_PATH = "results/h18_nvfp4_quant_snap/qwen3_4b_audit.json"
INDEX_FILENAME = "model.safetensors.index.json"

ProjectionSpec = tuple[str, str]  # (short name, weight key template)


def _resolve_symbol(attr: str, module_candidates: Sequence[str]) -> Any:
    """Resolve ``attr`` from the first candidate module that provides it."""
    last_error: Exception | None = None
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - import fallback path
            last_error = exc
            continue
        symbol = getattr(module, attr, None)
        if symbol is not None:
            return symbol
    raise ImportError(
        f"Could not resolve '{attr}' from any of: {list(module_candidates)}"
    ) from last_error


_SurgeryCandidates = ("sml.surgery.subspace", "sml.surgery")
project_and_snap_subspace_norm_preserving: Callable[..., torch.Tensor] = _resolve_symbol(
    "project_and_snap_subspace_norm_preserving", _SurgeryCandidates
)
_QuantCandidates = ("sml.surgery.quantization", "sml.surgery.subspace", "sml.surgery")
quantize_nvfp4_e2m1: Callable[..., torch.Tensor] = _resolve_symbol(
    "quantize_nvfp4_e2m1", _QuantCandidates
)
quantize_fp8_e4m3: Callable[..., torch.Tensor] = _resolve_symbol(
    "quantize_fp8_e4m3", _QuantCandidates
)
compute_microblock_crest_factor: Callable[..., Any] = _resolve_symbol(
    "compute_microblock_crest_factor", _QuantCandidates
)


# ---------------------------------------------------------------------------
# Pure quantization / metric helpers
# ---------------------------------------------------------------------------
def quantize_int4_symmetric(weight: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Symmetric block-``block_size`` INT4 fake quantization.

    Pads the last dim to a multiple of ``block_size`` with zeros, computes a
    per-block symmetric scale (``absmax / 7.0``, clamped to a min of ``1e-8``),
    rounds to ``[-7, 7]``, dequantizes and unpads back to the original shape.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    original_shape = weight.shape
    last_dim = original_shape[-1]
    pad = (-last_dim) % block_size
    w = torch.nn.functional.pad(weight.float(), (0, pad)) if pad else weight.float()
    blocks = w.reshape(-1, block_size)
    absmax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (absmax / 7.0).clamp_min(1e-8)
    q = torch.round(blocks / scale).clamp(-7.0, 7.0)
    dequantized = (q * scale).reshape(w.shape)
    if pad:
        dequantized = dequantized[..., :last_dim]
    return dequantized.reshape(original_shape).to(weight.dtype)


def load_lancet_basis(basis_path: Path) -> dict[int, torch.Tensor]:
    """Load ``layer_<idx>`` tensors from a Lancet basis safetensors file."""
    tensors = _safetensors_load_file(str(basis_path))
    basis: dict[int, torch.Tensor] = {}
    for key, tensor in tensors.items():
        match = re.fullmatch(r"layer_(\d+)", key)
        if match:
            basis[int(match.group(1))] = tensor.float()
    return dict(sorted(basis.items()))


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().reshape(-1)[0])
    return float(value)


def _crest_stats(w_q: torch.Tensor, block_size: int = 16) -> tuple[float, float]:
    """Interpret ``compute_microblock_crest_factor`` output as (max, mean)."""
    result = compute_microblock_crest_factor(w_q, block_size=block_size)
    if isinstance(result, dict):
        max_val: float | None = None
        mean_val: float | None = None
        collection: Any = None
        for key, value in result.items():
            lower = key.lower()
            if torch.is_tensor(value) or isinstance(value, (list, tuple)):
                collection = value if collection is None else collection
            if max_val is None and "max" in lower:
                max_val = _as_float(value)
            if mean_val is None and "mean" in lower:
                mean_val = _as_float(value)
        if collection is not None:
            if isinstance(collection, (list, tuple)):
                flat = torch.cat([v.detach().float().reshape(-1) for v in collection])
            else:
                flat = collection.detach().float().reshape(-1)
            if max_val is None:
                max_val = float(flat.max())
            if mean_val is None:
                mean_val = float(flat.mean())
        if max_val is None or mean_val is None:
            raise ValueError(f"Unrecognized crest-factor dict keys: {sorted(result)}")
        return max_val, mean_val
    if torch.is_tensor(result):
        flat = result.detach().float().reshape(-1)
        if flat.numel() == 1:
            value = float(flat[0])
            return value, value
        return float(flat.max()), float(flat.mean())
    if isinstance(result, (tuple, list)):
        if len(result) == 2 and not any(torch.is_tensor(v) and v.numel() > 1 for v in result):
            return _as_float(result[0]), _as_float(result[1])
        flat = torch.cat([torch.as_tensor(v).float().reshape(-1) for v in result])
        return float(flat.max()), float(flat.mean())
    value = float(result)
    return value, value


def _row_normalized_basis(direction: torch.Tensor) -> torch.Tensor:
    rows = direction.detach().float()
    if rows.dim() == 1:
        rows = rows.unsqueeze(0)
    norms = rows.norm(dim=-1, keepdim=True)
    return rows / norms.clamp_min(1e-12)


def _project_onto(weight: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return weight @ u.T  # [out, k]


def _preserve_row_norms(edited: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    ref_norms = reference.float().norm(dim=-1, keepdim=True)
    edit_norms = edited.float().norm(dim=-1, keepdim=True)
    return edited.float() * (ref_norms / edit_norms.clamp_min(1e-12))


def _variant_metrics(
    w_q: torch.Tensor,
    base_weight: torch.Tensor,
    u: torch.Tensor,
    base_target_energy: torch.Tensor,
) -> dict[str, float]:
    cf_max, cf_mean = _crest_stats(w_q, block_size=16)
    diff = w_q.float() - base_weight.float()
    rel_l2 = float(diff.norm() / (base_weight.float().norm() + 1e-12))
    releak = float(_project_onto(w_q.float(), u).norm() / (base_target_energy + 1e-12))
    return {
        "block16_crest_factor_max": cf_max,
        "block16_crest_factor_mean": cf_mean,
        "rel_l2_drift": rel_l2,
        "releak_ratio": releak,
    }


def audit_layer_variants(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    scale: float = 1.0,
    num_iters: int = 4,
    step_size: float = 0.75,
    block_size: int = 16,
    max_block_cf: float = 3.2,
    max_rel_drift: float = 0.02,
) -> dict[str, Any]:
    """Evaluate edit variants × storage formats for one ``[out, in]`` weight."""
    base = weight.detach().float()
    u = _row_normalized_basis(direction)
    base_target_energy = float(_project_onto(base, u).norm())

    projection = _project_onto(base, u) @ u  # [out, in]
    naive = base - projection
    h15 = _preserve_row_norms(base - scale * projection, base)
    h18 = project_and_snap_subspace_norm_preserving(
        weight,
        direction,
        scale=scale,
        snap_mode="hybrid_nvfp4_int4",
        num_iters=num_iters,
        step_size=step_size,
        block_size=block_size,
        max_block_cf=max_block_cf,
        preserve_row_norm=True,
        max_rel_drift=max_rel_drift,
    )
    if isinstance(h18, (tuple, list)):
        h18 = next(t for t in h18 if torch.is_tensor(t))
    h18 = torch.as_tensor(h18).detach().float()

    variants: dict[str, torch.Tensor] = {
        "naive_ablation": naive,
        "standard_lancet_h15": h15,
        "h18_quant_snap_lancet": h18,
    }
    quantizers: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "bf16": lambda w: w,
        "nvfp4_b16": lambda w: quantize_nvfp4_e2m1(w, block_size=16),
        "int4_b32": lambda w: quantize_int4_symmetric(w, block_size=32),
        "fp8_per_channel": lambda w: quantize_fp8_e4m3(w, per_channel=True),
    }

    results: dict[str, Any] = {
        "shape": list(base.shape),
        "direction_rows": int(u.shape[0]),
        "variants": {},
    }
    for variant_name, w_edit in variants.items():
        per_quant: dict[str, dict[str, float]] = {}
        for quant_name, quantize in quantizers.items():
            w_q = quantize(w_edit).detach().float()
            per_quant[quant_name] = _variant_metrics(w_q, base, u, base_target_energy)
        results["variants"][variant_name] = per_quant

    h15_results = results["variants"]["standard_lancet_h15"]
    h18_results = results["variants"]["h18_quant_snap_lancet"]
    ratios: dict[str, float] = {}
    for quant_name, ratio_key in (
        ("nvfp4_b16", "nvfp4_releakage_reduction_ratio"),
        ("int4_b32", "int4_b32_releakage_reduction_ratio"),
        ("fp8_per_channel", "fp8_releakage_reduction_ratio"),
    ):
        h15_releak = h15_results[quant_name]["releak_ratio"]
        h18_releak = h18_results[quant_name]["releak_ratio"]
        ratios[ratio_key] = (1.0 - h18_releak / h15_releak) if h15_releak > 1e-12 else 0.0
    results["summary_ratios"] = ratios
    return results


# ---------------------------------------------------------------------------
# Shard streaming / aggregation
# ---------------------------------------------------------------------------
def _load_weight_map(
    model_id: str, shard_loader: Callable[[str], dict[str, Any]] | None
) -> dict[str, str]:
    if shard_loader is not None:
        index = shard_loader(INDEX_FILENAME)
        return index["weight_map"] if isinstance(index, dict) and "weight_map" in index else index
    from huggingface_hub import hf_hub_download

    index_path = hf_hub_download(repo_id=model_id, filename=INDEX_FILENAME)
    with open(index_path, encoding="utf-8") as handle:
        return json.load(handle)["weight_map"]


def _load_shard_tensors(
    filename: str,
    model_id: str,
    shard_loader: Callable[[str], dict[str, Any]] | None,
    cleanup_cache: bool,
) -> dict[str, torch.Tensor]:
    if shard_loader is not None:
        return shard_loader(filename)
    from huggingface_hub import hf_hub_download

    shard_path = hf_hub_download(repo_id=model_id, filename=filename)
    tensors = _safetensors_load_file(shard_path)
    if cleanup_cache:
        try:
            from sml.hub.release import cleanup_hf_model_cache

            cleanup_hf_model_cache(model_id)
        except ImportError as exc:
            print(f"[h18-audit] warning: cache cleanup unavailable ({exc})")
    return tensors


def run_h18_audit(
    basis_path: Path,
    model_id: str = DEFAULT_MODEL_ID,
    output_path: Path | str | None = None,
    layers: Sequence[int] | None = None,
    shard_loader: Callable[[str], dict[str, Any]] | None = None,
    cleanup_cache: bool = True,
    scale: float = 1.0,
    num_iters: int = 4,
    step_size: float = 0.75,
    block_size: int = 16,
    max_block_cf: float = 3.2,
    max_rel_drift: float = 0.02,
    projections: Sequence[ProjectionSpec] = (
        ("mlp.down_proj", "model.layers.{idx}.mlp.down_proj.weight"),
        ("self_attn.o_proj", "model.layers.{idx}.self_attn.o_proj.weight"),
    ),
) -> dict[str, Any]:
    """Stream required shards, audit every basis layer × projection, write JSON.

    ``shard_loader(filename) -> dict`` overrides all HF Hub traffic (index via
    ``"model.safetensors.index.json"``, shards via shard filenames); no network
    access or cache cleanup happens when it is provided.
    """
    basis = load_lancet_basis(basis_path)
    selected_layers = sorted(basis) if layers is None else sorted(set(layers))
    missing = [idx for idx in selected_layers if idx not in basis]
    if missing:
        raise KeyError(f"Layers missing from basis file {basis_path}: {missing}")

    projection_specs: list[ProjectionSpec] = list(projections)
    weight_map = _load_weight_map(model_id, shard_loader)
    wanted_keys = {
        key_template.format(idx=idx): (idx, short)
        for idx in selected_layers
        for short, key_template in projection_specs
    }
    required_shards: dict[str, list[str]] = {}
    for key in wanted_keys:
        filename = weight_map.get(key)
        if filename is None:
            print(f"[h18-audit] warning: {key} not in index; skipping")
            continue
        required_shards.setdefault(filename, []).append(key)

    per_layer: dict[str, Any] = {}
    for filename, keys in sorted(required_shards.items()):
        shard_tensors = _load_shard_tensors(filename, model_id, shard_loader, cleanup_cache)
        for key in keys:
            if key not in shard_tensors:
                print(f"[h18-audit] warning: {key} not in shard {filename}; skipping")
                continue
            idx, short = wanted_keys[key]
            result = audit_layer_variants(
                shard_tensors[key].float(),
                basis[idx],
                scale=scale,
                num_iters=num_iters,
                step_size=step_size,
                block_size=block_size,
                max_block_cf=max_block_cf,
                max_rel_drift=max_rel_drift,
            )
            per_layer.setdefault(str(idx), {})[short] = result
        del shard_tensors

    totals: dict[str, list[float]] = {}
    ratio_totals: dict[str, list[float]] = {}
    for layer_results in per_layer.values():
        for proj_results in layer_results.values():
            for variant_name, quants in proj_results["variants"].items():
                for quant_name, metrics in quants.items():
                    for metric_name, value in metrics.items():
                        key = f"{variant_name}.{quant_name}.{metric_name}"
                        totals.setdefault(key, []).append(float(value))
            for ratio_name, ratio_val in proj_results.get("summary_ratios", {}).items():
                ratio_totals.setdefault(ratio_name, []).append(float(ratio_val))
    means = {name: sum(values) / len(values) for name, values in sorted(totals.items())}
    mean_summary_ratios = {
        name: sum(values) / len(values) for name, values in sorted(ratio_totals.items())
    }

    audit: dict[str, Any] = {
        "model_id": model_id,
        "basis_path": str(basis_path),
        "layers": selected_layers,
        "num_layers_audited": len(per_layer),
        "projections": [short for short, _ in projection_specs],
        "config": {
            "scale": scale,
            "num_iters": num_iters,
            "step_size": step_size,
            "block_size": block_size,
            "max_block_cf": max_block_cf,
            "max_rel_drift": max_rel_drift,
            "cleanup_cache": cleanup_cache,
            "shard_loader_injected": shard_loader is not None,
        },
        "per_layer": per_layer,
        "mean_metrics": means,
        "summary_ratios": mean_summary_ratios,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        audit["output_path"] = str(out)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2)
    return audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_layers(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H18 Quant-Snap Lancet audit (Qwen/Qwen3-4B)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--basis-path", type=Path, default=Path(DEFAULT_BASIS_PATH))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (default: all in basis)",
    )
    parser.add_argument("--cleanup-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    layers = _parse_layers(args.layers)
    print("[h18-audit] resolved config:")
    print(f"  model_id      = {args.model_id}")
    print(f"  basis_path    = {args.basis_path}")
    print(f"  output        = {args.output}")
    print(f"  layers        = {layers if layers is not None else 'all basis layers'}")
    print(f"  cleanup_cache = {args.cleanup_cache}")
    print(f"  dry_run       = {args.dry_run}")

    if args.dry_run:
        if args.basis_path.exists():
            basis_layers = sorted(load_lancet_basis(args.basis_path))
            print(f"  basis layers  = {basis_layers}")
        else:
            print(f"  basis layers  = (basis file not found: {args.basis_path})")
        return 0

    run_h18_audit(
        basis_path=args.basis_path,
        model_id=args.model_id,
        output_path=args.output,
        layers=layers,
        cleanup_cache=args.cleanup_cache,
    )
    print(f"[h18-audit] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
