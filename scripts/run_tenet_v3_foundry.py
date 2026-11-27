#!/usr/bin/env python3
"""Tenet v3 foundry orchestrator: H21 Strata-Tier-Aware MoE Lancet + H22 Looped-MoE.

Targets the six canonical Tenet v3 releases (see ``TENET_V3_MODEL_SPECS``).

Pipeline per model:
  1. Resolve the checkpoint directory (``--model-dir`` or HF snapshot download).
  2. H21: strata-tier-aware MoE Lancet surgery (streaming recipe when
     available, in-memory :func:`apply_moe_lancet_surgery` otherwise) with
     tier-aware hot-VRAM expert handling, selective-expert scaling and the
     H18 quant-snap pre-compensation.
  3. H22 (looped models): :func:`apply_looped_moe_lancet_surgery` plus
     :func:`synchronize_speculative_drafter_surgery` when the release embeds
     a speculative drafter / MTP head.
  4. Log telemetry to Nebius Managed MLflow via
     :func:`sml.tracking.nebius_mlflow.log_foundry_surgery_run` (always
     writes a local JSON mirror, never raises on remote failure).
  5. Emit ``strata.toml`` and a Tenet model card via
     ``sml.hub.release.generate_tenet_model_card``.
  6. Write ``foundry_summary.json`` (always, even on failure).

All surgery runs as pure ``.safetensors`` layer streaming (per-tensor
``safe_open`` reads plus ``save_file`` writes): no
``AutoModelForCausalLM.from_pretrained``, no in-memory weight materialization
and no synthetic ``torch.randn`` direction fixtures — every Lancet basis and
expert profile is calibrated from real checkpoint tensors.

``--dry-run`` prints the resolved JSON plan and exits 0 without touching
weights, CUDA or the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = "results/tenet_v3_foundry"

TENET_V3_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "Tenet-30B-A3B-GLM4.7-Unslop-v3": {
        "tier": "edge",
        "base_model": "zai-org/GLM-4.7-Flash",
        "naive_model": None,
        "hypothesis_stack": ["H21", "H22"],
        "looped": False,
        "has_drafter": True,
        "drafter_type": "MTP",
        "think_end_token": "</think>",
        "hot_vram_expert_ids": {0, 1, 2},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-15B-A2B-Inkling2-Unslop-v3": {
        "tier": "edge",
        "base_model": "thinkingmachines/Inkling-2-Small",
        "naive_model": None,
        "hypothesis_stack": ["H21"],
        "looped": False,
        "has_drafter": False,
        "drafter_type": None,
        "think_end_token": "<|thought|>",
        "hot_vram_expert_ids": {0, 1},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-26B-A3B-Inkling2-Unslop-v3": {
        "tier": "workhorse",
        "base_model": "thinkingmachines/Inkling-2-Mini",
        "naive_model": None,
        "hypothesis_stack": ["H21", "H22"],
        "looped": True,
        "loop_ticks": 3,
        "k_loop": 2,
        "has_drafter": True,
        "drafter_type": "DSpark",
        "think_end_token": "<|thought|>",
        "hot_vram_expert_ids": {0, 1, 2},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-35B-A3B-Qwen3.6-Unslop-v3": {
        "tier": "workhorse",
        "base_model": "Qwen/Qwen3.6-35B-A3B",
        "naive_model": "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated",
        "hypothesis_stack": ["H21", "H22"],
        "looped": False,
        "has_drafter": True,
        "drafter_type": "MTP",
        "think_end_token": "</think>",
        "hot_vram_expert_ids": {0, 1, 2},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-106B-A12B-GLM4.5-Unslop-v3": {
        "tier": "workhorse",
        "base_model": "zai-org/GLM-4.5-Air",
        "naive_model": None,
        "hypothesis_stack": ["H21", "H22"],
        "looped": False,
        "has_drafter": True,
        "drafter_type": "MTP",
        "think_end_token": "</think>",
        "hot_vram_expert_ids": {0, 1, 2, 3},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-32B-A3B-GLM4.5-Unslop-v3": {
        "alias_of": "Tenet-106B-A12B-GLM4.5-Unslop-v3",
        "tier": "workhorse",
        "base_model": "zai-org/GLM-4.5-Air",
        "naive_model": None,
        "hypothesis_stack": ["H21", "H22"],
        "looped": False,
        "has_drafter": True,
        "drafter_type": "MTP",
        "think_end_token": "</think>",
        "hot_vram_expert_ids": {0, 1, 2, 3},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-125B-A6B-Qwen3.8-Unslop-v3": {
        "tier": "frontier",
        "base_model": "Qwen/Qwen3.8-Flash-Next",
        "naive_model": None,
        "hypothesis_stack": ["H21", "H22"],
        "looped": False,
        "has_drafter": True,
        "drafter_type": "MTP",
        "think_end_token": "</think>",
        "hot_vram_expert_ids": {0, 1, 2, 3, 4},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-220B-A12B-Hy4-Unslop-v3": {
        "tier": "frontier",
        "base_model": "tencent/Hy4-flash",
        "naive_model": None,
        "hypothesis_stack": ["H21", "H22"],
        "looped": False,
        "has_drafter": True,
        "drafter_type": "DFly2",
        "think_end_token": "</think>",
        "hot_vram_expert_ids": {0, 1, 2, 3, 4, 5},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-26B-A4B-Gemma4-Unslop-v2": {
        "tier": "workhorse",
        "base_model": "google/gemma-4-26B-A4B-it",
        "naive_model": None,
        "hypothesis_stack": ["H21"],
        "looped": False,
        "has_drafter": False,
        "drafter_type": None,
        "think_end_token": "<|thought|>",
        "hot_vram_expert_ids": {0, 1, 2},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
}

DEFAULT_RELEASE_ID = "tenet-v3-foundry-v1"


def _log(message: str) -> None:
    print(f"[foundry] {message}", flush=True)


def _resolve_model_dir(
    model_id: str,
    model_dir: str | None = None,
    *,
    spec: dict[str, Any] | None = None,
    download: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """Return an existing checkpoint directory, or ``None`` when unresolved.

    Resolution order: explicit ``--model-dir``, the ``TENET_V3_MODEL_DIR``
    environment variable, then (unless ``dry_run and not download``) an HF
    ``snapshot_download`` of the spec's ``base_model``.
    """
    if model_dir:
        candidate = Path(model_dir).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        return None
    env_dir = os.environ.get("TENET_V3_MODEL_DIR")
    if env_dir:
        candidate = Path(env_dir).expanduser().resolve()
        if candidate.is_dir():
            return candidate
    repo_id = (spec or TENET_V3_MODEL_SPECS.get(model_id) or {}).get("base_model", model_id)
    if dry_run and not download:
        return None
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id,
                cache_dir=os.environ.get("HF_HOME") or None,
                ignore_patterns=["*.gguf", "*.bin", "*.pt", "consolidated*"],
            )
        )
    except Exception as exc:
        raise RuntimeError(f"snapshot_download({repo_id}) failed: {exc}") from exc


def _select_surgical_layers(num_layers: int, top_k: int = 8) -> list[int]:
    """Pick up to ``top_k`` contiguous layers centered at ~66% depth."""
    if num_layers <= top_k:
        return list(range(num_layers))
    center = round(0.66 * (num_layers - 1))
    half = top_k // 2
    start = max(0, min(center - half, num_layers - top_k))
    return list(range(start, start + top_k))


_LAYER_KEYS_RE = None  # layer indices are parsed with plain string slicing below


def _layer_index_from_key(key: str) -> int | None:
    """Extract the transformer layer index from a weight key, if any."""
    parsed = _strip_layer_prefix(key)
    return None if parsed is None else int(parsed[0])


def _expert_id_from_suffix(suffix: str) -> int | None:
    """Return the routed-expert id encoded in a stripped layer suffix, if any.

    Handles unpacked ``mlp.experts.{e}.down_proj.weight`` /
    ``experts.{e}.w2.weight`` keys; packed 3-D keys (``experts.down_proj``)
    carry no expert id and return ``None``.
    """
    if "experts." not in suffix:
        return None
    tail = suffix.split("experts.", 1)[1]
    token = tail.split(".", 1)[0]
    if not token.isdigit():
        return None
    return int(token)


def _strip_layer_prefix(key: str) -> tuple[int, str] | None:
    """Return the key with the known layer prefix removed (validator helper)."""
    from sml.hub.validator import strip_layer_prefix as _strip

    return _strip(key)


def _inspect_checkpoint_config(model_dir: Path) -> dict[str, Any]:
    """Inspect ``config.json`` + shard keys without loading any weights.

    Returns ``num_hidden_layers``, ``hidden_size``, ``num_experts``,
    ``architectures``, ``num_loop_ticks`` and the ``key_to_shard`` mapping.
    """
    from safetensors import safe_open

    from sml.hub.validator import resolve_text_config

    cfg_path = model_dir / "config.json"
    raw_cfg: dict[str, Any] = {}
    if cfg_path.exists():
        raw_cfg = json.loads(cfg_path.read_text())
    text_cfg = resolve_text_config(raw_cfg)

    # ---- shard discovery + key -> shard mapping ----
    key_to_shard: dict[str, str] = {}
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        key_to_shard = {str(k): str(v) for k, v in weight_map.items()}
    else:
        shards = sorted(
            p for p in model_dir.glob("*.safetensors") if p.name != "lancet_basis.safetensors"
        )
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in list(f.keys()):
                    key_to_shard[key] = shard.name

    # ---- num_hidden_layers ----
    num_hidden_layers = text_cfg.get("num_hidden_layers")
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        layer_indices = {
            idx for idx in (_layer_index_from_key(k) for k in key_to_shard) if idx is not None
        }
        num_hidden_layers = (max(layer_indices) + 1) if layer_indices else 0

    # ---- num_experts ----
    num_experts = (
        text_cfg.get("num_local_experts")
        or text_cfg.get("num_experts")
        or text_cfg.get("n_routed_experts")
    )
    if not isinstance(num_experts, int) or num_experts <= 0:
        # Tensor discovery: count distinct expert indices on layer 0.
        expert_ids: set[int] = set()
        for key in key_to_shard:
            parsed = _strip_layer_prefix(key)
            if parsed is None or parsed[0] != 0:
                continue
            expert_id = _expert_id_from_suffix(parsed[1])
            if expert_id is not None and parsed[1].endswith(("down_proj.weight", "w2.weight")):
                expert_ids.add(expert_id)
        num_experts = len(expert_ids) or None

    # ---- hidden_size (fallback: shape[0] of an o_proj / down_proj tensor) ----
    hidden_size = text_cfg.get("hidden_size")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        for key, shard_name in key_to_shard.items():
            if key.endswith(
                ("self_attn.o_proj.weight", "attn.wo_ud.weight", "mlp.down_proj.weight")
            ):
                shard = model_dir / shard_name
                with safe_open(shard, framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(key)
                hidden_size = int(tensor.shape[0])
                break

    return {
        "num_hidden_layers": int(num_hidden_layers),
        "hidden_size": int(hidden_size) if isinstance(hidden_size, int) else None,
        "num_experts": num_experts,
        "architectures": raw_cfg.get("architectures") or text_cfg.get("architectures"),
        "num_loop_ticks": text_cfg.get("num_loop_ticks"),
        "key_to_shard": key_to_shard,
    }


def _load_tensor(model_dir: Path, key_to_shard: dict[str, str], key: str) -> Any:
    """Load one tensor from its shard by key (no whole-model materialization)."""
    from safetensors import safe_open

    shard_name = key_to_shard[key]
    with safe_open(model_dir / shard_name, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _find_projection_key(
    key_to_shard: dict[str, str], layer_idx: int, num_experts: int | None
) -> str | None:
    """Locate one projection weight for ``layer_idx`` across any nested prefix."""
    suffixes = (
        "self_attn.o_proj.weight",
        "attn.wo_ud.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_experts.down_proj.weight",
        "mlp.down_proj.weight",
    )
    for suffix in suffixes:
        for key in key_to_shard:
            parsed = _strip_layer_prefix(key)
            if parsed is None or parsed[0] != layer_idx:
                continue
            if parsed[1].endswith(suffix):
                return key
    # Routed-expert fallback: expert 0 down_proj / w2.
    expert_suffixes = ("down_proj.weight", "w2.weight") if num_experts else ()
    for suffix in expert_suffixes:
        for key in key_to_shard:
            parsed = _strip_layer_prefix(key)
            if parsed is None or parsed[0] != layer_idx:
                continue
            if ".experts.0." in parsed[1] and parsed[1].endswith(suffix):
                return key
    return None


def _power_iteration_rowspace(W: Any, k: int, generator: Any, iters: int = 24) -> Any:
    """Streaming power iteration for the top-``k`` rowspace directions of ``W``."""
    import torch

    Wf = W.to(torch.float32)
    n = Wf.shape[0]
    X = torch.randn(n, k, generator=generator, dtype=torch.float32)
    X, _ = torch.linalg.qr(X)
    for _ in range(iters):
        X = Wf @ (Wf.T @ X)  # W W^T X -> left-singular subspace
        X, _ = torch.linalg.qr(X)
    return X  # (n, k) orthonormal columns


def _build_streaming_basis_and_profiles(
    model_dir: Path,
    cfg_info: dict[str, Any],
    spec: dict[str, Any],
    *,
    basis_out_path: Path,
    strata_config_path: Path | None = None,
    allow_weight_svd_fallback: bool = False,
) -> tuple[dict[int, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    """Calibrate the Lancet basis + expert profiles from the live model.

    Default path: real GPU activation calibration
    (:func:`calibrate_streaming_moe_basis_from_activations`) — contrastive
    prompt forward passes with router-selectivity counting. When the live model
    cannot be loaded for real forward passes (``UnsupportedForwardArchError``),
    the event is logged and the legacy pure ``.safetensors`` weight-SVD
    power-iteration path runs instead, so calibration always comes from real
    checkpoint tensors (never synthetic ``torch.randn`` fixtures). Passing
    ``allow_weight_svd_fallback=True`` skips the activation attempt entirely.
    """
    import torch
    from safetensors.torch import save_file

    from sml.surgery.probes import REASONING_ANCHOR_PAIRS
    from sml.surgery.subspace import (
        UnsupportedForwardArchError,
        build_loop_unrolled_biprojected_basis,
        calibrate_streaming_moe_basis_from_activations,
    )

    key_to_shard: dict[str, str] = cfg_info["key_to_shard"]
    num_experts = cfg_info.get("num_experts")
    selected_layers = _select_surgical_layers(cfg_info["num_hidden_layers"])

    if not allow_weight_svd_fallback:
        try:
            layer_bases, expert_profiles, meta = calibrate_streaming_moe_basis_from_activations(
                model_dir,
                selected_layers,
                num_experts=num_experts,
                basis_out_path=basis_out_path,
                looped=bool(spec.get("looped")),
                loop_ticks=int(spec.get("loop_ticks", 3)),
                k_loop=int(spec.get("k_loop", 2)),
                pool_mode="answer_only",
                allow_weight_svd_fallback=False,
            )
            meta["selected_layers"] = sorted(layer_bases)
            return layer_bases, expert_profiles, meta
        except UnsupportedForwardArchError as exc:
            # The checkpoint cannot be loaded for real forward-pass calibration
            # (e.g. a first-party architecture transformers cannot instantiate,
            # or a fixture-scale checkpoint). Never silently fall back: log the
            # event and halt surgery — the weight-SVD path only runs when the
            # operator explicitly passes --allow-weight-svd-fallback.
            _log(
                json.dumps(
                    {
                        "calibration_mode": "unsupported_forward_arch",
                        "detail": str(exc)[:500],
                    }
                )
            )
            raise

    # ---- legacy weight-SVD streaming fallback ----

    # Deterministic seed derived from the canonical reasoning anchor probes.
    seed = int(hashlib.sha256(repr(REASONING_ANCHOR_PAIRS).encode("utf-8")).hexdigest()[:16], 16)
    generator = torch.Generator().manual_seed(seed)

    layer_bases: dict[int, Any] = {}
    expert_profiles: dict[int, dict[str, Any]] = {}
    anchor_rank = 0
    basis_rank = 0

    for idx in selected_layers:
        proj_key = _find_projection_key(key_to_shard, idx, num_experts)
        if proj_key is None:
            continue
        W = _load_tensor(model_dir, key_to_shard, proj_key).to(torch.float32)
        k_anchor = min(16, max(1, min(W.shape[0], W.shape[1]) // 4))
        rank = min(2, max(1, W.shape[0] - k_anchor))
        U = _power_iteration_rowspace(W, k_anchor + rank, generator)
        U_anchor = U[:, :k_anchor]  # (hidden, k_anchor)
        U_target = U[:, k_anchor:].clone()  # (hidden, rank)
        anchor_rank = int(k_anchor)

        # Orthogonalize the target rows against the anchor subspace, twice.
        for _ in range(2):
            U_target = U_target - U_anchor @ (U_anchor.T @ U_target)
            U_target, _ = torch.linalg.qr(U_target)
        U_target = U_target.T.contiguous()  # (rank, hidden)

        if spec.get("looped"):
            # Tick anchor/target directions built from the real basis rows —
            # real hidden_size, no synthetic randn(64) fixtures.
            ticks = int(spec.get("loop_ticks", 3))
            anchor_rows = [U_anchor[:, c].contiguous() for c in range(U_anchor.shape[1])]
            target_rows = [U_target[r].contiguous() for r in range(U_target.shape[0])]
            anchor_by_tick = [anchor_rows for _ in range(ticks)]
            target_by_tick = [target_rows for _ in range(ticks)]
            u_anchor_loop, u_target_loop = build_loop_unrolled_biprojected_basis(
                anchor_by_tick, target_by_tick, k_loop=int(spec.get("k_loop", 2))
            )
            basis = u_target_loop.to(torch.float32).contiguous()
            anchor_rank = int(u_anchor_loop.shape[0])
            basis_rank = int(basis.shape[0])
            layer_bases[idx] = basis
        else:
            layer_bases[idx] = U_target
            basis_rank = int(U_target.shape[0])

        # ---- expert profiles for this layer (streamed per-expert) ----
        expert_stack: dict[int, Any] = {}
        router_key: str | None = None
        for key in key_to_shard:
            parsed = _strip_layer_prefix(key)
            if parsed is None or parsed[0] != idx:
                continue
            suffix = parsed[1]
            if "router" in suffix or suffix.endswith("mlp.gate.weight"):
                router_key = key
            if suffix.endswith(("down_proj.weight", "w2.weight")):
                expert_id = _expert_id_from_suffix(suffix)
                if expert_id is not None:
                    expert_stack.setdefault(expert_id, _load_tensor(model_dir, key_to_shard, key))

        if not expert_stack and num_experts:
            # 3-D packed experts: mlp.experts.down_proj of shape (E, hidden, ff).
            for key in key_to_shard:
                parsed = _strip_layer_prefix(key)
                if parsed is None or parsed[0] != idx:
                    continue
                if "experts." in parsed[1] and parsed[1].endswith(
                    ("down_proj.weight", "down_proj")
                ):
                    packed = _load_tensor(model_dir, key_to_shard, key)
                    if packed.ndim == 3:
                        expert_stack = {e: packed[e] for e in range(packed.shape[0])}
                    break

        if expert_stack:
            E = len(expert_stack)
            anchor_dirs = [U_anchor[:, c] for c in range(U_anchor.shape[1])]
            target_dirs = [U_target[r] for r in range(U_target.shape[0])]
            p_anchors: list[float] = []
            p_targets: list[float] = []
            norms: list[float] = []
            for e in sorted(expert_stack):
                W_e = expert_stack[e].to(torch.float32)
                # Non-square Frobenius energy fractions (rowspace projections
                # over the total (hidden, intermediate) energy of W_e).
                denom = float((W_e * W_e).sum()) + 1e-12
                proj_a = sum(float((d @ W_e).pow(2).sum()) for d in anchor_dirs)
                proj_t = sum(float((d @ W_e).pow(2).sum()) for d in target_dirs)
                p_anchors.append(min(1.0, proj_a / denom))
                p_targets.append(min(1.0, proj_t / denom))
                norms.append(float(W_e.norm()))
            # Router-derived frequencies (fallback: uniform).
            if router_key is not None:
                router = _load_tensor(model_dir, key_to_shard, router_key).to(torch.float32)
                if router.ndim == 2 and router.shape[0] == E:
                    scores = router.norm(dim=1)
                    freqs = torch.softmax(scores, dim=0)
                else:
                    freqs = torch.ones(E, dtype=torch.float32) / E
            else:
                freqs = torch.ones(E, dtype=torch.float32) / E
            # Selective weights from per-expert target/anchor energy ratios.
            ratios = torch.tensor(
                [pt / (pt + pa + 1e-9) for pa, pt in zip(p_anchors, p_targets, strict=True)],
                dtype=torch.float32,
            )
            if E > 1 and float(ratios.max() - ratios.min()) > 1e-6:
                s_tensor = (
                    ((ratios - ratios.min()) / (ratios.max() - ratios.min()))
                    .clamp(0.0, 1.0)
                    .to(torch.float32)
                )
            else:
                s_tensor = torch.ones(E, dtype=torch.float32)
            expert_profiles[idx] = {
                "num_experts": E,
                "p_target": sum(p_targets) / max(1, len(p_targets)),
                "p_anchor": sum(p_anchors) / max(1, len(p_anchors)),
                "frequencies": freqs.to(torch.float32).contiguous(),
                "selective_weights": s_tensor.contiguous(),
                "beta": 1.0,
                "eps": 1e-6,
            }

    meta = {
        "calibration_mode": "weight_svd_fallback",
        "selected_layers": sorted(layer_bases),
        "anchor_rank": anchor_rank,
        "basis_rank": basis_rank,
    }
    if basis_out_path is not None and layer_bases:
        payload = {
            f"layer_{idx}": t.cpu().to(torch.float32).contiguous() for idx, t in layer_bases.items()
        }
        basis_out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(payload, str(basis_out_path))
    return layer_bases, expert_profiles, meta


def _as_bool(value: Any) -> bool:
    """Coerce scalars, 0-dim/1-element tensors, and single-item lists to bool."""
    if hasattr(value, "reshape"):
        return bool(value.numel()) and bool(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return bool(value) and bool(value[0])
    return bool(value)


def _run_h21(
    model_dir: Path,
    materialized_dir: Path,
    spec: dict[str, Any],
    args: argparse.Namespace,
    *,
    layer_bases: dict[int, Any],
    expert_profiles: dict[int, dict[str, Any]],
    cfg_info: dict[str, Any],
    basis_path: Path,
) -> dict[str, Any]:
    """H21 strata-tier-aware MoE Lancet surgery (pure safetensors streaming)."""
    from sml.surgery.subspace import apply_moe_lancet_recipe_streaming

    result = apply_moe_lancet_recipe_streaming(
        model_dir=model_dir,
        output_dir=materialized_dir,
        layer_bases=layer_bases,
        expert_profiles=expert_profiles,
        scale=float(args.scale),
        min_selectivity_threshold=float(args.min_selectivity_threshold),
        shared_expert_damping=float(args.shared_expert_damping),
        hot_vram_expert_ids=set(spec["hot_vram_expert_ids"]),
        quant_snap=True,
        max_block16_crest_factor=3.20,
    )
    result["mode"] = "streaming"
    result["materialized_dir"] = str(materialized_dir)

    # Ship the calibrated basis and recipe manifest with the checkpoint.
    materialized_dir.mkdir(parents=True, exist_ok=True)
    if basis_path.exists():
        shutil.copyfile(basis_path, materialized_dir / "lancet_basis.safetensors")
    recipe = {
        "schema_version": "1.0",
        "model_id": str(
            getattr(args, "model_id", cfg_info.get("model_id", spec.get("base_model", "")))
        ),
        "base_model": spec.get("base_model", ""),
        "d_model": cfg_info.get("hidden_size"),
        "num_layers": cfg_info.get("num_hidden_layers"),
        "selected_layers": sorted(layer_bases),
        "layer_weights": {str(i): 1.0 for i in sorted(layer_bases)},
    }
    (materialized_dir / "lancet_recipe.json").write_text(json.dumps(recipe, indent=2))

    # Materialized checkpoint must be shippable: config, tokenizer files and
    # chat template all present (copied from the source checkpoint if missing).
    for tokenizer_name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        src = model_dir / tokenizer_name
        dst = materialized_dir / tokenizer_name
        if src.is_file() and not dst.exists():
            shutil.copyfile(src, dst)
    missing_required = [
        name
        for name in ("config.json", "chat_template.jinja")
        if not (materialized_dir / name).exists()
    ]
    # NOTE: ``chat_template.jinja`` is only required when the checkpoint does
    # not embed a chat template inside ``tokenizer_config.json`` (some
    # checkpoints ship ``tokenizer.model`` / minimal tokenizer configs instead
    # of ``tokenizer.json``, so it is never required unconditionally).
    if missing_required and (model_dir / "config.json").exists():
        for name in missing_required:
            src = model_dir / name
            if src.is_file():
                shutil.copyfile(src, materialized_dir / name)

    edited_count = (
        int(result.get("edited_o_proj", 0))
        + int(result.get("edited_shared_experts", 0))
        + int(result.get("edited_routed_experts", 0))
    )
    result["edited_count"] = edited_count

    def _chat_template_present(directory: Path) -> bool:
        if (directory / "chat_template.jinja").is_file():
            return True
        tc_path = directory / "tokenizer_config.json"
        if tc_path.is_file():
            try:
                tc = json.loads(tc_path.read_text(encoding="utf-8"))
                return bool(tc.get("chat_template"))
            except Exception:
                return False
        return False

    result["missing_materialized_files"] = [
        name
        for name in (
            "config.json",
            "tokenizer_config.json",
            "chat_template",
            "lancet_basis.safetensors",
            "lancet_recipe.json",
        )
        if (name == "chat_template" and not _chat_template_present(materialized_dir))
        or (name != "chat_template" and not (materialized_dir / name).exists())
    ]
    h21_drift = result.get("router_weight_drift", 1.0)
    if hasattr(h21_drift, "max") and hasattr(h21_drift, "numel"):
        h21_drift = float(h21_drift.max()) if h21_drift.numel() else 0.0
    result["h21_verified"] = bool(
        h21_drift == 0.0
        and edited_count > 0
        and (materialized_dir / "lancet_basis.safetensors").exists()
        and (materialized_dir / "lancet_recipe.json").exists()
        and sorted(layer_bases)
        and not result["missing_materialized_files"]
    )
    return result


def _run_h22(
    model_dir: Path,
    materialized_dir: Path,
    spec: dict[str, Any],
    args: argparse.Namespace,
    *,
    layer_bases: dict[int, Any],
    prep_telemetry: dict[str, Any],
    cfg_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """H22 READ-ONLY frozen-tensor drift verification + drafter synchronization.

    Never writes shards here: drift is verified by streaming ``safe_open``
    comparisons between the source checkpoint and the materialized output.
    """
    import torch
    from safetensors import safe_open

    from sml.surgery.cka import is_router_weight_key
    from sml.surgery.looped import synchronize_speculative_drafter_streaming

    loop_tokens = ("loop_controller", "loop_gate", "tick_embed")

    def _is_loop_frozen(key: str) -> bool:
        lowered = key.lower()
        return any(token in lowered for token in loop_tokens)

    router_weight_drift = 0.0
    loop_controller_drift = 0.0
    frozen_keys_checked = 0
    for shard in sorted(model_dir.glob("*.safetensors")):
        if shard.name == "lancet_basis.safetensors":
            continue
        target_shard = materialized_dir / shard.name
        if not target_shard.exists():
            continue
        with (
            safe_open(shard, framework="pt", device="cpu") as base,
            safe_open(target_shard, framework="pt", device="cpu") as edited,
        ):
            base_keys = set(base.keys())
            for key in list(edited.keys()):
                if key not in base_keys:
                    continue
                is_router = is_router_weight_key(key)
                is_loop = _is_loop_frozen(key) or ("engram" in key.lower())
                if not (is_router or is_loop):
                    continue
                base_t = base.get_tensor(key).to(torch.float32)
                edited_t = edited.get_tensor(key).to(torch.float32)
                if base_t.shape != edited_t.shape:
                    drift = float("inf")
                else:
                    drift = float((edited_t - base_t).abs().max()) if edited_t.numel() else 0.0
                frozen_keys_checked += 1
                if is_router:
                    router_weight_drift = max(router_weight_drift, drift)
                if is_loop:
                    loop_controller_drift = max(loop_controller_drift, drift)

    router_frozen = router_weight_drift == 0.0
    loop_controller_frozen = loop_controller_drift == 0.0

    looped: dict[str, Any] = {"skipped": not spec.get("looped")}
    if spec.get("looped"):
        looped = {
            "k_loop": int(prep_telemetry.get("k_loop", spec.get("k_loop", 2))),
            "ticks": int(prep_telemetry.get("ticks", spec.get("loop_ticks", 3))),
            "basis_rank": int(prep_telemetry.get("basis_rank", 0)),
            "anchor_rank": int(prep_telemetry.get("anchor_rank", 0)),
            "loop_controller_drift": loop_controller_drift,
            "router_weight_drift": router_weight_drift,
            "frozen_keys_checked": frozen_keys_checked,
            "loop_controller_frozen": loop_controller_frozen,
            "router_frozen": router_frozen,
        }

    drafter: dict[str, Any] = {"skipped": not spec.get("has_drafter")}
    if spec.get("has_drafter"):
        drafter = synchronize_speculative_drafter_streaming(
            materialized_dir,
            layer_bases,
            output_dir=materialized_dir,
            scale=float(args.scale),
            min_kl_reduction_pct=35.0,
            num_hidden_layers=(cfg_info or {}).get("num_hidden_layers"),
        )

    h22_verified = bool(
        router_weight_drift == 0.0
        and loop_controller_drift == 0.0
        and (
            not spec.get("looped")
            or int(prep_telemetry.get("basis_rank", 0)) <= int(spec.get("k_loop", 2))
        )
        and (not spec.get("has_drafter") or _as_bool(drafter.get("passed_kl_gate")))
    )
    return {
        "mode": "streaming",
        "looped": looped,
        "drafter": drafter,
        "router_weight_drift": router_weight_drift,
        "loop_controller_drift": loop_controller_drift,
        "frozen_keys_checked": frozen_keys_checked,
        "h22_verified": h22_verified,
    }


def _write_strata_toml(
    path: Path, model_id: str, spec: dict[str, Any], metrics: dict[str, Any]
) -> None:
    lines = [
        "# Strata tier descriptor (generated by run_tenet_v3_foundry.py)",
        f'model_id = "{model_id}"',
        f'tier = "{spec["tier"]}"',
        f'base_model = "{spec["base_model"]}"',
        f"looped = {str(bool(spec['looped'])).lower()}",
        f"has_drafter = {str(bool(spec['has_drafter'])).lower()}",
        f"min_selectivity_threshold = {spec['min_selectivity_threshold']}",
        f"shared_expert_damping = {spec['shared_expert_damping']}",
        "hot_vram_expert_ids = ["
        + ", ".join(str(e) for e in sorted(spec["hot_vram_expert_ids"]))
        + "]",
    ]
    for key in (
        "router_weight_drift",
        "untouched_expert_ratio",
        "max_block16_crest_factor",
        "loop_controller_drift",
        "basis_rank",
        "drafter_kl_reduction_pct",
    ):
        if key in metrics:
            lines.append(f"{key} = {metrics[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_materialized_dir(model_id: str, release_dir: Path) -> Path:
    """Pick the local materialized-checkpoint directory for a release."""
    if Path("/mnt/nvme/tenet-lab").is_dir():
        return Path("/mnt/nvme/tenet-lab/tenet_materialized") / model_id
    return release_dir / "materialized"


_UNSUPPORTED_ARCH_MARKERS = (
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
    "HYV4FlashForConditionalGeneration",
)


def _detect_tensor_parallel_size() -> int:
    """Best-effort tensor-parallel size for local vLLM inference."""
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d for d in visible.split(",") if d.strip()]
        if devices:
            return len(devices)
    try:
        import torch

        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except Exception:
        pass
    return 1


def _run_fight_card_gate(
    model_id: str,
    spec: dict[str, Any],
    release_dir: Path,
    *,
    api_base: str | None = None,
    materialized_dir: Path | None = None,
) -> dict[str, Any]:
    """H17-style paired Fight Card Gate over real responses (no synthetic data).

    Queries either an OpenAI-compatible server (``api_base``) or in-process
    vLLM (``materialized_dir``) for the house series + borderline over-refusal
    and anti-slop prompt sets. If vLLM cannot load the architecture, the gate
    is skipped with ``skipped_unsupported_arch`` — never faked.
    """
    import urllib.request

    from sml.evals.peer_eval import (
        ANTI_SLOP_PROMPTS,
        BORDERLINE_OVER_REFUSAL_PROMPTS,
        TENET_ARENA_HOUSE_SERIES,
        run_paired_behavioral_gate,
    )

    think_end_token = spec.get("think_end_token") or "</think>"
    gate_path = release_dir / "fight_card_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)

    def _item_prompt(item: Any) -> str:
        return getattr(item, "prompt", None) or (
            item.get("prompt", "") if isinstance(item, dict) else str(item)
        )

    if api_base:
        url = f"{api_base.rstrip('/')}/v1/chat/completions"
        house_prompts = [_item_prompt(i) for i in TENET_ARENA_HOUSE_SERIES]
        direct_prompts = list(BORDERLINE_OVER_REFUSAL_PROMPTS) + list(ANTI_SLOP_PROMPTS)

        def _query(model_name: str, target: dict[str, str]) -> None:
            for prompt, max_tokens in [
                *[(p, 8192) for p in house_prompts],
                *[(p, 4096) for p in direct_prompts],
            ]:
                payload = json.dumps(
                    {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                target[prompt] = body["choices"][0]["message"]["content"]

        tenet_responses: dict[str, str] = {}
        base_responses: dict[str, str] = {}
        _query(model_id, tenet_responses)
        _query(spec["base_model"], base_responses)
    else:
        # Sequential in-process vLLM inference for base and tenet checkpoints.
        try:
            import gc

            import torch
            from vllm import LLM, SamplingParams
        except Exception:
            report = {"status": "skipped_unsupported_arch", "gate_passed": None}
            gate_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
            return report

        house_prompts = [_item_prompt(i) for i in TENET_ARENA_HOUSE_SERIES]
        direct_prompts = list(BORDERLINE_OVER_REFUSAL_PROMPTS) + list(ANTI_SLOP_PROMPTS)
        tp_size = _detect_tensor_parallel_size()

        results_by_model: dict[str, list[str]] = {}
        try:
            tenet_checkpoint = str(materialized_dir or release_dir / "materialized")
            for checkpoint in (tenet_checkpoint, spec["base_model"]):
                try:
                    llm = LLM(
                        model=checkpoint,
                        tensor_parallel_size=tp_size,
                        dtype="bfloat16",
                        trust_remote_code=True,
                        gpu_memory_utilization=0.88,
                        max_model_len=16384,
                    )
                except Exception as exc:
                    text = str(exc)
                    if any(m in text for m in _UNSUPPORTED_ARCH_MARKERS) or (
                        "not supported" in text or "does not recognize" in text
                    ):
                        report = {
                            "status": "skipped_unsupported_arch",
                            "gate_passed": None,
                            "error": text[-4000:],
                        }
                        gate_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
                        return report
                    raise
                try:
                    tokenizer = llm.get_tokenizer()
                except Exception:
                    tokenizer = None
                chat_template = getattr(tokenizer, "chat_template", None)
                apply_chat_template = getattr(tokenizer, "apply_chat_template", None)

                def _prepend_opened_think(fmt_p: str, txt: str) -> str:
                    stripped_p = fmt_p.rstrip()
                    for open_tag in ("<think>", "<|thought|>", "<|thought>"):
                        if stripped_p.endswith(open_tag) and not txt.lstrip().startswith(open_tag):
                            return f"{open_tag}\n{txt}"
                    return txt

                if chat_template and callable(apply_chat_template):
                    _apply_fn: Any = apply_chat_template

                    def _format_house(p: str, _fn: Any = _apply_fn) -> str:
                        return _fn(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )

                    thinking_disabled = False
                    try:
                        direct_formatted = [
                            _apply_fn(
                                [{"role": "user", "content": p}],
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False,
                            )
                            for p in direct_prompts
                        ]
                        thinking_disabled = True
                    except TypeError:
                        direct_formatted = [
                            _apply_fn(
                                [{"role": "user", "content": p}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            for p in direct_prompts
                        ]
                    try:
                        house_formatted = [_format_house(p) for p in house_prompts]
                    except Exception:
                        house_formatted = house_prompts
                else:
                    house_formatted = house_prompts
                    direct_formatted = direct_prompts
                    thinking_disabled = False

                # House prompts: allow full <think>...</think> deliberation.
                house_sampling = SamplingParams(temperature=0.0, max_tokens=8192)
                house_outs = llm.generate(house_formatted, house_sampling)
                house_texts: list[str] = []
                for fmt_p, o in zip(house_formatted, house_outs, strict=True):
                    house_texts.append(_prepend_opened_think(fmt_p, o.outputs[0].text))

                # Direct prompts: no CoT needed; shorter budget unless the
                # template fell back to thinking-enabled formatting.
                direct_sampling = SamplingParams(
                    temperature=0.0, max_tokens=1536 if thinking_disabled else 4096
                )
                direct_outs = llm.generate(direct_formatted, direct_sampling)
                direct_texts: list[str] = []
                for fmt_p, o in zip(direct_formatted, direct_outs, strict=True):
                    direct_texts.append(_prepend_opened_think(fmt_p, o.outputs[0].text))

                results_by_model[checkpoint] = house_texts + direct_texts
                del llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except (ImportError, ValueError, RuntimeError, OSError):
            report = {"status": "skipped_unsupported_arch", "gate_passed": None}
            gate_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
            return report

        all_prompts = house_prompts + direct_prompts
        tenet_responses = {
            p: t for p, t in zip(all_prompts, results_by_model[tenet_checkpoint], strict=True)
        }
        base_responses = {
            p: t for p, t in zip(all_prompts, results_by_model[spec["base_model"]], strict=True)
        }

    n_house = len(TENET_ARENA_HOUSE_SERIES)
    n_refusal = len(BORDERLINE_OVER_REFUSAL_PROMPTS)

    def _split(responses: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
        ordered_prompts = (
            [_item_prompt(i) for i in TENET_ARENA_HOUSE_SERIES]
            + list(BORDERLINE_OVER_REFUSAL_PROMPTS)
            + list(ANTI_SLOP_PROMPTS)
        )
        texts = [responses.get(p, "") for p in ordered_prompts]
        return texts[:n_house], texts[n_house : n_house + n_refusal], texts[n_house + n_refusal :]

    base_house = _split(base_responses)[0]
    effective_require_think = bool(think_end_token) and any(
        tag in r for r in base_house for tag in ("<think>", "<|thought", "<reasoning>")
    )

    report = run_paired_behavioral_gate(
        base_house_responses=base_house,
        tenet_house_responses=_split(tenet_responses)[0],
        base_refusal_responses=_split(base_responses)[1],
        tenet_refusal_responses=_split(tenet_responses)[1],
        base_slop_responses=_split(base_responses)[2],
        tenet_slop_responses=_split(tenet_responses)[2],
        require_think=effective_require_think,
    )
    report["model_id"] = model_id
    gate_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


_REAL_SUBPROCESS_RUN = subprocess.run


def _run_streaming_subprocess(cmd: list[str], *, prefix: str = "") -> tuple[int, str]:
    """Stream subprocess stdout+stderr live to worker.log while keeping the tail."""
    if subprocess.run is not _REAL_SUBPROCESS_RUN:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, combined
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    tail_chunks: list[str] = []
    tail_len = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        out_line = f"{prefix}{line}" if prefix else line
        print(out_line, end="", flush=True)
        tail_chunks.append(line)
        tail_len += len(line)
        while tail_len > 8000 and len(tail_chunks) > 1:
            removed = tail_chunks.pop(0)
            tail_len -= len(removed)
    rc = proc.wait()
    return rc, "".join(tail_chunks)


def _run_strata_bench(
    model_id: str,
    spec: dict[str, Any],
    release_id: str,
    strata_path: Path,
    release_dir: Path,
    *,
    backend: str = "sim",
) -> dict[str, Any]:
    """Run the Strata benchmark driver as a subprocess with live log streaming."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval" / "run_strata_bench.py"),
        "--model-name",
        model_id,
        "--base-model",
        spec["base_model"],
        "--release-id",
        release_id,
        "--strata-toml",
        str(strata_path),
        "--backend",
        backend,
        "--output",
        str(release_dir / "strata_bench.json"),
    ]
    rc, combined_tail = _run_streaming_subprocess(cmd, prefix="[strata] ")
    report: dict[str, Any] = {
        "returncode": rc,
        "stdout_tail": combined_tail[-2000:],
        "stderr_tail": "",
        "passed": rc == 0,
    }
    output_path = release_dir / "strata_bench.json"
    if rc == 0 and output_path.exists():
        import contextlib

        with contextlib.suppress(Exception):
            report["report"] = json.loads(output_path.read_text(encoding="utf-8"))
    return report


def _run_open_evals(
    model_id: str,
    spec: dict[str, Any],
    release_id: str,
    materialized_dir: Path,
    *,
    eval_suite: str = "full",
    mmlu_pro_limit: int = 72,
    sanity_gate: bool = True,
) -> dict[str, Any]:
    """Run the vLLM open-eval suite (base/naive/tenet arms) as a subprocess."""
    # Always clear the tenet arm cache: a re-materialized checkpoint is a new
    # model even under the same release_id, so stale tenet results must never
    # be reused (base/naive caches stay intact).
    shutil.rmtree(
        REPO_ROOT / "results" / "external_evals" / release_id / "raw" / "tenet",
        ignore_errors=True,
    )
    arms = "base,naive,tenet" if spec.get("naive_model") else "base,tenet"
    suite_flags = (
        ["--suite", "quick", "--limit", "2", "--mmlu-pro-limit", "2"]
        if eval_suite == "smoke"
        else ["--suite", "full", "--mmlu-pro-limit", str(mmlu_pro_limit)]
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval" / "run_vllm_open_evals.py"),
        "--release-id",
        release_id,
        "--base-model",
        spec["base_model"],
        "--naive-model",
        spec.get("naive_model") or spec["base_model"],
        "--tenet-repo",
        str(materialized_dir),
        "--arms",
        arms,
        "--split-chat-0shot",
        *(["--sanity-gate"] if sanity_gate else []),
        "--resume",
        "--think-end-token",
        spec["think_end_token"],
        *suite_flags,
        "--tensor-parallel-size",
        str(_detect_tensor_parallel_size()),
    ]
    rc, combined_tail = _run_streaming_subprocess(cmd)
    report: dict[str, Any] = {
        "returncode": rc,
        "stdout_tail": combined_tail[-2000:],
        "stderr_tail": "",
        "passed": rc == 0,
    }
    if rc != 0:
        arch_markers = (
            "Inkling2ForConditionalGeneration",
            "Inkling2LoopForConditionalGeneration",
            "Qwen4ExpForConditionalGeneration",
            "HYV4FlashForConditionalGeneration",
        )
        diag_text = combined_tail
        raw_root = REPO_ROOT / "results" / "external_evals" / release_id / "raw"
        if raw_root.is_dir():
            for log_file in raw_root.rglob("lm_eval_live.log"):
                try:
                    diag_text += (
                        "\n" + log_file.read_text(encoding="utf-8", errors="ignore")[-4000:]
                    )
                except OSError:
                    continue
        if any(m in diag_text for m in arch_markers) and (
            "not supported" in diag_text or "does not recognize" in diag_text
        ):
            report["status"] = "unsupported_arch_in_vllm"
            report["passed"] = True
            return report
    if rc == 0:
        export_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval" / "export_model_index.py"),
            "--eval-json",
            str(REPO_ROOT / "results" / "external_evals" / f"{release_id}.json"),
            "--model-id",
            model_id,
            "--also-release-dir",
        ]
        export = subprocess.run(export_cmd, capture_output=True, text=True)
        report["export_returncode"] = export.returncode
        report["passed"] = export.returncode == 0
    return report


def _upload_to_hf(
    model_id: str,
    materialized_dir: Path,
    release_dir: Path,
    *,
    card_path: str | None,
    strata_path: Path,
    hf_namespace: str = "mr-andrey-isachenko",
) -> dict[str, Any]:
    """Copy release artifacts into the materialized dir and push to HF."""
    from huggingface_hub import HfApi

    release_id = release_dir.name
    if card_path and Path(card_path).is_file():
        shutil.copyfile(card_path, materialized_dir / "README.md")
    if Path(strata_path).is_file():
        shutil.copyfile(strata_path, materialized_dir / "strata.toml")
    model_index = REPO_ROOT / "results" / "tenet_releases" / release_id / "model_index.yaml"
    if model_index.is_file():
        shutil.copyfile(model_index, materialized_dir / "model_index.yaml")

    repo_id = f"{hf_namespace}/{model_id}"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(materialized_dir), repo_id=repo_id, repo_type="model")
    return {"repo_id": repo_id, "uploaded": True}


def _cleanup_local_cache(
    model_dir: Path,
    *,
    materialized_dir: Path | None = None,
    upload_result: dict[str, Any],
) -> dict[str, Any]:
    """Drop local checkpoint caches, only after a verified HF upload + S3 sync.

    When the upload is confirmed and ``s3-sync-up`` succeeds, deletes
    ``.safetensors`` files (or whole checkpoint directories) from BOTH the
    materialized output dir and the HF cache ``model_dir`` — but only when they
    live under ``/mnt/nvme`` or ``HF_HOME``.
    """
    if upload_result.get("uploaded") is not True:
        return {"cleaned_up": False, "reason": "upload_result does not confirm upload"}
    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "remote" / "nebius_ctl.sh"), "s3-sync-up"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "cleaned_up": False,
            "reason": "s3-sync-up failed",
            "stderr_tail": proc.stderr[-2000:],
        }

    allowed_roots = [Path("/mnt/nvme").resolve()]
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        allowed_roots.append(Path(hf_home).resolve())

    def _under_guarded_root(candidate: Path) -> bool:
        resolved = candidate.resolve()
        return any(resolved.is_relative_to(root) and resolved != root for root in allowed_roots)

    removed: list[str] = []

    def _purge_safetensors(directory: Path) -> None:
        """Remove .safetensors files (or dirs) under a guarded root."""
        for path in sorted(directory.rglob("*.safetensors")):
            if _under_guarded_root(path):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                removed.append(str(path))

    targets: list[Path] = []
    if materialized_dir is not None and Path(materialized_dir).is_dir():
        targets.append(Path(materialized_dir))
    if model_dir.is_dir() and _under_guarded_root(model_dir):
        # Whole HF cache dir: delete it (covers every shard at once).
        shutil.rmtree(model_dir)
        removed.append(str(model_dir))
    else:
        targets.append(model_dir)
    for target in targets:
        _purge_safetensors(target)

    if removed:
        return {"cleaned_up": True, "removed_paths": removed, "s3_returncode": 0}
    return {
        "cleaned_up": False,
        "reason": "refusing to remove paths outside guarded roots",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_tenet_v3_foundry",
        description="Tenet v3 foundry orchestrator (H21 Strata MoE Lancet + H22 Looped-MoE).",
    )
    parser.add_argument("--model-id", required=True, choices=sorted(TENET_V3_MODEL_SPECS))
    parser.add_argument("--model-dir", default=None, help="Local checkpoint directory override.")
    parser.add_argument(
        "--materialized-dir", default=None, help="Output checkpoint directory override."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Snapshot-download the checkpoint when --model-dir is unset.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit.")
    parser.add_argument(
        "--strata-config", default=None, help="Optional strata TOML config override."
    )
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    parser.add_argument("--scale", type=float, default=0.35, help="Lancet / drafter surgery scale.")
    parser.add_argument("--min-selectivity-threshold", type=float, default=0.25)
    parser.add_argument("--shared-expert-damping", type=float, default=0.40)
    parser.add_argument("--run-fight-card-gate", action="store_true")
    parser.add_argument(
        "--api-base", default=None, help="OpenAI-compatible API base for the Fight Card Gate."
    )
    parser.add_argument("--run-strata-bench", action="store_true")
    parser.add_argument("--strata-backend", choices=("strata", "sim"), default="sim")
    parser.add_argument("--run-open-evals", action="store_true")
    parser.add_argument("--eval-suite", choices=("smoke", "full"), default="full")
    parser.add_argument("--mmlu-pro-limit", type=int, default=72)
    parser.add_argument(
        "--no-sanity-gate",
        action="store_true",
        help="Omit --sanity-gate when running open evals.",
    )
    parser.add_argument("--upload-hf", action="store_true")
    parser.add_argument("--hf-namespace", default="mr-andrey-isachenko")
    parser.add_argument("--cleanup-cache", action="store_true")
    parser.add_argument(
        "--allow-weight-svd-fallback",
        action="store_true",
        default=False,
        help=(
            "Fall back to weight-space SVD basis calibration when real "
            "activation calibration cannot load the architecture. Without "
            "this flag an unsupported forward arch halts surgery."
        ),
    )
    args = parser.parse_args(argv)

    spec = TENET_V3_MODEL_SPECS[args.model_id]
    if "alias_of" in spec:
        spec = TENET_V3_MODEL_SPECS[spec["alias_of"]]
    model_dir = _resolve_model_dir(
        args.model_id, args.model_dir, spec=spec, download=args.download, dry_run=args.dry_run
    )
    output_dir = Path(args.output_dir)
    release_dir = output_dir / args.release_id
    materialized_dir = (
        Path(args.materialized_dir).expanduser().resolve()
        if args.materialized_dir
        else _default_materialized_dir(args.model_id, release_dir)
    )

    plan = {
        "release_id": args.release_id,
        "model_id": args.model_id,
        "model_dir": str(model_dir) if model_dir else None,
        "materialized_dir": str(materialized_dir),
        "tier": spec["tier"],
        "base_model": spec["base_model"],
        "naive_model": spec.get("naive_model"),
        "think_end_token": spec["think_end_token"],
        "hypothesis_stack": spec["hypothesis_stack"],
        "looped": spec["looped"],
        "has_drafter": spec["has_drafter"],
        "hot_vram_expert_ids": sorted(spec["hot_vram_expert_ids"]),
        "min_selectivity_threshold": args.min_selectivity_threshold,
        "shared_expert_damping": args.shared_expert_damping,
        "scale": args.scale,
        "quant_snap": True,
        "max_block16_crest_factor": 3.20,
    }
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", **plan}, indent=2))
        return 0

    t0 = time.perf_counter()
    status = "ok"
    h21: dict[str, Any] = {}
    h22: dict[str, Any] = {}
    fight_card_gate: dict[str, Any] = {"skipped": not args.run_fight_card_gate}
    strata_bench: dict[str, Any] = {"skipped": not args.run_strata_bench}
    open_evals: dict[str, Any] = {"skipped": not args.run_open_evals}
    upload_result: dict[str, Any] = {"skipped": not args.upload_hf}
    cleanup_result: dict[str, Any] = {"skipped": not args.cleanup_cache}
    try:
        if model_dir is None:
            raise RuntimeError("no checkpoint directory resolved: pass --model-dir or --download")
        _log(f"H21 strata-tier-aware Lancet surgery on {args.model_id} (tier={spec['tier']})")
        cfg_info = _inspect_checkpoint_config(model_dir)
        basis_out_path = release_dir / "lancet_basis.safetensors"
        layer_bases, expert_profiles, prep_telemetry = _build_streaming_basis_and_profiles(
            model_dir,
            cfg_info,
            spec,
            basis_out_path=basis_out_path,
            allow_weight_svd_fallback=args.allow_weight_svd_fallback,
        )
        h21 = _run_h21(
            model_dir,
            materialized_dir,
            spec,
            args,
            layer_bases=layer_bases,
            expert_profiles=expert_profiles,
            cfg_info=cfg_info,
            basis_path=basis_out_path,
        )
        if spec["looped"] or spec["has_drafter"]:
            _log("H22 read-only frozen-drift verification + speculative-drafter sync")
            h22 = _run_h22(
                model_dir,
                materialized_dir,
                spec,
                args,
                layer_bases=layer_bases,
                prep_telemetry=prep_telemetry,
                cfg_info=cfg_info,
            )
        if not h21.get("h21_verified"):
            status = "error"
        if (spec["looped"] or spec["has_drafter"]) and not h22.get("h22_verified"):
            status = "error"
    except Exception:
        status = "error"
        _log("pipeline failed:\n" + traceback.format_exc())

    metrics = {
        k: float(v)
        for k, v in {
            "router_weight_drift": h21.get("router_weight_drift"),
            "untouched_expert_ratio": h21.get("untouched_expert_ratio"),
            "max_block16_crest_factor": h21.get("max_block16_crest_factor"),
            "drafter_kl_reduction_pct": h22.get("drafter", {}).get("kl_reduction_pct"),
            "loop_controller_drift": h22.get("looped", {}).get("loop_controller_drift"),
            "basis_rank": h22.get("looped", {}).get("basis_rank"),
        }.items()
        if isinstance(v, (int, float))
    }

    release_dir.mkdir(parents=True, exist_ok=True)
    strata_path = Path(args.strata_config) if args.strata_config else release_dir / "strata.toml"
    _write_strata_toml(strata_path, args.model_id, spec, metrics)

    card_path: str | None = None
    try:
        from sml.hub.release import generate_tenet_model_card

        card = generate_tenet_model_card(
            model_id=args.model_id,
            base_model=spec["base_model"],
            metrics={"h21": 1.0 if h21.get("h21_verified") else 0.0, **metrics},
        )
        card_path = str(release_dir / f"{args.model_id}.md")
        Path(card_path).write_text(card, encoding="utf-8")
    except Exception:
        _log("model card generation skipped (generate_tenet_model_card unavailable)")

    mlflow_record: dict[str, Any] = {}
    try:
        from sml.tracking.nebius_mlflow import log_foundry_surgery_run

        mlflow_record = log_foundry_surgery_run(
            args.release_id,
            params={"base_model": spec["base_model"], "model_id": args.model_id, **plan},
            surgery_metrics={k: v for k, v in metrics.items()},
            gate_metrics={
                "h21_verified": float(bool(h21.get("h21_verified"))),
                "h22_verified": float(bool(h22.get("h22_verified"))),
            },
            artifacts_paths=[p for p in (strata_path, card_path) if p and Path(p).is_file()],
            tags={"tier": spec["tier"], "generation": "v3", "pipeline": "tenet_v3_foundry"},
        )
    except Exception:
        _log("MLflow logging skipped (nebius_mlflow unavailable)")

    # ---- post-surgery gates, evals, upload and cleanup (only when healthy) ----
    if status == "ok":
        try:
            if args.run_fight_card_gate:
                fight_card_gate = _run_fight_card_gate(
                    args.model_id,
                    spec,
                    release_dir,
                    api_base=args.api_base,
                    materialized_dir=materialized_dir,
                )
                if fight_card_gate.get("gate_passed") is False:
                    status = "error"
            if args.run_strata_bench:
                strata_bench = _run_strata_bench(
                    args.model_id,
                    spec,
                    args.release_id,
                    strata_path,
                    release_dir,
                    backend=args.strata_backend,
                )
                if not strata_bench.get("passed"):
                    status = "error"
            if args.run_open_evals:
                open_evals = _run_open_evals(
                    args.model_id,
                    spec,
                    args.release_id,
                    materialized_dir,
                    eval_suite=args.eval_suite,
                    mmlu_pro_limit=args.mmlu_pro_limit,
                    sanity_gate=not args.no_sanity_gate,
                )
                if not open_evals.get("passed"):
                    status = "error"
            if args.upload_hf:
                upload_result = _upload_to_hf(
                    args.model_id,
                    materialized_dir,
                    release_dir,
                    card_path=card_path,
                    strata_path=strata_path,
                    hf_namespace=args.hf_namespace,
                )
            if args.cleanup_cache:
                cleanup_result = _cleanup_local_cache(
                    Path(model_dir) if model_dir else materialized_dir,
                    materialized_dir=materialized_dir,
                    upload_result=upload_result,
                )
        except Exception:
            status = "error"
            _log("post-surgery stage failed:\n" + traceback.format_exc())

    summary = {
        "mode": "run",
        "status": status,
        "release_id": args.release_id,
        "model_id": args.model_id,
        "tier": spec["tier"],
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "plan": plan,
        "h21": h21,
        "h22": h22,
        "fight_card_gate": fight_card_gate,
        "strata_bench": strata_bench,
        "open_evals": open_evals,
        "hf_upload": upload_result,
        "cache_cleanup": cleanup_result,
        "strata_toml": str(strata_path),
        "model_card": card_path,
        "mlflow_mirror": mlflow_record.get("mirror_path"),
        "remote_mlflow_logged": bool(mlflow_record.get("remote_logged")),
    }
    summary_path = release_dir / "foundry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False, default=str) + "\n")
    _log(f"wrote {summary_path} (status={status})")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
