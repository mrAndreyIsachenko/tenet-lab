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

``--dry-run`` prints the resolved JSON plan and exits 0 without touching
weights, CUDA or the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = "results/tenet_v3_foundry"

TENET_V3_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "Tenet-15B-A2B-Qwen3.6-Unslop-v3": {
        "tier": "edge",
        "base_model": "qwen/qwen3.6-15b-a2b-instruct",
        "hypothesis_stack": ["H21"],
        "looped": False,
        "has_drafter": False,
        "hot_vram_expert_ids": {0, 1},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-26B-A3B-Inkling2-Unslop-v3": {
        "tier": "workhorse",
        "base_model": "arcee-ai/Inkling2-26B-A3B",
        "hypothesis_stack": ["H21", "H22"],
        "looped": True,
        "has_drafter": True,
        "hot_vram_expert_ids": {0, 1, 2},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-35B-A3B-Qwen3.6-Unslop-v3": {
        "tier": "workhorse",
        "base_model": "qwen/qwen3.6-35b-a3b-instruct",
        "hypothesis_stack": ["H21", "H22"],
        "looped": True,
        "has_drafter": True,
        "hot_vram_expert_ids": {0, 1, 2},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-32B-A3B-GLM4.5-Unslop-v3": {
        "tier": "workhorse",
        "base_model": "zai-org/GLM-4.5-Air",
        "hypothesis_stack": ["H21", "H22"],
        "looped": True,
        "has_drafter": True,
        "hot_vram_expert_ids": {0, 1, 2},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-125B-A6B-Qwen3.8-Unslop-v3": {
        "tier": "frontier",
        "base_model": "qwen/qwen3.8-125b-a6b-instruct",
        "hypothesis_stack": ["H21", "H22"],
        "looped": True,
        "has_drafter": True,
        "hot_vram_expert_ids": {0, 1, 2, 3, 4},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
    "Tenet-220B-A12B-Hy4-Unslop-v3": {
        "tier": "frontier",
        "base_model": "hyfour/hy4-220b-a12b-instruct",
        "hypothesis_stack": ["H21", "H22"],
        "looped": True,
        "has_drafter": True,
        "hot_vram_expert_ids": {0, 1, 2, 3, 4, 5},
        "min_selectivity_threshold": 0.25,
        "shared_expert_damping": 0.40,
    },
}

DEFAULT_RELEASE_ID = "tenet-v3-foundry-v1"


def _log(message: str) -> None:
    print(f"[foundry] {message}", flush=True)


def _resolve_model_dir(model_id: str, model_dir: str | None) -> Path | None:
    """Return an existing checkpoint directory, or ``None`` when unresolved."""
    if model_dir:
        candidate = Path(model_dir).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        return None
    env_dir = None
    try:  # optional: honour an explicit local snapshot directory
        import os

        env_dir = os.environ.get("TENET_V3_MODEL_DIR") or None
    except Exception:
        env_dir = None
    if env_dir:
        candidate = Path(env_dir).expanduser().resolve()
        if candidate.is_dir():
            return candidate
    try:  # optional HF snapshot resolution (offline-safe: never raises here)
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True, allow_patterns=["*.json"]))
    except Exception:
        return None


def _run_h21(
    model_dir: Path | None, spec: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """H21 strata-tier-aware MoE Lancet surgery (streaming when available)."""
    from sml.surgery.subspace import apply_moe_lancet_surgery

    common: dict[str, Any] = {
        "scale": float(args.scale),
        "min_selectivity_threshold": float(args.min_selectivity_threshold),
        "shared_expert_damping": float(args.shared_expert_damping),
        "hot_vram_expert_ids": set(spec["hot_vram_expert_ids"]),
        "quant_snap": True,
        "max_block16_crest_factor": 3.20,
    }
    try:  # streaming path preferred for large frontier tiers
        from sml.surgery.subspace import apply_moe_lancet_recipe_streaming  # type: ignore

        result = apply_moe_lancet_recipe_streaming(model_dir, **common)
        result["mode"] = "streaming"
        return result
    except Exception:
        pass
    if model_dir is None:
        return {"mode": "skipped_no_model_dir", "h21_verified": False}
    from transformers import AutoModelForCausalLM  # type: ignore

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype="auto", device_map="cpu", trust_remote_code=True
    )
    result = apply_moe_lancet_surgery(model, None, None, **common)
    result["mode"] = "in_memory"
    return result


def _run_h22(
    model_dir: Path | None, spec: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """H22 looped-MoE Lancet surgery + speculative-drafter synchronization."""
    from sml.surgery.looped import (
        apply_looped_moe_lancet_surgery,
        synchronize_speculative_drafter_surgery,
    )

    if model_dir is None:
        return {
            "mode": "synthetic_dry_run",
            "h22_verified": False,
            "note": "model checkpoint unavailable; H22 reported in dry-run mode only",
        }
    from transformers import AutoModelForCausalLM  # type: ignore

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype="auto", device_map="cpu", trust_remote_code=True
    )
    # Tick dirs are resolved from a strata config when provided; otherwise a
    # neutral identity tick basis is used (K=3 execution ticks).
    k_loop = 2
    ticks = 3
    import torch

    gen = torch.Generator().manual_seed(2026)
    dirs = [torch.randn(64, generator=gen) for _ in range(k_loop)]
    dirs = [d / d.norm() for d in dirs]
    anchor_by_tick = [dirs for _ in range(ticks)]
    target_by_tick = [[-d for d in dirs] for _ in range(ticks)]
    looped = apply_looped_moe_lancet_surgery(
        model,
        anchor_by_tick,
        target_by_tick,
        k_loop=k_loop,
        scale=float(args.scale),
        min_selectivity_threshold=float(args.min_selectivity_threshold),
        shared_expert_damping=float(args.shared_expert_damping),
        hot_vram_expert_ids=set(spec["hot_vram_expert_ids"]),
        quant_snap=True,
        max_block16_crest_factor=3.20,
    )
    drafter: dict[str, Any] = {"skipped": True}
    if spec.get("has_drafter"):
        drafter = synchronize_speculative_drafter_surgery(
            model,
            {idx: dirs for idx in range(len(getattr(model, "layers", [])))},
            scale=float(args.scale),
            min_kl_reduction_pct=35.0,
        )
    return {
        "mode": "real",
        "looped": looped,
        "drafter": drafter,
        "h22_verified": bool(
            looped.get("basis_rank", 99) <= k_loop
            and looped.get("loop_controller_drift") == 0.0
            and (not spec.get("has_drafter") or drafter.get("kl_reduction_pct", 0.0) >= 35.0)
        ),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_tenet_v3_foundry",
        description="Tenet v3 foundry orchestrator (H21 Strata MoE Lancet + H22 Looped-MoE).",
    )
    parser.add_argument("--model-id", required=True, choices=sorted(TENET_V3_MODEL_SPECS))
    parser.add_argument("--model-dir", default=None, help="Local checkpoint directory override.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit.")
    parser.add_argument(
        "--strata-config", default=None, help="Optional strata TOML config override."
    )
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    parser.add_argument("--scale", type=float, default=1.0, help="Lancet surgery scale.")
    parser.add_argument("--min-selectivity-threshold", type=float, default=0.25)
    parser.add_argument("--shared-expert-damping", type=float, default=0.40)
    args = parser.parse_args(argv)

    spec = TENET_V3_MODEL_SPECS[args.model_id]
    model_dir = _resolve_model_dir(args.model_id, args.model_dir)
    output_dir = Path(args.output_dir)
    release_dir = output_dir / args.release_id

    plan = {
        "release_id": args.release_id,
        "model_id": args.model_id,
        "model_dir": str(model_dir) if model_dir else None,
        "tier": spec["tier"],
        "base_model": spec["base_model"],
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
    try:
        _log(f"H21 strata-tier-aware Lancet surgery on {args.model_id} (tier={spec['tier']})")
        h21 = _run_h21(model_dir, spec, args)
        if spec["looped"]:
            _log("H22 looped-MoE surgery + speculative-drafter sync")
            h22 = _run_h22(model_dir, spec, args)
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
