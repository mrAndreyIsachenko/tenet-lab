#!/usr/bin/env python3
"""H20 pilot driver: Tenet v3 + H18 Quant-Snap on non-Qwen/Meta models.

Targets:
  --target moe   -> google/gemma-4-26b-a4b-it        (Tenet-26B-Gemma4-MoE-Unslop-v3)
  --target dense -> mistralai/Mistral-Small-3.1-24B-Instruct-2503
                    (Tenet-24B-Mistral-Small-3.1-Unslop-v3)

Pipeline: inspect config -> build Lancet basis -> CKA quant recipe ->
streaming surgery (+ quant snap) -> H17 Fight Card Gate -> vLLM open evals ->
optional HF upload -> drop local weights -> cleanup HF cache ->
write foundry_summary.json (always).

--dry-run prints a JSON plan and exits 0 without touching weights or CUDA.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED_ROOT = REPO_ROOT / "results" / "tenet_materialized"
BASIS_FILENAME = "lancet_basis.safetensors"
RECIPE_FILENAME = "cka_quant_recipe.json"

TARGETS = {
    "moe": {
        "model_id": "google/gemma-4-26b-a4b-it",
        "release_id": "Tenet-26B-Gemma4-MoE-Unslop-v3",
    },
    "dense": {
        "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "release_id": "Tenet-24B-Mistral-Small-3.1-Unslop-v3",
    },
}

WEIGHT_SUFFIXES = (".safetensors", ".gguf")


def log(msg: str) -> None:
    print(f"[tenet-v3-pilot] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# defensive helpers (signatures of sml.* internals may drift between hoops)
# --------------------------------------------------------------------------- #


def filter_kwargs(fn: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by fn's signature (VAR_KEYWORD accepts all)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def call_first_working(fn: Callable, attempts: list[dict[str, Any]], label: str) -> Any:
    """Try each kwarg shape in order; a TypeError raised *inside* fn is a real
    failure, not a signature mismatch (checked via traceback depth)."""
    errors: list[str] = []
    for kwargs in attempts:
        filtered = filter_kwargs(fn, kwargs)
        try:
            return fn(**filtered)
        except TypeError as exc:
            internal = any(
                frame.filename != __file__ for frame in traceback.extract_tb(exc.__traceback__)
            )
            if internal:
                raise
            errors.append(f"{label}{sorted(filtered)}: {exc}")
    raise TypeError(f"no matching signature for {label}; tried: {' | '.join(errors)}")


def import_attr(candidates: tuple[tuple[str, str], ...]) -> Any:
    """Return the first importable (module, attr) pair, else None."""
    for modname, attr in candidates:
        try:
            module = importlib.import_module(modname)
            obj = getattr(module, attr)
        except (ImportError, AttributeError):
            continue
        if obj is not None:
            return obj
    return None


def as_tensor_dict(result: Any) -> dict[str, Any]:
    """Normalize a direction-extraction result into {key: cpu float tensor}."""
    import torch

    tensors: dict[str, Any] = {}

    def _clean(key: str) -> str:
        return "".join(c if (c.isalnum() or c in "_.-") else "_" for c in key)

    if isinstance(result, dict):
        for key, val in result.items():
            if torch.is_tensor(val) and val.numel():
                tensors[_clean(str(key))] = val.detach().to("cpu").float()
    elif torch.is_tensor(result) and result.numel():
        t = result.detach().to("cpu").float()
        if t.dim() == 1:
            tensors["lancet_direction"] = t
        else:
            tensors["basis"] = t
            for i in range(t.shape[0]):
                tensors[f"direction_{i}"] = t[i].contiguous()
    if not tensors:
        raise ValueError(f"could not interpret direction result of type {type(result)!r}")
    return tensors


# --------------------------------------------------------------------------- #
# pipeline steps
# --------------------------------------------------------------------------- #


def inspect_model(model_id: str) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # tokenizer is optional metadata
        log(f"tokenizer load failed ({exc!r}); continuing without it")
        tokenizer = None
    info = {
        "model_id": model_id,
        "architecture": (getattr(config, "architectures", None) or [None])[0],
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "torch_dtype": str(getattr(config, "torch_dtype", None) or ""),
        "has_tokenizer": tokenizer is not None,
        "tokenizer_class": type(tokenizer).__name__ if tokenizer else None,
    }
    if info["num_hidden_layers"] is None or info["hidden_size"] is None:
        raise ValueError(f"config for {model_id} lacks num_hidden_layers/hidden_size: {info}")
    log(
        f"inspected {model_id}: {info['architecture']}, L={info['num_hidden_layers']}, "
        f"d={info['hidden_size']}"
    )
    return {"config": config, "tokenizer": tokenizer, "info": info}


def build_basis(
    model_id: str, loaded: dict[str, Any], args: argparse.Namespace, out_path: Path
) -> dict[str, Any]:
    """Contrastive target/anchor directions -> lancet_basis.safetensors."""
    import torch

    from sml.surgery.probes import REASONING_ANCHOR_PAIRS, encode_contrast_pairs  # noqa: F401

    tri_fn = import_attr(
        (
            ("sml.surgery.probes", "extract_tri_anchor_directions"),
            ("sml.surgery.subspace", "extract_tri_anchor_directions"),
        )
    )
    stats_fn = import_attr(
        (
            ("sml.surgery.probes", "extract_contrastive_direction_with_stats"),
            ("sml.surgery.subspace", "extract_contrastive_direction_with_stats"),
        )
    )
    tokenizer = loaded["tokenizer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_layers = loaded["info"]["num_hidden_layers"]
    shared = {
        "model_id": model_id,
        "tokenizer": tokenizer,
        "pairs": REASONING_ANCHOR_PAIRS,
        "anchor_pairs": REASONING_ANCHOR_PAIRS,
        "beta": args.beta,
        "device": device,
        "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "num_layers": num_layers,
        "num_hidden_layers": num_layers,
        "hidden_size": loaded["info"]["hidden_size"],
        "layer": num_layers // 2,
        "layer_idx": num_layers // 2,
        "output_path": str(out_path),
    }
    result = None
    used = None
    attempts = [dict(shared), {k: v for k, v in shared.items() if k not in ("layer", "layer_idx")}]
    if tri_fn is not None:
        try:
            result = call_first_working(tri_fn, attempts, "extract_tri_anchor_directions")
            used = "extract_tri_anchor_directions"
        except Exception as exc:
            log(f"extract_tri_anchor_directions failed ({exc!r}); trying stats extractor")
    if result is None and stats_fn is not None:
        result = call_first_working(stats_fn, attempts, "extract_contrastive_direction_with_stats")
        used = "extract_contrastive_direction_with_stats"
    if result is None:
        raise RuntimeError("no direction extractor available from sml.surgery.probes")

    tensors = as_tensor_dict(result)
    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    stats = {
        "extractor": used,
        "num_directions": len(tensors),
        "norms": {k: round(float(v.norm()), 6) for k, v in tensors.items()},
        "output_path": str(out_path),
    }
    log(f"saved {out_path.name} via {used}: {stats['norms']}")
    return stats


def compact_layer_sensitivity(
    model_id: str, loaded: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Streaming-only fallback: heuristic per-layer CKA sensitivity profile."""
    import math

    n = loaded["info"]["num_hidden_layers"]
    center, width = 0.66 * (n - 1), max(n / 4.0, 1.0)
    layers = [
        {
            "index": i,
            "cka_sensitivity": round(math.exp(-((i - center) ** 2) / (2 * width**2)), 6),
            "quant_snap": args.quant_snap,
            "quant_snap_iters": args.quant_snap_iters,
        }
        for i in range(n)
    ]
    return {
        "schema": "cka_quant_recipe.compact.v1",
        "source": "streaming_compact_fallback",
        "model_id": model_id,
        "num_hidden_layers": n,
        "hidden_size": loaded["info"]["hidden_size"],
        "beta": args.beta,
        "layers": layers,
    }


def build_recipe(
    model_id: str, loaded: dict[str, Any], args: argparse.Namespace, out_path: Path
) -> tuple[dict[str, Any], str]:
    build_fn = import_attr((("sml.hub.release", "build_cka_quant_recipe"),))
    recipe: dict[str, Any] | None = None
    source = "compact_fallback"
    if build_fn is not None:
        shared = {
            "model_id": model_id,
            "config": loaded["config"],
            "tokenizer": loaded["tokenizer"],
            "output_path": str(out_path),
            "output_dir": str(out_path.parent),
            "beta": args.beta,
            "quant_snap": args.quant_snap,
            "quant_snap_iters": args.quant_snap_iters,
        }
        try:
            result = call_first_working(build_fn, [shared], "build_cka_quant_recipe")
            if isinstance(result, dict):
                recipe = result
                source = "build_cka_quant_recipe"
            elif out_path.exists():
                recipe = json.loads(out_path.read_text())
                source = "build_cka_quant_recipe(file)"
        except Exception as exc:
            log(f"build_cka_quant_recipe failed ({exc!r}); using compact layer-sensitivity JSON")
    if recipe is None:
        recipe = compact_layer_sensitivity(model_id, loaded, args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(recipe, indent=2, sort_keys=False) + "\n")
    log(f"wrote {out_path} (source={source}, layers={len(recipe.get('layers', []))})")
    return recipe, source


def run_surgery(
    target: str,
    model_id: str,
    recipe: dict[str, Any],
    recipe_path: Path,
    basis_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    from sml.surgery.subspace import (  # noqa: F401
        apply_lancet_recipe_streaming,
        apply_moe_lancet_recipe_streaming,
    )

    fn = apply_moe_lancet_recipe_streaming if target == "moe" else apply_lancet_recipe_streaming
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "model_id": model_id,
        "output_dir": str(out_dir),
        "beta": args.beta,
        "quant_snap": args.quant_snap,
        "quant_snap_iters": args.quant_snap_iters,
    }
    attempts = [
        {**base, "recipe": recipe, "basis_path": str(basis_path)},
        {**base, "recipe_path": str(recipe_path), "basis_path": str(basis_path)},
        {**base, "recipe_path": str(recipe_path)},
    ]
    result = call_first_working(fn, attempts, fn.__name__)
    log(f"streaming surgery done ({fn.__name__}) -> {out_dir}")
    report = {
        "function": fn.__name__,
        "output_dir": str(out_dir),
        "quant_snap": args.quant_snap,
        "quant_snap_iters": args.quant_snap_iters,
    }
    if isinstance(result, dict):
        report["result"] = {
            k: str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v
            for k, v in result.items()
            if isinstance(v, (str, int, float, bool, list, dict))
            or not isinstance(v, (str, int, float, bool, list, dict))
        }
    return report


def verify_router_freeze(model_id: str, out_dir: Path) -> dict[str, Any]:
    """Check MoE router tensors are bit-for-bit frozen (best effort)."""
    try:
        from safetensors import safe_open
    except ImportError:
        return {"checked": False, "reason": "safetensors not importable"}
    router_files = []
    for path in sorted(out_dir.glob("*.safetensors")):
        try:
            with safe_open(str(path), framework="pt") as f:
                if any("router" in k or k.endswith(".gate.weight") for k in f):
                    router_files.append(path)
        except Exception:
            continue
    if not router_files:
        return {"checked": False, "reason": "no router tensors in materialized output"}
    verify_fn = import_attr((("sml.hub.validator", "verify_moe_router_freeze"),))
    if verify_fn is None:
        return {
            "checked": False,
            "reason": "sml.hub.validator.verify_moe_router_freeze unavailable",
            "router_files": [str(p) for p in router_files],
        }
    result = call_first_working(
        verify_fn,
        [
            {
                "materialized_dir": str(out_dir),
                "base_model_id": model_id,
                "router_files": [str(p) for p in router_files],
            },
            {"out_dir": str(out_dir), "model_id": model_id},
        ],
        "verify_moe_router_freeze",
    )
    report = {"checked": True, "router_files": [str(p) for p in router_files]}
    if isinstance(result, dict):
        report.update(
            {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list))}
        )
    else:
        report["result"] = str(result)
    log(f"router freeze verification: {report}")
    return report


def run_fight_card_gate(
    model_id: str, naive_model_id: str, args: argparse.Namespace, release_dir: Path
) -> dict[str, Any]:
    run_gate = import_attr((("sml.evals.peer_eval", "run_peer_eval_gate"),))
    if run_gate is None:
        return {
            "status": "unavailable",
            "reason": "sml.evals.peer_eval.run_peer_eval_gate not importable",
        }
    gate_kwargs = {
        "model_id": model_id,
        "materialized_dir": model_id,
        "naive_model": naive_model_id,
        "naive_model_id": naive_model_id,
        "output_dir": str(release_dir),
        "think_end_token": args.think_end_token,
        "max_gen_toks": args.max_gen_toks,
        "batch_size": args.batch_size,
    }
    attempts = [dict(gate_kwargs), {**gate_kwargs, "model": model_id}]
    try:
        result = call_first_working(run_gate, attempts, "run_peer_eval_gate")
    except Exception as exc:
        return {"status": "failed", "error": repr(exc)}
    report: dict[str, Any] = {"status": "ok"}
    if isinstance(result, dict):
        flat = {}
        for k, v in result.items():
            flat[str(k)] = v if isinstance(v, (str, int, float, bool, list)) else str(v)
        report.update(flat)
    else:
        report["result"] = str(result)
    log(f"H17 fight card gate: {report['status']}")
    return report


def run_vllm_evals(
    model_id: str, naive_model_id: str, args: argparse.Namespace, release_dir: Path
) -> dict[str, Any]:
    eval_script = REPO_ROOT / "scripts" / "eval" / "run_vllm_open_evals.py"
    if not eval_script.exists():
        return {"status": "skipped", "reason": f"{eval_script} not found"}
    eval_dir = release_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(eval_script),
        "--model",
        model_id,
        "--suite",
        args.eval_suite,
        "--think-end-token",
        args.think_end_token,
        "--max-gen-toks",
        str(args.max_gen_toks),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(eval_dir),
    ]
    if naive_model_id:
        cmd += ["--naive-model", naive_model_id]
    log("running vLLM open evals: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    report: dict[str, Any] = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "command": cmd,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    json_files = sorted(eval_dir.glob("*.json"))
    report["eval_json"] = [str(p) for p in json_files]
    if proc.returncode == 0 and json_files:
        export_script = REPO_ROOT / "scripts" / "eval" / "export_model_index.py"
        if export_script.exists():
            exp = subprocess.run(
                [sys.executable, str(export_script)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            report["export_model_index"] = {
                "returncode": exp.returncode,
                "stderr_tail": exp.stderr[-1000:],
            }
        else:
            report["export_model_index"] = {"skipped": f"{export_script} not found"}
    return report


def upload_to_hf(materialized_dir: Path, release_id: str, namespace: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    repo_id = f"{namespace}/{release_id}"
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    except TypeError:  # older hub versions
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(materialized_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Tenet v3 release: {release_id}",
    )
    log(f"uploaded {materialized_dir} -> https://huggingface.co/{repo_id}")
    return {"status": "ok", "repo_id": repo_id, "folder": str(materialized_dir)}


def drop_heavy_weights(directories: list[Path]) -> dict[str, Any]:
    dropped, freed = [], 0
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            if path.name == BASIS_FILENAME:  # keep the small Lancet basis artifact
                continue
            size = path.stat().st_size
            path.unlink()
            dropped.append(str(path.relative_to(directory)))
            freed += size
    log(f"dropped {len(dropped)} weight file(s), freed {freed / 1e9:.2f} GB")
    return {"dropped_count": len(dropped), "freed_bytes": freed, "dropped": dropped}


def cleanup_cache(model_id: str) -> dict[str, Any]:
    fn = import_attr(
        (
            ("sml.hub.release", "cleanup_hf_model_cache"),
            ("sml.hub.cache", "cleanup_hf_model_cache"),
            ("sml.hub.validator", "cleanup_hf_model_cache"),
            ("sml.hub", "cleanup_hf_model_cache"),
        )
    )
    if fn is None:
        return {"status": "unavailable", "reason": "cleanup_hf_model_cache not found in sml.hub.*"}
    try:
        result = fn(model_id)
    except Exception as exc:
        return {"status": "failed", "error": repr(exc)}
    report: dict[str, Any] = {"status": "ok"}
    if isinstance(result, dict):
        report.update(
            {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list))}
        )
    return report


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    materialized_dir = MATERIALIZED_ROOT / args.release_id
    steps = [
        "inspect_config",
        "build_lancet_basis",
        "build_cka_quant_recipe",
        f"streaming_surgery_{args.target}",
        "verify_moe_router_freeze",
        "fight_card_gate_h17",
    ]
    if args.eval_suite != "none":
        steps += ["vllm_open_evals", "export_model_index"]
    if args.upload_to_hf:
        steps += ["upload_to_hf"]
    if args.drop_local_weights:
        steps += ["drop_local_weights"]
    if args.cleanup_cache:
        steps += ["cleanup_hf_cache"]
    steps.append("write_foundry_summary")
    return {
        "target": args.target,
        "model_id": args.model_id,
        "release_id": args.release_id,
        "quant_snap": args.quant_snap,
        "quant_snap_iters": args.quant_snap_iters,
        "beta": args.beta,
        "eval_suite": args.eval_suite,
        "naive_model": args.naive_model,
        "output_dir": str(Path(args.output_dir)),
        "artifacts_dir": str(materialized_dir),
        "upload_to_hf": args.upload_to_hf,
        "drop_local_weights": args.drop_local_weights,
        "cleanup_cache": args.cleanup_cache,
        "dry_run": True,
        "steps": steps,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H20 pilot: Tenet v3 + H18 Quant-Snap on non-Qwen/Meta models",
    )
    parser.add_argument("--target", choices=["moe", "dense"], required=True)
    parser.add_argument("--model-id", default=None, help="override target default model id")
    parser.add_argument("--release-id", default=None, help="override target default release id")
    parser.add_argument(
        "--naive-model",
        default=None,
        help="naive baseline id; defaults to --model-id so vLLM evals can still run or be skipped",
    )
    parser.add_argument("--output-dir", default="results/tenet_releases")
    parser.add_argument("--quant-snap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quant-snap-iters", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eval-suite", choices=["full", "quick", "none"], default="full")
    parser.add_argument("--think-end-token", default="</think>")
    parser.add_argument("--max-gen-toks", type=int, default=8192)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--hf-namespace", default="mr-andrey-isachenko")
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--drop-local-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    target = TARGETS[args.target]
    args.model_id = args.model_id or target["model_id"]
    args.release_id = args.release_id or target["release_id"]
    args.naive_model = args.naive_model or args.model_id
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    release_dir = output_dir / args.release_id
    materialized_dir = MATERIALIZED_ROOT / args.release_id
    summary: dict[str, Any] = {
        "release_id": args.release_id,
        "target": args.target,
        "model_id": args.model_id,
        "naive_model": args.naive_model,
        "hoop": "H20",
        "quant_snap": args.quant_snap,
        "quant_snap_iters": args.quant_snap_iters,
        "beta": args.beta,
        "eval_suite": args.eval_suite,
        "artifacts_dir": str(materialized_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if args.dry_run:
        print(json.dumps(build_plan(args), indent=2))
        return 0

    status = "ok"
    try:
        loaded = inspect_model(args.model_id)
        summary["model_info"] = loaded["info"]
        summary["lancet_basis"] = build_basis(
            args.model_id, loaded, args, materialized_dir / BASIS_FILENAME
        )
        recipe, recipe_source = build_recipe(
            args.model_id, loaded, args, materialized_dir / RECIPE_FILENAME
        )
        summary["cka_quant_recipe"] = {
            "source": recipe_source,
            "layers": len(recipe.get("layers", [])),
        }
        summary["surgery"] = run_surgery(
            args.target,
            args.model_id,
            recipe,
            materialized_dir / RECIPE_FILENAME,
            materialized_dir / BASIS_FILENAME,
            args,
            materialized_dir,
        )
        if args.target == "moe":
            summary["moe_router_freeze"] = verify_router_freeze(args.model_id, materialized_dir)
        summary["fight_card_gate"] = run_fight_card_gate(
            materialized_dir.as_posix(), args.naive_model, args, release_dir
        )
        if args.eval_suite != "none":
            summary["vllm_evals"] = run_vllm_evals(
                materialized_dir.as_posix(), args.naive_model, args, release_dir
            )
        if args.upload_to_hf:
            summary["hf_upload"] = upload_to_hf(
                materialized_dir, args.release_id, args.hf_namespace
            )
        if args.drop_local_weights:
            summary["drop_local_weights"] = drop_heavy_weights([materialized_dir, release_dir])
        if args.cleanup_cache:
            summary["hf_cache_cleanup"] = cleanup_cache(args.model_id)
    except Exception as exc:
        status = "failed"
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc(limit=12)
        log(f"FAILED: {exc!r}")
    finally:
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        summary["status"] = status
        release_dir.mkdir(parents=True, exist_ok=True)
        summary_path = release_dir / "foundry_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False, default=str) + "\n")
        log(f"wrote {summary_path} (status={status})")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
