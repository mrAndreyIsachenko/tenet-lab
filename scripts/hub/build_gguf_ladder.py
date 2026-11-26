"""Build the official Tenet v2 ``-GGUF`` companion-repo quant ladder.

Downloads a released Tenet v2 parent checkpoint (FP8), dequantizes it to a
BF16 GGUF-ready checkpoint, converts it with ``convert_hf_to_gguf.py``, then
runs ``llama-quantize`` across the full quant ladder with CKA-aware
``--tensor-type`` protection flags and uploads every quant to the official
companion repository on the Hugging Face Hub.

Smoke mode (``--smoke-only``) verifies the tensor-mapping regexes against
the freshly produced base GGUF without uploading anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
for _src_path in (_REPO_ROOT / "src",):
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

from sml.hub.model_card import (  # noqa: E402
    GGUF_LADDER_QUANTS,
    TENET_V2_GGUF_REPOS,
    generate_tenet_gguf_model_card,
    parse_tenet_model_name,
)
from sml.hub.release import render_ollama_modelfile  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from safetensors.torch import safe_open, save_file  # noqa: E402
except ImportError:  # pragma: no cover
    safe_open = None  # type: ignore[assignment]
    save_file = None  # type: ignore[assignment]

DEFAULT_HF_NAMESPACE = "mr-andrey-isachenko"

# Rank ladder: higher rank = higher fidelity. Assignments at a strictly
# higher rank than the target quant are re-applied verbatim on top of the
# ladder run being produced (CKA subspace protection).
QUANT_RANK: dict[str, int] = {
    "IQ4_XS": 1,
    "Q4_K_M": 2,
    "Q5_K_M": 3,
    "Q6_K": 4,
    "Q8_0": 5,
    "F16": 6,
    "BF16": 7,
}

BASE_MODELS: dict[str, str] = {
    "Tenet-26B-A4B-Gemma4-Unslop-v2": "google/gemma-4-26B-A4B-it",
    "Tenet-24B-Mistral3.1-Unslop-v2": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "Tenet-4B-Qwen3-Unslop-v2": "Qwen/Qwen3-4B",
    "Tenet-8B-Qwen3-Unslop-v2": "Qwen/Qwen3-8B",
    "Tenet-14B-Qwen3-Unslop-v2": "Qwen/Qwen3-14B",
    "Tenet-32B-Qwen3-Unslop-v2": "Qwen/Qwen3-32B",
    "Tenet-35B-A3B-Qwen3.6-Unslop-v3": "Qwen/Qwen3.6-35B-A3B",
    "Tenet-30B-A3B-GLM4.7-Unslop-v3": "zai-org/GLM-4.7-Flash",
}

_PROTECTED_TENSORS: tuple[str, ...] = ("attn_output", "ffn_down")
_SCALE_SUFFIXES: tuple[str, ...] = (".weight_scale", ".weight_scale_inv")
_FLOAT8_DTYPES = ("f8_e4m3", "f8_e5m2", "float8_e4m3fn", "float8_e5m2", "float8")


def resolve_gguf_repo_spec(
    model_id: str, hf_namespace: str = DEFAULT_HF_NAMESPACE
) -> dict[str, Any]:
    """Resolve the official GGUF companion-repo spec for a Tenet v2 model id."""
    if model_id not in TENET_V2_GGUF_REPOS:
        known = ", ".join(TENET_V2_GGUF_REPOS)
        raise ValueError(f"no official GGUF repo registered for {model_id!r}; known: {known}")
    gguf_repo_name = TENET_V2_GGUF_REPOS[model_id]
    return {
        "model_id": model_id,
        "parent_repo_id": f"{hf_namespace}/{model_id}",
        "gguf_repo_name": gguf_repo_name,
        "gguf_repo_id": f"{hf_namespace}/{gguf_repo_name}",
        "file_stem": gguf_repo_name.removesuffix("-GGUF"),
        "base_model": BASE_MODELS[model_id],
        "is_moe": bool(parse_tenet_model_name(model_id)["is_moe"]),
    }


def build_tensor_type_args(
    cka_recipe: dict[str, Any], target_quant: str, *, is_moe: bool = False
) -> list[str]:
    """Build llama-quantize ``--tensor-type`` args for one ladder run.

    Only assignments whose quant rank is strictly higher than the target
    run's rank are applied: lower-rank protections would be overwritten by
    the target quant anyway, and same-rank flags are redundant.
    """
    target_rank = QUANT_RANK.get(target_quant.upper(), 5)
    args: list[str] = []

    assignments = cka_recipe.get("layer_assignments")
    if not isinstance(assignments, list) or not assignments:
        assignments = []
        for key, quant in (("q8_0_layers", "Q8_0"), ("q6_k_layers", "Q6_K")):
            for layer in cka_recipe.get(key) or []:
                assignments.append(
                    {"layer": layer, "quant": quant, "tensors": list(_PROTECTED_TENSORS)}
                )

    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        quant = assignment.get("quant")
        if QUANT_RANK.get(str(quant).upper(), 0) <= target_rank:
            continue
        try:
            layer = int(assignment.get("layer"))
        except (TypeError, ValueError):
            continue
        for tensor in assignment.get("tensors") or []:
            if is_moe and tensor == "ffn_down":
                pattern = rf"blk\.{layer}\.ffn_down(_exps|_shexp)?\.weight={quant}"
            else:
                pattern = rf"blk\.{layer}\.{tensor}\.weight={quant}"
            args.extend(["--tensor-type", pattern])

    if is_moe:
        # Router gate always stays F16: quantized expert-routing gates
        # collapse the top-2 expert selection.
        args.extend(["--tensor-type", "ffn_gate_inp\\.weight=F16"])
        if target_rank < QUANT_RANK["Q8_0"]:
            args.extend(
                [
                    "--tensor-type",
                    "ffn_down_shexp\\.weight=Q8_0",
                    "--tensor-type",
                    "shared_expert.*=Q8_0",
                ]
            )
    return args


def _is_fp8_tensor(name: str, dtype: str) -> bool:
    lowered = f"{name} {dtype}".lower()
    return any(token in lowered for token in _FLOAT8_DTYPES)


def _scale_key_for(name: str, available: set[str]) -> str | None:
    candidates = [name]
    if name.endswith(".weight"):
        candidates.append(name.removesuffix(".weight"))
    for base in candidates:
        for suffix in (*_SCALE_SUFFIXES, "_scale"):
            candidate = f"{base}{suffix}"
            if candidate in available:
                return candidate
    return None


def _shard_is_fp8(shard_path: Path) -> bool:
    """Inspect one safetensors shard for float8 weights or scale tensors."""
    if safe_open is None:
        return False
    with safe_open(str(shard_path), framework="pt") as handle:
        for name in handle.keys():  # noqa: SIM118
            tensor = handle.get_slice(name)
            dtype = str(getattr(tensor, "get_dtype", lambda: "")())
            # Gemma-4 MoE checkpoints carry BF16 `router.per_expert_scale`
            # and `router.scale` tensors; never mistake them for FP8 scales.
            if _is_fp8_tensor(name, dtype):
                return True
            if name.endswith((".weight_scale", ".weight_scale_inv")):
                return True
    return False


def prepare_checkpoint_for_gguf(src_dir: Path, dst_dir: Path) -> Path:
    """Return a GGUF-ready checkpoint dir, dequantizing FP8 shards if needed.

    FP8 (E4M3) checkpoints cannot feed ``convert_hf_to_gguf.py`` directly.
    When ``src_dir`` carries a ``quantization_config`` or float8 / scale
    tensors, copies config/tokenizer files into ``dst_dir`` (stripping
    ``quantization_config``), dequantizes every float8 tensor to bfloat16
    via its matching scale tensor, and rebuilds the shard index. Otherwise
    ``src_dir`` is returned untouched.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    config_path = src_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    shards = sorted(
        p for p in src_dir.glob("*.safetensors") if p.name != "lancet_basis.safetensors"
    )
    has_quant_config = "quantization_config" in config
    has_fp8_shard = any(_shard_is_fp8(shard) for shard in shards)
    # Mistral checkpoints ship both `tekken.json` and a HuggingFace
    # `tokenizer.json`; llama.cpp must pick the tekken vocab
    # (`_set_vocab_mistral()`), so the conflicting tokenizer.json is excluded.
    has_tekken_conflict = (src_dir / "tekken.json").is_file() and (
        src_dir / "tokenizer.json"
    ).is_file()
    if not has_quant_config and not has_fp8_shard:
        if not has_tekken_conflict:
            return src_dir
        shutil.rmtree(dst_dir, ignore_errors=True)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(src_dir.iterdir()):
            if path.name == "tokenizer.json":
                continue
            (dst_dir / path.name).symlink_to(path.resolve())
        return dst_dir

    dst_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {"model.safetensors.index.json"}
    if has_tekken_conflict:
        skip_names.add("tokenizer.json")
    for path in sorted(src_dir.iterdir()):
        if path.is_dir() or path.suffix == ".safetensors" or path.name in skip_names:
            continue
        shutil.copy2(path, dst_dir / path.name)

    if config_path.is_file():
        config.pop("quantization_config", None)
        (dst_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    index_entries: dict[str, str] = {}
    for shard in shards:
        out_tensors: dict[str, Any] = {}
        with safe_open(str(shard), framework="pt") as handle:
            names = list(handle.keys())
            available = set(names)
            # Map each FP8 tensor to its matching scale key (checking both
            # `name + suffix` and `name.removesuffix(".weight") + suffix`).
            scale_map: dict[str, str] = {}
            for name in names:
                tensor_slice = handle.get_slice(name)
                dtype = str(getattr(tensor_slice, "get_dtype", lambda: "")())
                if _is_fp8_tensor(name, dtype):
                    scale_name = _scale_key_for(name, available)
                    if scale_name is not None:
                        scale_map[name] = scale_name
            consumed = set(scale_map.values())
            for name in names:
                tensor = handle.get_tensor(name)
                if name in scale_map:
                    scale = handle.get_tensor(scale_map[name]).float()
                    out_tensors[name] = (tensor.float() * scale).to(torch.bfloat16)
                elif name in consumed:
                    continue  # scale tensors consumed by their FP8 weight
                elif tensor.is_floating_point():
                    out_tensors[name] = tensor.to(torch.bfloat16)
                else:
                    out_tensors[name] = tensor
        out_name = f"model-{shard.stem}-bf16.safetensors"
        save_file(out_tensors, str(dst_dir / out_name))
        index_entries.update({name: out_name for name in out_tensors})

    if len(shards) > 1:
        index = {
            "metadata": {"total_size": 0},
            "weight_map": index_entries,
        }
        (dst_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
    return dst_dir


def read_gguf_tensor_names(gguf_path: Path) -> set[str]:
    """Read tensor names from a GGUF file (gguf-py when available, else raw).

    The raw fallback parses the GGUF v2/v3 header with ``struct``:
    magic ``GGUF``, version, tensor_count, kv_count, then skips the typed
    metadata key-value pairs before reading tensor name strings.
    """
    try:
        from gguf import GGUFReader

        return {t.name for t in GGUFReader(str(gguf_path)).tensors}
    except ImportError:
        pass

    _GGUF_TYPES_WITH_VALUE = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "?",
        8: "<Q",
        9: "<q",
        10: "<d",
    }

    with open(gguf_path, "rb") as handle:
        magic = handle.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {gguf_path}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", handle.read(20))

        def _skip_string() -> None:
            (length,) = struct.unpack("<Q", handle.read(8))
            handle.seek(length, os.SEEK_CUR)

        for _ in range(kv_count):
            _skip_string()
            (key_type,) = struct.unpack("<I", handle.read(4))
            if key_type == 8:  # string
                _skip_string()
            elif key_type == 11:  # array
                (value_type, count) = struct.unpack("<IQ", handle.read(12))
                if value_type == 8:  # array of strings
                    for _ in range(count):
                        _skip_string()
                elif value_type == 7:  # array of bools
                    handle.seek(count, os.SEEK_CUR)
                elif value_type in _GGUF_TYPES_WITH_VALUE:
                    handle.seek(
                        struct.calcsize(_GGUF_TYPES_WITH_VALUE[value_type]) * count, os.SEEK_CUR
                    )
                else:  # nested arrays: not worth descending, bail loudly
                    raise ValueError(f"unsupported nested GGUF kv type {value_type}")
            elif key_type == 7:
                handle.seek(1, os.SEEK_CUR)
            elif key_type in _GGUF_TYPES_WITH_VALUE:
                handle.seek(struct.calcsize(_GGUF_TYPES_WITH_VALUE[key_type]), os.SEEK_CUR)
            else:
                raise ValueError(f"unsupported GGUF kv type {key_type}")

        names: set[str] = set()
        for _ in range(tensor_count):
            names.add(_read_string(handle))
            (n_dims,) = struct.unpack("<I", handle.read(4))
            handle.seek(8 * n_dims + 4 + 8, os.SEEK_CUR)  # dims + type + offset
    return names


def _read_string(handle: Any) -> str:
    (length,) = struct.unpack("<Q", handle.read(8))
    return handle.read(length).decode("utf-8")


def smoke_check_tensor_mapping(
    base_gguf_path: Path, cka_recipe: dict[str, Any], *, is_moe: bool = False
) -> dict[str, Any]:
    """Verify the CKA ``--tensor-type`` regexes match the base GGUF tensors.

    Raises ``RuntimeError`` when no protected layer matches, when any
    selected layer has zero matched surgical tensors, or when a MoE model
    has zero router-gate matches.
    """
    tensor_names = read_gguf_tensor_names(base_gguf_path)
    args = build_tensor_type_args(cka_recipe, "Q4_K_M", is_moe=is_moe)
    patterns = [args[i + 1].split("=", 1)[0] for i in range(0, len(args), 2)]
    compiled = [re.compile(pattern) for pattern in patterns]

    selected_layers = set(cka_recipe.get("selected_layers") or [])
    matched_layers: set[int] = set()
    matched_tensors = 0
    router_matches = 0
    for name in tensor_names:
        if any(pattern.search(name) for pattern in compiled):
            matched_tensors += 1
        layer_match = re.search(r"blk\.(\d+)\.", name)
        if (
            layer_match
            and int(layer_match.group(1)) in selected_layers
            and any(pattern.search(name) for pattern in compiled)
        ):
            matched_layers.add(int(layer_match.group(1)))
        if is_moe and "ffn_gate_inp" in name:
            router_matches += 1

    if not matched_layers:
        sample = ", ".join(sorted(tensor_names)[:8])
        raise RuntimeError(
            "tensor-type patterns matched no selected layers in "
            f"{base_gguf_path.name} (sample tensors: {sample})"
        )
    missing = sorted(selected_layers - matched_layers)
    if missing:
        sample = ", ".join(
            sorted(name for name in tensor_names if f"blk.{missing[0]}." in name)[:8]
        )
        raise RuntimeError(
            f"layers with zero matched surgical tensors: {missing} (sample: {sample})"
        )
    if is_moe and router_matches == 0:
        sample = ", ".join(sorted(tensor_names)[:8])
        raise RuntimeError(f"no ffn_gate_inp router tensors matched (sample: {sample})")

    return {
        "total_tensors": len(tensor_names),
        "matched_tensors": matched_tensors,
        "matched_layers": len(matched_layers),
        "router_matches": router_matches,
    }


def _load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = _REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_card_metrics(release_dir: Path, external: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    arms = external.get("arms") if isinstance(external.get("arms"), dict) else None
    payload = arms.get("tenet", {}) if arms is not None else external.get("after", external)
    if isinstance(payload, dict):
        metrics.update(payload)
    # Preserve the full external-benchmarks payload and the Tenet arm scores
    # so `_open_eval_values(metrics)` sees every OpenEvals metric.
    if isinstance(external.get("external_benchmarks"), dict):
        metrics["external_benchmarks"] = external["external_benchmarks"]
    if arms is not None and isinstance(arms.get("tenet", {}).get("scores"), dict):
        metrics["scores"] = arms["tenet"]["scores"]
    return metrics


def _convert_to_base_gguf(
    ckpt_dir: Path, base_gguf: Path, llama_cpp_dir: Path, outtype: str
) -> Path:
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    # Convert into a temp file so an interrupted run never leaves a truncated
    # base GGUF that a later invocation would treat as reusable.
    tmp_gguf = base_gguf.with_suffix(".tmp.gguf")
    tmp_gguf.unlink(missing_ok=True)
    for resolved_outtype in (outtype, "f16"):
        cmd = [
            sys.executable,
            str(converter),
            str(ckpt_dir),
            "--outfile",
            str(tmp_gguf),
            "--outtype",
            resolved_outtype,
        ]
        print(f"[convert] {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            tmp_gguf.replace(base_gguf)
            return base_gguf
        tmp_gguf.unlink(missing_ok=True)
        print(f"[convert] --outtype {resolved_outtype} failed; trying fallback")
    raise RuntimeError(f"convert_hf_to_gguf.py failed for {ckpt_dir}")


def _process_model(spec: dict[str, Any], args: argparse.Namespace, token: str | None) -> None:
    model_id = spec["model_id"]
    work_dir = Path(args.work_dir) / model_id
    raw_dir = work_dir / "hf_snapshot"
    ckpt_dir = work_dir / "ckpt"
    work_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, create_repo, snapshot_download

    base_gguf = work_dir / f"{spec['file_stem']}-BF16.gguf"
    if base_gguf.is_file():
        print(f"[reuse] {base_gguf}")
    else:
        if (raw_dir / "config.json").is_file() and any(
            p.name != "lancet_basis.safetensors" and p.suffix == ".safetensors"
            for p in raw_dir.glob("*.safetensors")
        ):
            print(f"[reuse-snapshot] {raw_dir}")
        else:
            print(f"[download] {spec['parent_repo_id']} -> {raw_dir}")
            snapshot_download(
                repo_id=spec["parent_repo_id"],
                local_dir=str(raw_dir),
                token=token,
                ignore_patterns=["*.gguf"],
            )
        ckpt = prepare_checkpoint_for_gguf(raw_dir, ckpt_dir)
        _convert_to_base_gguf(ckpt, base_gguf, Path(args.llama_cpp_dir), args.outtype)

        if not args.keep_intermediate:
            # Free ~50 GB of safetensors before quantization begins.
            shutil.rmtree(raw_dir, ignore_errors=True)
            if ckpt_dir.exists() and ckpt_dir != raw_dir:
                shutil.rmtree(ckpt_dir, ignore_errors=True)

    release_dir = Path(_REPO_ROOT) / "results" / "tenet_releases" / model_id
    cka_recipe = _read_json_if_present(release_dir / "cka_quant_recipe.json")
    if not cka_recipe:
        raise RuntimeError(f"missing {release_dir / 'cka_quant_recipe.json'}")
    external = _read_json_if_present(release_dir / "external_benchmarks.json")
    metrics = _load_card_metrics(release_dir, external)

    smoke = smoke_check_tensor_mapping(base_gguf, cka_recipe, is_moe=spec["is_moe"])
    print(f"[smoke] {model_id}: {smoke}")
    if args.smoke_only:
        return

    api = HfApi(token=token)
    if args.upload:
        create_repo(spec["gguf_repo_id"], repo_type="model", exist_ok=True, token=token)

    recipe_path = release_dir / "cka_quant_recipe.json"
    modelfile_path = work_dir / "Modelfile"
    modelfile_path.write_text(
        render_ollama_modelfile(
            spec["gguf_repo_name"], cka_recipe, gguf_filename=f"{spec['file_stem']}-Q4_K_M.gguf"
        ),
        encoding="utf-8",
    )
    quant_files: dict[str, dict[str, Any]] = {}

    def _upload_static_files() -> None:
        if not args.upload:
            return
        api.upload_file(
            path_or_fileobj=str(recipe_path),
            path_in_repo="cka_quant_recipe.json",
            repo_id=spec["gguf_repo_id"],
            token=token,
        )
        api.upload_file(
            path_or_fileobj=str(modelfile_path),
            path_in_repo="Modelfile",
            repo_id=spec["gguf_repo_id"],
            token=token,
        )
        readme = generate_tenet_gguf_model_card(
            model_id,
            spec["base_model"],
            gguf_repo_name=spec["gguf_repo_name"],
            hf_namespace=args.hf_namespace,
            quant_files=quant_files,
            cka_recipe=cka_recipe,
            metrics=metrics,
            is_moe=spec["is_moe"],
        )
        readme_path = work_dir / "README.md"
        readme_path.write_text(readme, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=spec["gguf_repo_id"],
            token=token,
        )

    _upload_static_files()

    quantize_bin = Path(args.llama_cpp_dir) / "build" / "bin" / "llama-quantize"
    for quant in args.quants.split(","):
        quant = quant.strip()
        if not quant:
            continue
        out_gguf = work_dir / f"{spec['file_stem']}-{quant}.gguf"
        cmd = [
            str(quantize_bin),
            *build_tensor_type_args(cka_recipe, quant, is_moe=spec["is_moe"]),
            str(base_gguf),
            str(out_gguf),
            quant,
            str(args.threads),
        ]
        print(f"[quantize] {quant}")
        subprocess.run(cmd, check=True)
        quant_files[quant] = {"size": f"{out_gguf.stat().st_size / 1e9:.2f} GB"}
        if args.upload:
            api.upload_file(
                path_or_fileobj=str(out_gguf),
                path_in_repo=out_gguf.name,
                repo_id=spec["gguf_repo_id"],
                token=token,
            )
        if not args.keep_intermediate:
            out_gguf.unlink(missing_ok=True)

    _upload_static_files()  # refresh README with actual file sizes
    if not args.keep_intermediate:
        base_gguf.unlink(missing_ok=True)


def main() -> None:
    """CLI entry point for the Tenet v2 GGUF ladder builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        action="append",
        required=True,
        help="Tenet v2 model id (comma-separated or repeated flag)",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/nvme/gguf_work"))
    parser.add_argument("--llama-cpp-dir", type=Path, default=Path("/mnt/nvme/llama.cpp"))
    parser.add_argument("--quants", default=",".join(GGUF_LADDER_QUANTS))
    parser.add_argument("--outtype", default="bf16")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--hf-namespace", default=DEFAULT_HF_NAMESPACE)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_ids: list[str] = []
    for value in args.model_id:
        model_ids.extend(part.strip() for part in value.split(",") if part.strip())

    token = _load_hf_token()
    if args.upload and not token:
        raise SystemExit("--upload requires HF_TOKEN in the environment or .env")
    if args.dry_run:
        for model_id in model_ids:
            print(json.dumps(resolve_gguf_repo_spec(model_id, args.hf_namespace), indent=2))
        return

    for model_id in model_ids:
        spec = resolve_gguf_repo_spec(model_id, args.hf_namespace)
        _process_model(spec, args, token)


if __name__ == "__main__":
    main()
