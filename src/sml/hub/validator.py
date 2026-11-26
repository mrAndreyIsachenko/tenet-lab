"""Provider readiness validation for exported Tenet checkpoints.

A checkpoint is "provider ready" when it can be published to hosted
inference providers (vLLM / OpenRouter / DeepInfra / Novita AI / Chutes /
Featherless) without special handling:

* a standard, first-party architecture (no custom ``auto_map`` classes),
* no ``trust_remote_code`` escape hatches anywhere,
* a usable chat template,
* clean weights (no unmerged LoRA adapters, no NaN/Inf values, and no
  weight magnitudes that would break FP8 quantization).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

try:
    from safetensors.torch import load_file as _safetensors_load_file
except ImportError:  # pragma: no cover
    _safetensors_load_file = None  # type: ignore[assignment]

STANDARD_ARCHITECTURES = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "MistralForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3MoeForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Phi3ForCausalLM",
    "MixtralForCausalLM",
}

# v3 MoE Tenet checkpoints ship packed/fused expert weights under
# first-party Gemma4 / Mistral3.1 architectures; they are supported
# alongside the standard set but are not part of the v1 baseline.
MOE_V3_ARCHITECTURES = {
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3_6MoeForCausalLM",
    "Qwen3_8MoeForCausalLM",
    "Glm4MoeForCausalLM",
    "Glm4MoeLiteForCausalLM",
    "Inkling2ForCausalLM",
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
    "Hunyuan4MoeForCausalLM",
    "HYV4FlashForConditionalGeneration",
    "Hy4MoeForCausalLM",
}

# Canonical Tenet v3 lineup (6 MoE models) whose config architectures
# above are validated by ``verify_provider_readiness``.
TENET_V3_MODEL_NAMES = (
    # Legacy backward-compat entries.
    "Tenet-15B-A2B-Qwen3.6-Unslop-v3",
    "Tenet-32B-A3B-GLM4.5-Unslop-v3",
    # Updated v3 lineup.
    "Tenet-30B-A3B-GLM4.7-Unslop-v3",
    "Tenet-15B-A2B-Inkling2-Unslop-v3",
    "Tenet-26B-A3B-Inkling2-Unslop-v3",
    "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
    "Tenet-106B-A12B-GLM4.5-Unslop-v3",
    "Tenet-125B-A6B-Qwen3.8-Unslop-v3",
    "Tenet-220B-A12B-Hy4-Unslop-v3",
)

SUPPORTED_ARCHITECTURES = STANDARD_ARCHITECTURES | MOE_V3_ARCHITECTURES

MAX_ABS_WEIGHT_THRESHOLD = 50.0
# Pretrained Qwen2.5 down_proj rows reach a crest factor of ~49 and
# Mistral-Nemo-12B reaches ~96, so the threshold must sit above real-model
# values while still catching spikes.
MAX_ROW_CREST_FACTOR_THRESHOLD = 128.0

RECIPE_FILENAME = "lancet_recipe.json"

# Weight-key layer prefixes used by multimodal / nested HF configs, most
# specific first. ``strip_layer_prefix`` tries each in order.
WEIGHT_LAYER_PREFIXES: tuple[str, ...] = (
    "model.language_model.layers.",
    "language_model.model.layers.",
    "language_model.layers.",
    "model.llm.layers.",
    "model.layers.",
    "layers.",
)


def resolve_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the nested ``text_config`` when present, else ``config`` itself."""
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config


def strip_layer_prefix(key: str) -> tuple[int, str] | None:
    """Split a weight key into ``(layer_idx, suffix)`` under any known prefix.

    Returns ``None`` when the key does not match a known layer prefix.
    """
    for prefix in WEIGHT_LAYER_PREFIXES:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        idx, sep, suffix = remainder.partition(".")
        if sep and idx.isdigit():
            return int(idx), suffix
    return None


RECIPE_BASIS_FILENAME = "lancet_basis.safetensors"
RECIPE_SCHEMA_VERSION = "1.0"
RECIPE_BASIS_NORM_TOLERANCE = 1e-4

_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt")

CHECK_NAMES = (
    "standard_architecture",
    "no_trust_remote_code",
    "valid_chat_template",
    "no_nan_inf",
    "no_unmerged_lora",
    "weight_outliers_ok",
    "acceptable_row_crest_factor",
)


def verify_provider_readiness(checkpoint_dir: Path | str) -> dict[str, Any]:
    """Validate that ``checkpoint_dir`` is ready for provider deployment.

    Returns a dict with ``ready``, ``errors``, ``checks``, ``architectures``,
    ``max_abs_weight``, ``max_row_crest_factor``, ``weight_files_checked``,
    ``row_crest_factor_p50``, ``row_crest_factor_p99`` and
    ``top_3_outlier_layers`` keys.
    """
    checkpoint = Path(checkpoint_dir)
    checks: dict[str, bool] = {name: True for name in CHECK_NAMES}
    errors: list[str] = []
    architectures: list[str] = []

    config, config_ok = _load_json_object(checkpoint / "config.json", errors, "config.json")
    if config_ok:
        _check_architecture(config, checks, errors, architectures)
        _check_no_trust_remote_code(config, checks, errors)
    else:
        checks["standard_architecture"] = False
        checks["no_trust_remote_code"] = False

    tokenizer_config, tokenizer_ok = _load_json_object(
        checkpoint / "tokenizer_config.json", errors, "tokenizer_config.json"
    )
    if tokenizer_ok:
        _check_chat_template(tokenizer_config, checks, errors, checkpoint_dir=checkpoint)
    else:
        checks["valid_chat_template"] = False

    if (checkpoint / RECIPE_FILENAME).is_file():
        prov = verify_recipe_provenance(checkpoint)
        checks["recipe_provenance"] = prov["valid"]
        if not prov["valid"]:
            errors.extend(prov["errors"])

    weight_files = sorted(
        path
        for pattern in _WEIGHT_GLOBS
        for path in checkpoint.glob(pattern)
        if path.is_file() and path.name != RECIPE_BASIS_FILENAME
    )
    weight_stats = _inspect_weights(weight_files, checks, errors)
    max_abs_weight = weight_stats["max_abs_weight"]
    max_row_crest_factor = weight_stats["max_row_crest_factor"]
    checks["weight_outliers_ok"] = max_abs_weight < MAX_ABS_WEIGHT_THRESHOLD
    if not checks["weight_outliers_ok"]:
        errors.append(
            f"weights: max |weight| {max_abs_weight:.3f} exceeds FP8 outlier "
            f"threshold {MAX_ABS_WEIGHT_THRESHOLD}"
        )
    checks["acceptable_row_crest_factor"] = max_row_crest_factor < MAX_ROW_CREST_FACTOR_THRESHOLD
    if not checks["acceptable_row_crest_factor"]:
        errors.append(
            f"weights: max row crest factor {max_row_crest_factor:.3f} exceeds "
            f"threshold {MAX_ROW_CREST_FACTOR_THRESHOLD}"
        )

    return {
        "ready": all(checks.values()),
        "errors": errors,
        "checks": checks,
        "architectures": architectures,
        "max_abs_weight": max_abs_weight,
        "max_row_crest_factor": max_row_crest_factor,
        "weight_files_checked": len(weight_files),
        "row_crest_factor_p50": weight_stats["row_crest_factor_p50"],
        "row_crest_factor_p99": weight_stats["row_crest_factor_p99"],
        "top_3_outlier_layers": weight_stats["top_3_outlier_layers"],
    }


def verify_recipe_provenance(checkpoint_dir: Path | str) -> dict[str, Any]:
    """Verify the Lancet recipe artifact pair shipped with a checkpoint.

    Checks that ``lancet_recipe.json`` and ``lancet_basis.safetensors``
    exist and parse cleanly, that the recipe fields are internally
    consistent (schema version, dimensions, selected layers, per-layer
    weights), that the basis artifact carries a ``(k, d_model)`` tensor
    with finite, unit-norm rows for every selected layer, and that
    ``config.json`` dimensions (when present) match the recipe.

    Returns ``{'valid': bool, 'errors': list[str], 'recipe': dict | None}``.
    """
    checkpoint = Path(checkpoint_dir)
    errors: list[str] = []
    recipe: dict[str, Any] | None = None

    recipe_path = checkpoint / RECIPE_FILENAME
    basis_path = checkpoint / RECIPE_BASIS_FILENAME
    if not recipe_path.is_file():
        errors.append(f"{RECIPE_FILENAME} not found in checkpoint directory")
    if not basis_path.is_file():
        errors.append(f"{RECIPE_BASIS_FILENAME} not found in checkpoint directory")

    dimensions: tuple[int, int, list[int]] | None = None
    if recipe_path.is_file():
        recipe = _parse_recipe(recipe_path, errors)
        if recipe is not None:
            dimensions = _check_recipe_fields(recipe, errors)

    if basis_path.is_file() and dimensions is not None:
        _check_basis_tensors(basis_path, dimensions, errors)

    config_path = checkpoint / "config.json"
    if config_path.is_file() and dimensions is not None:
        _check_config_dimensions(config_path, dimensions, errors)

    return {"valid": not errors, "errors": errors, "recipe": recipe}


def verify_moe_router_freeze(
    baseline: Path | str | dict[str, torch.Tensor] | nn.Module,
    edited: Path | str | dict[str, torch.Tensor] | nn.Module,
) -> dict[str, Any]:
    """Verify that MoE router/gate weights are bit-identical after surgery.

    Router (gate) weights must stay frozen during expert-weight surgery so
    the trained routing behaviour is preserved exactly. Both ``baseline``
    and ``edited`` may be a plain state dict, an ``nn.Module``, or a
    ``Path``/``str`` pointing at a checkpoint directory of
    ``*.safetensors`` files (or a single safetensors file).

    Every key accepted by ``is_router_weight_key`` (from ``sml.surgery.cka``)
    that is present in both state dicts is compared; the result reports the
    maximum absolute difference across all router tensors.

    Returns ``{'router_weight_drift': float, 'router_freeze_ok': bool,
    'checked_router_tensors': int, 'drifted_keys': list[str]}`` where
    ``router_freeze_ok`` is ``True`` only when the drift is exactly ``0.0``.
    """
    from sml.surgery.cka import is_router_weight_key

    if torch is None:  # pragma: no cover
        raise ImportError("torch is required to verify MoE router freeze")

    baseline_state = _extract_state_dict(baseline)
    edited_state = _extract_state_dict(edited)

    shared_router_keys = [
        key
        for key in baseline_state
        if key in edited_state
        and is_router_weight_key(key)
        and isinstance(baseline_state[key], torch.Tensor)
        and isinstance(edited_state[key], torch.Tensor)
    ]

    router_weight_drift = 0.0
    drifted_keys: list[str] = []
    for key in shared_router_keys:
        base_tensor = baseline_state[key].detach().to(dtype=torch.float32)
        edited_tensor = edited_state[key].detach().to(dtype=torch.float32)
        if base_tensor.shape != edited_tensor.shape:
            drift = float("inf")
        else:
            drift = float((edited_tensor - base_tensor).abs().max().item())
        if drift != drift:  # NaN drift propagates as an infinite drift
            drift = float("inf")
        router_weight_drift = max(router_weight_drift, drift)
        if drift != 0.0:
            drifted_keys.append(key)

    return {
        "router_weight_drift": router_weight_drift,
        "router_freeze_ok": router_weight_drift == 0.0,
        "checked_router_tensors": len(shared_router_keys),
        "drifted_keys": drifted_keys,
    }


def _extract_state_dict(source: Path | str | dict[str, Any] | Any) -> dict[str, Any]:
    """Return a flat ``{key: tensor}`` state dict for a supported source."""
    if isinstance(source, dict):
        return dict(source)
    if nn is not None and isinstance(source, nn.Module):
        return dict(source.state_dict().items())
    path = Path(source)
    if _safetensors_load_file is None:  # pragma: no cover
        raise ImportError("safetensors is required to load router weights from disk")
    if path.is_dir():
        state: dict[str, Any] = {}
        for weight_file in sorted(path.glob("*.safetensors")):
            state.update(_safetensors_load_file(str(weight_file), device="cpu"))
        return state
    if path.is_file():
        return _safetensors_load_file(str(path), device="cpu")
    raise FileNotFoundError(f"no checkpoint directory or safetensors file at {path}")


def _parse_recipe(path: Path, errors: list[str]) -> dict[str, Any] | None:
    """Read ``lancet_recipe.json`` as a JSON object, recording failures."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{RECIPE_FILENAME} could not be parsed: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{RECIPE_FILENAME} must contain a JSON object")
        return None
    return payload


def _check_recipe_fields(
    recipe: dict[str, Any],
    errors: list[str],
) -> tuple[int, int, list[int]] | None:
    """Validate recipe fields; return ``(d_model, num_layers, selected_layers)``.

    Returns ``None`` when the core dimension/layer fields are invalid.
    """
    if recipe.get("schema_version") != RECIPE_SCHEMA_VERSION:
        errors.append(
            f"{RECIPE_FILENAME}: 'schema_version' must be "
            f"'{RECIPE_SCHEMA_VERSION}' (got {recipe.get('schema_version')!r})"
        )

    d_model = recipe.get("d_model")
    if not _is_positive_int(d_model):
        errors.append(f"{RECIPE_FILENAME}: 'd_model' must be a positive integer")
        d_model = None

    num_layers = recipe.get("num_layers")
    if not _is_positive_int(num_layers):
        errors.append(f"{RECIPE_FILENAME}: 'num_layers' must be a positive integer")
        num_layers = None

    selected = recipe.get("selected_layers")
    if not isinstance(selected, list) or not selected:
        errors.append(f"{RECIPE_FILENAME}: 'selected_layers' must be a non-empty list")
        selected = None
    elif num_layers is None:
        errors.append(
            f"{RECIPE_FILENAME}: 'selected_layers' cannot be range-checked "
            "without a valid 'num_layers'"
        )
        selected = None
    else:
        invalid = [layer for layer in selected if not _is_layer_index(layer, num_layers)]
        if invalid:
            errors.append(
                f"{RECIPE_FILENAME}: 'selected_layers' entries must be integers "
                f"in [0, {num_layers}) (invalid: {invalid})"
            )
            selected = None

    _check_layer_weights(recipe, selected, errors)

    if d_model is None or num_layers is None or selected is None:
        return None
    return d_model, num_layers, selected


def _check_layer_weights(
    recipe: dict[str, Any],
    selected: list[int] | None,
    errors: list[str],
) -> None:
    """Validate ``layer_weights`` keys against ``selected_layers``."""
    weights = recipe.get("layer_weights")
    if not isinstance(weights, dict) or not weights:
        errors.append(f"{RECIPE_FILENAME}: 'layer_weights' must be a non-empty object")
        return
    if selected is not None:
        weight_indices: set[int] = set()
        bad_keys: list[Any] = []
        for key in weights:
            try:
                weight_indices.add(int(key))
            except (TypeError, ValueError):
                bad_keys.append(key)
        if bad_keys:
            errors.append(
                f"{RECIPE_FILENAME}: 'layer_weights' keys must be layer indices "
                f"(invalid: {bad_keys})"
            )
        if weight_indices != set(selected):
            missing = sorted(set(selected) - weight_indices)
            unexpected = sorted(weight_indices - set(selected))
            errors.append(
                f"{RECIPE_FILENAME}: 'layer_weights' keys must match "
                f"'selected_layers' (missing: {missing}, unexpected: {unexpected})"
            )
    invalid_values = [(key, value) for key, value in weights.items() if not _is_weight_value(value)]
    if invalid_values:
        errors.append(
            f"{RECIPE_FILENAME}: 'layer_weights' values must be numbers in "
            f"(0, 1] (invalid: {invalid_values})"
        )


def _check_basis_tensors(
    basis_path: Path,
    dimensions: tuple[int, int, list[int]],
    errors: list[str],
) -> None:
    """Verify basis tensors exist with finite, unit-norm rows."""
    d_model, _num_layers, selected = dimensions
    if _safetensors_load_file is None or torch is None:  # pragma: no cover
        errors.append(
            f"{RECIPE_BASIS_FILENAME}: torch/safetensors unavailable, "
            "basis tensors cannot be verified"
        )
        return
    try:
        payload = _safetensors_load_file(str(basis_path), device="cpu")
    except Exception as exc:
        errors.append(f"{RECIPE_BASIS_FILENAME}: failed to load ({exc})")
        return
    if not isinstance(payload, dict):
        errors.append(f"{RECIPE_BASIS_FILENAME}: expected a flat tensor mapping")
        return
    for index in selected:
        key = f"layer_{index}"
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"{RECIPE_BASIS_FILENAME}: missing basis tensor '{key}'")
            continue
        if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] != d_model:
            errors.append(
                f"{RECIPE_BASIS_FILENAME}: tensor '{key}' must have shape "
                f"(k, {d_model}) with k >= 1 (got {tuple(tensor.shape)})"
            )
            continue
        if not tensor.is_floating_point():
            errors.append(
                f"{RECIPE_BASIS_FILENAME}: tensor '{key}' must contain floating-point values"
            )
            continue
        rows = tensor.detach().to(dtype=torch.float32)
        if bool(torch.isnan(rows).any()) or bool(torch.isinf(rows).any()):
            errors.append(f"{RECIPE_BASIS_FILENAME}: tensor '{key}' contains NaN or Inf values")
            continue
        norms = torch.linalg.vector_norm(rows, dim=1)
        if bool(((norms - 1.0).abs() > RECIPE_BASIS_NORM_TOLERANCE).any()):
            errors.append(
                f"{RECIPE_BASIS_FILENAME}: tensor '{key}' rows must have unit "
                f"L2 norm within {RECIPE_BASIS_NORM_TOLERANCE}"
            )


def _check_config_dimensions(
    config_path: Path,
    dimensions: tuple[int, int, list[int]],
    errors: list[str],
) -> None:
    """Cross-check ``config.json`` dimensions against the recipe."""
    d_model, num_layers, _selected = dimensions
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"config.json could not be parsed: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append("config.json must contain a JSON object")
        return
    text_cfg = resolve_text_config(payload)
    hidden_size = text_cfg.get("hidden_size", payload.get("hidden_size"))
    if hidden_size is not None and hidden_size != d_model:
        errors.append(
            f"config.json: 'hidden_size' ({hidden_size}) does not match recipe "
            f"'d_model' ({d_model})"
        )
    config_layers = text_cfg.get("num_hidden_layers", payload.get("num_hidden_layers"))
    if config_layers is not None and config_layers != num_layers:
        errors.append(
            f"config.json: 'num_hidden_layers' ({config_layers}) does not match "
            f"recipe 'num_layers' ({num_layers})"
        )


def _is_positive_int(value: Any) -> bool:
    """Return True for a non-bool integer greater than zero."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_layer_index(value: Any, num_layers: int) -> bool:
    """Return True for a non-bool integer layer index in ``[0, num_layers)``."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < num_layers


def _is_weight_value(value: Any) -> bool:
    """Return True for a finite numeric weight in ``(0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 < float(value) <= 1.0


def _load_json_object(
    path: Path,
    errors: list[str],
    label: str,
) -> tuple[dict[str, Any], bool]:
    """Read ``path`` as a JSON object, recording failures in ``errors``."""
    if not path.is_file():
        errors.append(f"{label} not found in checkpoint directory")
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} could not be parsed: {exc}")
        return {}, False
    if not isinstance(payload, dict):
        errors.append(f"{label} must contain a JSON object")
        return {}, False
    return payload, True


def _check_architecture(
    config: dict[str, Any],
    checks: dict[str, bool],
    errors: list[str],
    architectures: list[str],
) -> None:
    """Require a non-empty ``architectures`` list with a supported entry."""
    entries = config.get("architectures")
    if not isinstance(entries, list) or not entries:
        entries = resolve_text_config(config).get("architectures")
    if not isinstance(entries, list) or not entries:
        checks["standard_architecture"] = False
        errors.append("config.json: 'architectures' must be a non-empty list")
        return
    names = [entry for entry in entries if isinstance(entry, str)]
    architectures.extend(names)
    if not any(name in SUPPORTED_ARCHITECTURES for name in names):
        checks["standard_architecture"] = False
        expected = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        errors.append(f"config.json: no supported architecture found (expected one of {expected})")


def _check_no_trust_remote_code(
    config: dict[str, Any],
    checks: dict[str, bool],
    errors: list[str],
) -> None:
    """Reject custom ``auto_map`` code and ``trust_remote_code`` opt-ins."""
    text_cfg = resolve_text_config(config)
    has_auto_map = "auto_map" in config or (text_cfg is not config and "auto_map" in text_cfg)
    trusts_remote = bool(config.get("trust_remote_code")) or (
        text_cfg is not config and bool(text_cfg.get("trust_remote_code"))
    )
    if has_auto_map:
        checks["no_trust_remote_code"] = False
        errors.append("config.json: 'auto_map' present (remote code required)")
    if trusts_remote:
        checks["no_trust_remote_code"] = False
        errors.append("config.json: 'trust_remote_code' is enabled")


def _check_chat_template(
    tokenizer_config: dict[str, Any],
    checks: dict[str, bool],
    errors: list[str],
    *,
    checkpoint_dir: Path | None = None,
) -> None:
    """Require a usable chat template and no ``trust_remote_code``."""
    template = tokenizer_config.get("chat_template")
    if not _has_chat_template(template) and checkpoint_dir is not None:
        jinja = checkpoint_dir / "chat_template.jinja"
        try:
            jinja_text = jinja.read_text(encoding="utf-8")
        except OSError:
            jinja_text = ""
        if jinja_text.strip():
            template = jinja_text
    if not _has_chat_template(template):
        checks["valid_chat_template"] = False
        errors.append("tokenizer_config.json: 'chat_template' is missing or empty")
    if tokenizer_config.get("trust_remote_code"):
        checks["valid_chat_template"] = False
        errors.append("tokenizer_config.json: 'trust_remote_code' is enabled")


def _has_chat_template(template: Any) -> bool:
    """Return True for a non-empty string template or template mapping."""
    if isinstance(template, str):
        return bool(template.strip())
    if isinstance(template, dict):
        return len(template) > 0
    if isinstance(template, (list, tuple)):
        return len(template) > 0
    return False


def _inspect_weights(
    weight_files: list[Path],
    checks: dict[str, bool],
    errors: list[str],
) -> dict[str, Any]:
    """Scan weight tensors and summarize weight magnitudes and row crest factors.

    Returns a dict with ``max_abs_weight``, ``max_row_crest_factor``,
    ``max_block16_crest_factor``, ``row_crest_factor_p50``,
    ``row_crest_factor_p99`` and ``top_3_outlier_layers`` keys. The
    percentile diagnostics fall back to ``0.0`` (and the outlier list to
    empty) when there are no 2D floating-point weight tensors.
    """
    from sml.surgery.quantization import compute_microblock_crest_factor

    max_abs_weight = 0.0
    max_row_crest_factor = 0.0
    max_block16_crest_factor = 0.0
    crest_factors: list[float] = []
    outlier_layers: list[dict[str, Any]] = []
    for path in weight_files:
        if path.name == RECIPE_BASIS_FILENAME:
            continue  # Lancet recipe basis artifact, not model weights
        payload = _load_weights(path, errors)
        for key, tensor in _iter_tensors(payload):
            if "lora_" in key.lower():
                checks["no_unmerged_lora"] = False
                errors.append(f"{path.name}: unmerged LoRA adapter '{key}'")
            if not _is_floating_tensor(tensor):
                continue
            if bool(torch.isnan(tensor).any()) or bool(torch.isinf(tensor).any()):
                checks["no_nan_inf"] = False
                errors.append(f"{path.name}: tensor '{key}' contains NaN or Inf values")
            # Only 2-D+ weights count toward the FP8 outlier threshold; 1-D
            # biases (e.g. Qwen2.5 k_proj.bias ~= 91.5) are exempt.
            if tensor.ndim >= 2 and tensor.numel():
                magnitude = float(tensor.abs().max().item())
                max_abs_weight = max(max_abs_weight, magnitude)
            if tensor.ndim == 2 and tensor.numel() > 0 and tensor.is_floating_point():
                w = tensor.detach().to(dtype=torch.float32)
                row_abs_max = w.abs().amax(dim=1)
                row_rms = torch.sqrt((w * w).mean(dim=1) + 1e-8)
                row_crest = float((row_abs_max / row_rms).max().item())
                max_row_crest_factor = max(max_row_crest_factor, row_crest)
                block16_cf = compute_microblock_crest_factor(tensor, block_size=16)[
                    "max_block_crest_factor"
                ]
                max_block16_crest_factor = max(max_block16_crest_factor, block16_cf)
                crest_factors.append(row_crest)
                outlier_layers.append(
                    {
                        "tensor": key,
                        "crest_factor": row_crest,
                        "max_abs": float(row_abs_max.max().item()),
                    }
                )
            elif tensor.ndim == 3 and tensor.is_floating_point():
                # Packed/fused MoE expert tensors stack one 2-D expert
                # matrix per leading index; inspect each expert slice
                # individually instead of skipping ndim != 2 so per-expert
                # outliers and NaN/Inf do not hide inside the packed dim.
                for expert in range(tensor.shape[0]):
                    expert_slice = tensor[expert]
                    if not expert_slice.is_floating_point():
                        continue
                    if bool(torch.isnan(expert_slice).any()) or bool(
                        torch.isinf(expert_slice).any()
                    ):
                        checks["no_nan_inf"] = False
                        errors.append(
                            f"{path.name}: tensor '{key}'[{expert}] contains NaN or Inf values"
                        )
                    if expert_slice.numel():
                        magnitude = float(expert_slice.abs().max().item())
                        max_abs_weight = max(max_abs_weight, magnitude)
                    if expert_slice.numel() > 0:
                        block16_cf = compute_microblock_crest_factor(expert_slice, block_size=16)[
                            "max_block_crest_factor"
                        ]
                        max_block16_crest_factor = max(max_block16_crest_factor, block16_cf)
    if crest_factors:
        distribution = torch.tensor(crest_factors, dtype=torch.float32)
        row_crest_p50 = float(torch.quantile(distribution, 0.5).item())
        row_crest_p99 = float(torch.quantile(distribution, 0.99).item())
    else:
        row_crest_p50 = 0.0
        row_crest_p99 = 0.0
    top_3_outlier_layers = sorted(
        outlier_layers, key=lambda entry: entry["crest_factor"], reverse=True
    )[:3]
    return {
        "max_abs_weight": max_abs_weight,
        "max_row_crest_factor": max_row_crest_factor,
        "max_block16_crest_factor": max_block16_crest_factor,
        "row_crest_factor_p50": row_crest_p50,
        "row_crest_factor_p99": row_crest_p99,
        "top_3_outlier_layers": top_3_outlier_layers,
    }


def _load_weights(path: Path, errors: list[str]) -> Any:
    """Load a weight file into a nested dict of tensors (or ``{}``)."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".safetensors":
            if _safetensors_load_file is None:
                return {}
            return _safetensors_load_file(str(path), device="cpu")
        if torch is None:
            return {}
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch without ``weights_only``
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        errors.append(f"{path.name}: failed to load weights ({exc})")
        return {}


def _is_floating_tensor(tensor: Any) -> bool:
    """Return True for floating-point or complex torch tensors."""
    if torch is None or not isinstance(tensor, torch.Tensor):
        return False
    return tensor.is_floating_point() or tensor.is_complex()


def _iter_tensors(payload: Any, prefix: str = ""):
    """Yield ``(key, value)`` pairs for every leaf in a nested state dict."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(value, name)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            name = f"{prefix}[{index}]"
            yield from _iter_tensors(value, name)
    else:
        yield prefix or "<root>", payload


GGUF_LADDER_QUANTS: tuple[str, ...] = ("IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")


def verify_gguf_repo_metadata(
    readme_text: str,
    *,
    expected_parent_repo: str | None = None,
    existing_files: list[str] | None = None,
    expected_quants: tuple[str, ...] = GGUF_LADDER_QUANTS,
) -> dict[str, Any]:
    """Validate YAML frontmatter and file ladder of a GGUF repo README."""
    checks: dict[str, bool] = {}
    errors: list[str] = []

    fm: dict[str, Any] = {}
    lines = readme_text.splitlines()
    if lines and lines[0].strip() == "---":
        try:
            end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
        except StopIteration:
            end = None
        if end is None:
            errors.append("readme: unterminated YAML frontmatter block")
        else:
            try:
                loaded = yaml.safe_load("\n".join(lines[1:end]))
            except yaml.YAMLError as exc:
                loaded = None
                errors.append(f"readme: invalid YAML frontmatter ({exc})")
            if loaded is None:
                pass
            elif not isinstance(loaded, dict):
                errors.append("readme: YAML frontmatter is not a mapping")
            else:
                fm = loaded

    checks["library_name_gguf"] = fm.get("library_name") == "gguf"
    if not checks["library_name_gguf"]:
        errors.append("readme: frontmatter library_name must be 'gguf'")
    checks["pipeline_tag_text_generation"] = fm.get("pipeline_tag") == "text-generation"
    if not checks["pipeline_tag_text_generation"]:
        errors.append("readme: frontmatter pipeline_tag must be 'text-generation'")
    checks["base_model_relation_quantized"] = fm.get("base_model_relation") == "quantized"
    if not checks["base_model_relation_quantized"]:
        errors.append("readme: frontmatter base_model_relation must be 'quantized'")
    checks["base_model_present"] = isinstance(fm.get("base_model"), str) and bool(
        fm.get("base_model", "").strip()
    )
    if not checks["base_model_present"]:
        errors.append("readme: frontmatter base_model must be a non-empty string")
    checks["quantized_by_present"] = isinstance(fm.get("quantized_by"), str) and bool(
        fm.get("quantized_by", "").strip()
    )
    if not checks["quantized_by_present"]:
        errors.append("readme: frontmatter quantized_by must be a non-empty string")
    if expected_parent_repo is not None:
        checks["expected_parent_match"] = fm.get("base_model") == expected_parent_repo
        if not checks["expected_parent_match"]:
            errors.append(
                f"readme: frontmatter base_model {fm.get('base_model')!r} does not "
                f"match expected parent repo {expected_parent_repo!r}"
            )

    if existing_files is not None:
        filenames = [Path(f).name for f in existing_files]
        gguf_files = [f for f in filenames if f.endswith(".gguf")]
        found_quants = [
            q for q in expected_quants if any(f.endswith(f"-{q}.gguf") for f in gguf_files)
        ]
        missing_quants = [q for q in expected_quants if q not in found_quants]
        checks["gguf_ladder_complete"] = len(missing_quants) == 0
        if missing_quants:
            errors.append("files: missing GGUF quant ladder entries: " + ", ".join(missing_quants))
        has_recipe = "cka_quant_recipe.json" in filenames
        has_modelfile = "Modelfile" in filenames
    else:
        found_quants = []
        missing_quants = []
        has_recipe = False
        has_modelfile = False

    return {
        "ready": len(errors) == 0,
        "errors": errors,
        "checks": checks,
        "frontmatter": fm,
        "found_quants": found_quants,
        "missing_quants": missing_quants,
        "has_recipe": has_recipe,
        "has_modelfile": has_modelfile,
    }
