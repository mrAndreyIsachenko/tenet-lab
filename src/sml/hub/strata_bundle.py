"""Strata Engine tier bundling helpers.

Builds a VRAM-hot / NVMe-cold tier profile from expert activity/CKA
sensitivity data, renders a launchable TOML config for the Strata Engine, and
quantifies routing-probability drift (co-activation graph distance) between a
base and a tenet-finetuned router.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dep, defensive only
    np = None  # type: ignore[assignment]

# Tier-1 precision for experts that must always stay resident in VRAM.
HOT_TIER_ALWAYS_QUANT = "Q8_0"
COLD_TIER_QUANTS = ("Q4_K_M", "IQ3_XXS", "NVFP4")
COLD_TIER_LABEL = "H18 Quant-Snapped Q4_K_M / IQ3_XXS / NVFP4"

_ALWAYS_HOT_KEYS = ("shared_expert", "attention", "drafter")


def _normalize_expert_profiles(
    expert_profiles: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize expert profiles to ``expert_id -> metadata``."""
    normalized: dict[str, dict[str, Any]] = {}
    if isinstance(expert_profiles, dict):
        for expert_id, meta in expert_profiles.items():
            if isinstance(meta, dict):
                normalized[str(expert_id)] = dict(meta)
            else:
                normalized[str(expert_id)] = {"score": meta}
        return normalized
    for entry in expert_profiles:
        if not isinstance(entry, dict):
            continue
        expert_id = entry.get("expert_id", entry.get("id"))
        if expert_id is None:
            continue
        normalized[str(expert_id)] = {
            k: v for k, v in entry.items() if k not in ("expert_id", "id")
        }
    return normalized


def _normalize_cka_scores(
    cka_scores: dict[str, float] | list[float],
    expert_ids: list[str],
) -> dict[str, float]:
    """Normalize CKA scores to ``expert_id -> score`` aligned with expert order."""
    if isinstance(cka_scores, dict):
        return {str(k): float(v) for k, v in cka_scores.items()}
    values = [float(v) for v in cka_scores]
    return {expert_id: values[i] for i, expert_id in enumerate(expert_ids) if i < len(values)}


def _expert_activity(meta: dict[str, Any]) -> float:
    """Best available activity/sensitivity signal from an expert's metadata."""
    for key in ("activity", "activation", "score", "usage", "load"):
        if key in meta:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def build_strata_tier_profile(
    expert_profiles: dict[str, Any] | list[dict[str, Any]],
    cka_scores: dict[str, float] | list[float],
    hot_vram_ratio: float = 0.10,
) -> dict[str, Any]:
    """Split experts into a VRAM-hot tier and an NVMe-cold tier.

    The hot tier always includes the always-resident components
    (``shared_expert``, ``attention``, ``drafter``, quantized ``Q8_0``/``FP8``)
    plus the top ``ceil(N * hot_vram_ratio)`` (min 1 when ``N > 0``) experts
    ranked by combined activity / CKA sensitivity. Everything else lands in the
    cold tier.
    """
    profiles = _normalize_expert_profiles(expert_profiles)
    expert_ids = sorted(profiles)
    cka = _normalize_cka_scores(cka_scores, expert_ids)

    total_experts = len(expert_ids)
    hot_count = (
        min(total_experts, max(1, math.ceil(total_experts * hot_vram_ratio)))
        if total_experts
        else 0
    )

    ranked: list[tuple[str, float]] = []
    for expert_id in expert_ids:
        activity = _expert_activity(profiles[expert_id])
        sensitivity = abs(cka.get(expert_id, 0.0))
        combined = activity + sensitivity
        meta = profiles[expert_id]
        quant = str(meta.get("quant", meta.get("quantization", ""))).upper()
        always_hot = any(key in expert_id.lower() for key in _ALWAYS_HOT_KEYS) or quant in (
            "Q8_0",
            "FP8",
        )
        if always_hot:
            continue  # never consumes a ranked hot slot
        ranked.append((expert_id, combined))
    ranked.sort(key=lambda item: (-item[1], item[0]))

    hot_expert_ids = {expert_id for expert_id, _ in ranked[:hot_count]}
    # Always-resident components (named or Q8_0/FP8) join the hot tier too.
    for expert_id in expert_ids:
        meta = profiles[expert_id]
        quant = str(meta.get("quant", meta.get("quantization", ""))).upper()
        if any(key in expert_id.lower() for key in _ALWAYS_HOT_KEYS) or quant in ("Q8_0", "FP8"):
            hot_expert_ids.add(expert_id)

    vram_hot_tier = []
    nvme_cold_tier = []
    for expert_id in expert_ids:
        meta = profiles[expert_id]
        entry = {
            "expert_id": expert_id,
            "activity": round(_expert_activity(meta), 6),
            "cka_sensitivity": round(cka.get(expert_id, 0.0), 6),
            "quant": HOT_TIER_ALWAYS_QUANT if expert_id in hot_expert_ids else COLD_TIER_LABEL,
        }
        (vram_hot_tier if expert_id in hot_expert_ids else nvme_cold_tier).append(entry)

    return {
        "hot_vram_ratio": hot_vram_ratio,
        "total_experts": total_experts,
        "hot_expert_count": len(vram_hot_tier),
        "cold_expert_count": len(nvme_cold_tier),
        "vram_hot_tier": vram_hot_tier,
        "nvme_cold_tier": nvme_cold_tier,
    }


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_strata_toml(
    model_name: str,
    base_model: str,
    tier_profile: dict[str, Any],
    gguf_filename: str,
) -> str:
    """Render a valid TOML config (parsable by ``tomllib.loads``) for the Strata Engine."""
    hot_ids = [entry["expert_id"] for entry in tier_profile.get("vram_hot_tier", [])]
    cold_ids = [entry["expert_id"] for entry in tier_profile.get("nvme_cold_tier", [])]

    def _array(ids: list[str]) -> str:
        if not ids:
            return "[]"
        lines = ",\n".join(f"  {_toml_str(e)}" for e in ids)
        return "[\n" + lines + ",\n]"

    return (
        "# Strata Engine configuration (generated)\n"
        "\n"
        "[model]\n"
        f"name = {_toml_str(model_name)}\n"
        f"base = {_toml_str(base_model)}\n"
        f"gguf = {_toml_str(gguf_filename)}\n"
        f"total_experts = {int(tier_profile.get('total_experts', 0))}\n"
        "\n"
        "[tiers.vram_hot]\n"
        f"hot_vram_ratio = {float(tier_profile.get('hot_vram_ratio', 0.10))}\n"
        f"quant = {_toml_str(HOT_TIER_ALWAYS_QUANT)}\n"
        f"experts = {_array(hot_ids)}\n"
        "\n"
        "[tiers.nvme_cold]\n"
        f"quant = {_toml_str(COLD_TIER_LABEL)}\n"
        f"experts = {_array(cold_ids)}\n"
        "\n"
        "[runtime]\n"
        'offload_backend = "nvme"\n'
        "prefetch = true\n"
        "max_concurrent_expert_fetch = 4\n"
    )


def compute_coactivation_graph_drift(
    base_routing_probs: Any,
    tenet_routing_probs: Any,
) -> float:
    """Frobenius-norm drift between base and tenet co-activation graphs.

    Each routing probability matrix ``[tokens, num_experts]`` is turned into a
    co-activation (Gram) matrix ``P^T @ P`` normalized per token contribution;
    the returned value is the Frobenius norm of the difference. Identical
    routing yields ``0.0``; a frozen/unperturbed router (near-identical probs)
    stays below ``1e-6``.
    """
    if np is None:  # pragma: no cover - numpy is a core dep
        raise ImportError("numpy is required for compute_coactivation_graph_drift")
    base = np.asarray(base_routing_probs, dtype=float)
    tenet = np.asarray(tenet_routing_probs, dtype=float)
    if base.shape != tenet.shape:
        raise ValueError(f"routing prob shapes differ: base={base.shape} tenet={tenet.shape}")
    if base.size == 0:
        return 0.0

    def _coactivation(probs: Any) -> Any:
        # Normalize each token's routing row so the Gram matrix reflects the
        # co-activation pattern independent of per-token routing mass.
        row_norms = np.linalg.norm(probs, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 0, row_norms, 1.0)
        normalized = probs / row_norms
        gram = normalized.T @ normalized
        n_tokens = probs.shape[0]
        return gram / max(1, n_tokens)

    drift = float(np.linalg.norm(_coactivation(base) - _coactivation(tenet), ord="fro"))
    return drift
