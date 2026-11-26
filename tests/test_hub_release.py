"""Unit tests for provider readiness validation and Tenet model cards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.hub.build_gguf_ladder import (  # noqa: E402
    build_tensor_type_args,
    resolve_gguf_repo_spec,
)

from sml.hub.model_card import (  # noqa: E402
    GGUF_LADDER_QUANTS,
    TENET_V2_GGUF_REPOS,
    build_model_index,
    generate_tenet_gguf_model_card,
    generate_tenet_model_card,
)
from sml.hub.validator import (  # noqa: E402
    STANDARD_ARCHITECTURES,
    verify_gguf_repo_metadata,
    verify_provider_readiness,
)

CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

LEGACY_KEYS = (
    "ready",
    "errors",
    "checks",
    "architectures",
    "max_abs_weight",
    "max_row_crest_factor",
    "weight_files_checked",
)

NEW_KEYS = (
    "row_crest_factor_p50",
    "row_crest_factor_p99",
    "top_3_outlier_layers",
)


def make_checkpoint(
    tmp_path: Path,
    architecture: str = "LlamaForCausalLM",
    tensors: dict[str, torch.Tensor] | None = None,
    weight_filename: str = "model.safetensors",
    *,
    auto_map: bool = False,
    trust_remote_code: bool = False,
    chat_template: str | None = CHAT_TEMPLATE,
) -> Path:
    """Create a minimal checkpoint directory and return its path."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    config: dict[str, object] = {"architectures": [architecture], "hidden_size": 16}
    if auto_map:
        config["auto_map"] = {"AutoModelForCausalLM": "custom.Modeling"}
    if trust_remote_code:
        config["trust_remote_code"] = True
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tokenizer_config: dict[str, object] = {"model_max_length": 512}
    if chat_template is not None:
        tokenizer_config["chat_template"] = chat_template
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config), encoding="utf-8"
    )
    if tensors is not None:
        save_file(tensors, str(checkpoint / weight_filename))
    return checkpoint


def row_crest(tensor: torch.Tensor) -> float:
    """Compute the row crest factor exactly as the validator does."""
    w = tensor.detach().to(dtype=torch.float32)
    row_abs_max = w.abs().amax(dim=1)
    row_rms = torch.sqrt((w * w).mean(dim=1) + 1e-8)
    return float((row_abs_max / row_rms).max().item())


# ---------------------------------------------------------------------------
# 1. STANDARD_ARCHITECTURES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", sorted(STANDARD_ARCHITECTURES))
def test_standard_architecture_is_ready(tmp_path: Path, architecture: str) -> None:
    assert len(STANDARD_ARCHITECTURES) == 10
    checkpoint = make_checkpoint(tmp_path, architecture=architecture)
    report = verify_provider_readiness(checkpoint)
    assert report["ready"] is True
    assert report["checks"]["standard_architecture"] is True
    assert report["architectures"] == [architecture]
    assert report["errors"] == []


def test_standard_architectures_content() -> None:
    assert {
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
    } == STANDARD_ARCHITECTURES


# ---------------------------------------------------------------------------
# 2. Row crest-factor distribution diagnostics
# ---------------------------------------------------------------------------


def test_report_contains_legacy_and_new_keys(tmp_path: Path) -> None:
    torch.manual_seed(0)
    tensors = {f"model.layers.{i}.self_attn.q_proj.weight": torch.randn(16, 16) for i in range(5)}
    checkpoint = make_checkpoint(tmp_path, tensors=tensors)
    report = verify_provider_readiness(checkpoint)
    for key in LEGACY_KEYS + NEW_KEYS:
        assert key in report, f"missing report key: {key}"


def test_row_crest_diagnostics_five_tensors(tmp_path: Path) -> None:
    torch.manual_seed(42)
    tensors: dict[str, torch.Tensor] = {}
    expected_crests: list[float] = []
    # Vary the row crest factor: gaussian rows plus a single-spike row whose
    # crest factor (sqrt(num_cols) = 8.0) dominates any 16-column row (<= 4.0).
    scales = [0.05, 0.2, 0.6, 1.0]
    for i, scale in enumerate(scales):
        weight = torch.randn(8, 16) * scale
        name = f"model.layers.{i}.mlp.down_proj.weight"
        tensors[name] = weight
        expected_crests.append(row_crest(weight))
    spike = torch.zeros(8, 64)
    spike[:, 0] = 4.0
    tensors["model.layers.4.mlp.down_proj.weight"] = spike
    expected_crests.append(row_crest(spike))
    checkpoint = make_checkpoint(tmp_path, tensors=tensors)
    report = verify_provider_readiness(checkpoint)

    outliers = report["top_3_outlier_layers"]
    assert len(outliers) == 3
    crest_values = [entry["crest_factor"] for entry in outliers]
    assert crest_values == sorted(crest_values, reverse=True)
    for entry in outliers:
        assert isinstance(entry["tensor"], str)
        assert isinstance(entry["crest_factor"], float)
        assert isinstance(entry["max_abs"], float)
    assert outliers[0]["crest_factor"] == pytest.approx(report["max_row_crest_factor"])
    assert outliers[0]["tensor"] == "model.layers.4.mlp.down_proj.weight"
    assert outliers[0]["crest_factor"] == pytest.approx(8.0)
    assert outliers[0]["max_abs"] == pytest.approx(4.0)

    assert report["row_crest_factor_p50"] <= report["row_crest_factor_p99"]
    assert report["row_crest_factor_p99"] <= report["max_row_crest_factor"]
    assert report["max_row_crest_factor"] == pytest.approx(max(expected_crests))
    assert report["weight_files_checked"] == 1


def test_row_crest_diagnostics_only_1d_tensors(tmp_path: Path) -> None:
    tensors = {
        "model.layers.0.input_layernorm.weight": torch.randn(16),
        "model.norm.weight": torch.randn(4),
    }
    checkpoint = make_checkpoint(tmp_path, tensors=tensors)
    report = verify_provider_readiness(checkpoint)
    assert report["row_crest_factor_p50"] == 0.0
    assert report["row_crest_factor_p99"] == 0.0
    assert report["top_3_outlier_layers"] == []
    assert report["max_row_crest_factor"] == 0.0


# ---------------------------------------------------------------------------
# 3. generate_tenet_model_card
# ---------------------------------------------------------------------------

EXPECTED_TAGS = (
    "tenet",
    "lancet",
    "unslop",
    "fp8",
    "vllm",
    "openrouter",
    "grpo",
    "abliterated",
    "uncensored",
    "agentic",
    "tool-use",
    "coding",
    "safetensors",
    "gguf",
    "conversational",
)

FULL_METRICS = {
    "lancet_kl_collateral": 0.0042,
    "naive_kl_collateral": 0.3817,
    "kl_collateral_reduction_ratio": 0.9891,
    "target_suppression_ratio": 0.9765,
    "anchor_orthogonality_error": 0.0031,
    "tool_sql_orthogonality_error": 0.0024,
    "slop_and_refusal_rate": 0.031,
    "h10_gated_vs_naive_gain": 0.118,
    "h10_post_surgery_gain": 0.9876,
    "subspace_safe_lora_releakage": 0.012,
    "unprojected_lora_releakage": 0.64,
    "post_merge_suppression_retention": 0.995,
    "fp8_retention_ratio": 0.994,
    "h11_fp8_kl_reduction_ratio": 0.9,
    "fp8_subspace_releakage_lancet": 0.009,
    "fp8_subspace_releakage_naive": 0.55,
    "max_row_crest_factor": 18.2,
    "max_abs_weight": 12.75,
    "external_benchmarks": {
        "ifeval_strict": 0.81,
        "bfcl_tool_acc": 0.74,
        "humaneval_pass1": 0.68,
    },
}

MODEL_ID = "Tenet-8B-Llama3.1-Unslop-v1"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _frontmatter(card: str) -> str:
    assert card.startswith("---\n")
    return card.split("---", 2)[1]


def test_frontmatter_contains_all_14_tags() -> None:
    card = generate_tenet_model_card(MODEL_ID, BASE_MODEL, {})
    frontmatter = _frontmatter(card)
    tags = [
        line.removeprefix("- ").strip()
        for line in frontmatter.splitlines()
        if line.startswith("- ")
    ]
    assert tags == list(EXPECTED_TAGS)
    assert len(tags) == 15
    for tag in (
        "abliterated",
        "uncensored",
        "agentic",
        "tool-use",
        "coding",
        "safetensors",
        "gguf",
        "conversational",
    ):
        assert tag in tags


def test_full_metrics_render_values_without_na() -> None:
    card = generate_tenet_model_card(MODEL_ID, BASE_MODEL, dict(FULL_METRICS))
    assert "n/a" not in card
    # Numerical values actually rendered.
    assert "4.20e-03" in card  # lancet_kl_collateral (scientific under 0.01)
    assert "0.38" in card  # naive_kl_collateral
    assert "0.99" in card  # kl_collateral_reduction_ratio / retention
    assert "9.00e-03" in card  # fp8_subspace_releakage_lancet
    assert "0.55" in card  # fp8_subspace_releakage_naive
    assert "Max row crest-factor: 18.20" in card
    assert "Max absolute weight: 12.75" in card
    # External benchmarks rendered numerically.
    assert "| IFEval (strict) | 0.81 |" in card
    assert "| BFCL tool accuracy | 0.74 |" in card
    assert "| HumanEval pass@1 | 0.68 |" in card


def test_expected_sections_present() -> None:
    card = generate_tenet_model_card(MODEL_ID, BASE_MODEL, dict(FULL_METRICS))
    assert "## Why Tenet instead of Naive Abliteration?" in card
    assert "## External Benchmarks" in card
    assert "## Side-by-Side: Naive Abliteration vs Tenet (Engineering & SQL Tool-Call)" in card
    assert "## Quickstart (vLLM FP8 / Transformers / llama.cpp CKA-Quant)" in card


def test_pareto_asset_flag_controls_asset_rendering() -> None:
    card_without = generate_tenet_model_card(MODEL_ID, BASE_MODEL, dict(FULL_METRICS))
    assert "lancet_vs_naive_pareto.svg" not in card_without
    card_with = generate_tenet_model_card(
        MODEL_ID, BASE_MODEL, dict(FULL_METRICS), include_pareto_asset=True
    )
    assert "![Lancet vs Naive Pareto](assets/lancet_vs_naive_pareto.svg)" in card_with


def test_empty_metrics_falls_back_to_na_placeholders() -> None:
    card = generate_tenet_model_card(MODEL_ID, BASE_MODEL, {})
    assert card.count("| n/a |") == 9
    assert "## External Benchmarks" not in card


def test_load_stage_metrics_prioritizes_llm_over_nano(tmp_path: Path) -> None:
    import json

    from sml.hub.release import _load_stage_metrics

    nano_dir = tmp_path / "h11_provider_fp8_readiness"
    llm_dir = tmp_path / "h11_qwen3b_mps"
    nano_dir.mkdir(parents=True)
    llm_dir.mkdir(parents=True)

    (nano_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "backend": "nano",
                "model_name": "Qwen/Qwen2.5-3B-Instruct",
                "max_row_crest_factor": 1.86,
            }
        ),
        encoding="utf-8",
    )
    (llm_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "backend": "llm",
                "model_name": "Qwen/Qwen2.5-3B-Instruct",
                "max_row_crest_factor": 17.2706,
            }
        ),
        encoding="utf-8",
    )

    loaded, source = _load_stage_metrics(tmp_path, "h11", "Qwen/Qwen2.5-3B-Instruct")
    assert loaded["backend"] == "llm"
    assert loaded["max_row_crest_factor"] == pytest.approx(17.2706)
    assert source is not None and "h11_qwen3b_mps" in source


# ---------------------------------------------------------------------------
# GGUF companion repo: cards, validator, model-index, ladder tensor-type args
# ---------------------------------------------------------------------------

_GGUF_QUANT_FILES: dict[str, dict[str, str]] = {
    quant: {"size": "15 GB", "bpw": "4.5"} for quant in GGUF_LADDER_QUANTS
}
_CKA_RECIPE: dict[str, list[int]] = {
    "selected_layers": [7, 15, 23],
    "q8_0_layers": [0, 7, 15],
    "q6_k_layers": [3, 21],
    "estimated_avg_bpw": 5.1,
}
_GGUF_EXISTING_FILES = [
    f"Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-{q}.gguf" for q in GGUF_LADDER_QUANTS
] + ["cka_quant_recipe.json", "Modelfile"]


def test_generate_tenet_gguf_model_card_and_validator() -> None:
    # MoE companion card (Gemma4 26B A4B).
    moe_card = generate_tenet_gguf_model_card(
        "Tenet-26B-A4B-Gemma4-Unslop-v2",
        "google/gemma-4-26B-A4B-it",
        quant_files=_GGUF_QUANT_FILES,
        cka_recipe=dict(_CKA_RECIPE),
    )
    moe_report = verify_gguf_repo_metadata(
        moe_card,
        expected_parent_repo="mr-andrey-isachenko/Tenet-26B-A4B-Gemma4-Unslop-v2",
        existing_files=_GGUF_EXISTING_FILES,
    )
    assert moe_report["ready"], moe_report["errors"]
    fm = moe_report["frontmatter"]
    assert fm["library_name"] == "gguf"
    assert fm["pipeline_tag"] == "text-generation"
    assert fm["base_model_relation"] == "quantized"
    assert (
        "ollama run hf.co/mr-andrey-isachenko/"
        "Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF:Q4_K_M" in moe_card
    )
    assert (
        "llama-cli -hf mr-andrey-isachenko/"
        "Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF:Q4_K_M" in moe_card
    )
    assert "ffn_gate_inp" in moe_card

    # Dense companion card (Mistral 24B).
    dense_card = generate_tenet_gguf_model_card(
        "Tenet-24B-Mistral3.1-Unslop-v2",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        quant_files=_GGUF_QUANT_FILES,
        cka_recipe=dict(_CKA_RECIPE),
    )
    dense_report = verify_gguf_repo_metadata(
        dense_card,
        expected_parent_repo="mr-andrey-isachenko/Tenet-24B-Mistral3.1-Unslop-v2",
        existing_files=_GGUF_EXISTING_FILES,
    )
    assert dense_report["ready"], dense_report["errors"]
    assert dense_report["frontmatter"]["library_name"] == "gguf"


def test_parent_card_gguf_tag_and_callout() -> None:
    gguf_repo = TENET_V2_GGUF_REPOS["Tenet-26B-A4B-Gemma4-Unslop-v2"]
    assert gguf_repo == "Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF"

    for model_id, base_model in (
        (
            "Tenet-26B-A4B-Gemma4-Unslop-v2",
            "google/gemma-4-26B-A4B-it",
        ),
        (
            "Tenet-24B-Mistral3.1-Unslop-v2",
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        ),
    ):
        card = generate_tenet_model_card(model_id, base_model, {})
        frontmatter = _frontmatter(card)
        tags = [
            line.removeprefix("- ").strip()
            for line in frontmatter.splitlines()
            if line.startswith("- ")
        ]
        assert "gguf" not in tags, model_id
        assert TENET_V2_GGUF_REPOS[model_id] in card, model_id

    # v1 models keep the legacy inline ``gguf`` tag.
    card_v1 = generate_tenet_model_card(
        "Tenet-8B-Llama3.1-Unslop-v1", "meta-llama/Llama-3.1-8B-Instruct", {}
    )
    tags_v1 = _frontmatter(card_v1)
    assert "- gguf" in tags_v1


def test_build_model_index_strips_workspace_prefix() -> None:
    entries = build_model_index(
        "/workspace/tenet_materialized/Tenet-24B-Mistral3.1-Unslop-v2",
        {"ifeval_strict": 0.757856},
    )
    assert entries and entries[0]["name"] == "Tenet-24B-Mistral3.1-Unslop-v2"
    result = entries[0]["results"][0]
    assert result["source"]["url"] == (
        "https://huggingface.co/mr-andrey-isachenko/Tenet-24B-Mistral3.1-Unslop-v2"
    )
    assert result["metrics"][0]["value"] == pytest.approx(0.757856)


def test_build_tensor_type_args_respects_quant_rank_and_moe() -> None:
    spec = resolve_gguf_repo_spec("Tenet-26B-A4B-Gemma4-Unslop-v2")
    assert spec["is_moe"] is True
    assert spec["file_stem"] == "Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2"

    recipe = {
        "selected_layers": [0, 1, 2, 3],
        "q8_0_layers": [0, 1, 2],
        "q6_k_layers": [3],
    }

    # Q4_K_M run applies both strictly-higher-rank protections.
    q4_args = build_tensor_type_args(recipe, "Q4_K_M")
    assert any("=Q8_0" in arg for arg in q4_args)
    assert any("=Q6_K" in arg for arg in q4_args)

    # Q6_K run applies only Q8_0 rules (never downgrades Q6_K).
    q6_args = build_tensor_type_args(recipe, "Q6_K")
    assert any("=Q8_0" in arg for arg in q6_args)
    assert not any("=Q6_K" in arg for arg in q6_args)

    # Dense Q8_0 run: no surgical downgrade above rank 5 in this recipe.
    dense_q8_args = build_tensor_type_args(recipe, "Q8_0")
    assert dense_q8_args == []

    # MoE runs always freeze the router gate at F16.
    moe_q4_args = build_tensor_type_args(recipe, "Q4_K_M", is_moe=True)
    assert any("ffn_gate_inp" in arg and arg.endswith("=F16") for arg in moe_q4_args)


@pytest.mark.parametrize(
    ("model_id", "expected_gguf_repo", "expected_base_model"),
    [
        (
            "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
            "Tenet-Qwen3.6-35B-A3B-Abliterated-Unslop-v3-GGUF",
            "Qwen/Qwen3.6-35B-A3B",
        ),
        (
            "Tenet-30B-A3B-GLM4.7-Unslop-v3",
            "Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF",
            "zai-org/GLM-4.7-Flash",
        ),
    ],
)
def test_resolve_gguf_repo_spec_v3_moe(
    model_id: str, expected_gguf_repo: str, expected_base_model: str
) -> None:
    spec = resolve_gguf_repo_spec(model_id)
    assert spec["model_id"] == model_id
    assert spec["gguf_repo_name"] == expected_gguf_repo
    assert spec["gguf_repo_id"] == f"mr-andrey-isachenko/{expected_gguf_repo}"
    assert spec["parent_repo_id"] == f"mr-andrey-isachenko/{model_id}"
    assert spec["base_model"] == expected_base_model
    assert spec["is_moe"] is True
    assert spec["file_stem"] == expected_gguf_repo.removesuffix("-GGUF")
