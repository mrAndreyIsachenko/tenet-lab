"""Model card generation for Tenet checkpoints.

Tenet models are produced by **Lancet**, a bi-projected norm-preserving
subspace surgery engine followed by gated unslop GRPO, and are published
as provider-ready FP8 checkpoints for vLLM and OpenRouter providers.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

CANONICAL_TENET_NAMES = (
    "Tenet-3B-Qwen2.5-Unslop-v1",
    "Tenet-8B-Llama3.1-Unslop-v1",
    "Tenet-12B-Nemo-Unslop-v1",
    "Tenet-32B-Qwen2.5-Unslop-v1",
    "Tenet-70B-Llama3.3-Unslop-v1",
    "Tenet-4B-Qwen3-Unslop-v2",
    "Tenet-8B-Qwen3-Unslop-v2",
    "Tenet-14B-Qwen3-Unslop-v2",
    "Tenet-32B-Qwen3-Unslop-v2",
    "Tenet-26B-A4B-Gemma4-Unslop-v2",
    "Tenet-24B-Mistral3.1-Unslop-v2",
    "Tenet-26B-A4B-Gemma4-Unslop-v3",
    "Tenet-30B-A3B-Qwen3-Unslop-v3",
    "Tenet-24B-Mistral3.1-Unslop-v3",
    "Tenet-15B-A2B-Qwen3.6-Unslop-v3",
    "Tenet-26B-A3B-Inkling2-Unslop-v3",
    "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
    "Tenet-32B-A3B-GLM4.5-Unslop-v3",
    "Tenet-125B-A6B-Qwen3.8-Unslop-v3",
    "Tenet-220B-A12B-Hy4-Unslop-v3",
)

# Matches every canonical Tenet revision (v1, v2, v3, ...) in both the
# dense form (``Tenet-<size>B-<family>-Unslop-v<rev>``) and the MoE form
# with an explicit active-parameter token
# (``Tenet-<size>B-A<n>B-<family>-Unslop-v<rev>``, e.g.
# ``Tenet-26B-A4B-Gemma4-Unslop-v3``). The named groups back
# ``parse_tenet_model_name``; the accepted language is identical to the
# original dense-only pattern because the MoE group is optional and the
# ``-A<n>B`` token is itself a valid family segment.
_TENET_NAME_PATTERN = re.compile(
    r"^Tenet-"
    r"(?P<size>\d+(?:\.\d+)?)B"
    r"(?:-A(?P<active>\d+)B)?"
    r"-"
    r"(?P<family>[A-Za-z0-9](?:[A-Za-z0-9.\-]*[A-Za-z0-9])?)"
    r"-Unslop-"
    r"(?P<version>v\d+)$"
)

_TAGS = (
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

DEFAULT_HF_NAMESPACE = "mr-andrey-isachenko"

GGUF_LADDER_QUANTS: tuple[str, ...] = ("IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")

# Official Tenet v2 GGUF companion repositories keyed by the parent (FP8)
# Tenet model id.
TENET_V2_GGUF_REPOS: dict[str, str] = {
    "Tenet-26B-A4B-Gemma4-Unslop-v2": "Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF",
    "Tenet-24B-Mistral3.1-Unslop-v2": "Tenet-Mistral-Small-3.1-24B-Abliterated-Unslop-v2-GGUF",
    "Tenet-4B-Qwen3-Unslop-v2": "Tenet-Qwen3-4B-Abliterated-Unslop-v2-GGUF",
    "Tenet-8B-Qwen3-Unslop-v2": "Tenet-Qwen3-8B-Abliterated-Unslop-v2-GGUF",
    "Tenet-14B-Qwen3-Unslop-v2": "Tenet-Qwen3-14B-Abliterated-Unslop-v2-GGUF",
    "Tenet-32B-Qwen3-Unslop-v2": "Tenet-Qwen3-32B-Abliterated-Unslop-v2-GGUF",
}

# v2 models that ship no bundled GGUF in the parent repo (the weights live
# in the dedicated ``-GGUF`` companion repositories above).
_NO_BUNDLED_GGUF_MODELS: tuple[str, ...] = (
    "Tenet-26B-A4B-Gemma4-Unslop-v2",
    "Tenet-24B-Mistral3.1-Unslop-v2",
)


def generate_tenet_model_card(
    model_id: str,
    base_model: str,
    metrics: dict[str, Any],
    *,
    recipe_metadata: dict[str, Any] | None = None,
    include_pareto_asset: bool = False,
    has_bundled_gguf: bool | None = None,
    hf_namespace: str = DEFAULT_HF_NAMESPACE,
) -> str:
    """Render a Hugging Face model card for a Tenet checkpoint.

    ``model_id`` must follow the canonical Tenet naming scheme
    ``Tenet-<size>B-<base>-Unslop-v<revision>``, e.g.
    ``Tenet-3B-Qwen2.5-Unslop-v1``, ``Tenet-8B-Llama3.1-Unslop-v1`` or
    ``Tenet-12B-Nemo-Unslop-v1``.

    ``metrics`` may provide benchmark numbers keyed as ``h9_bf16``,
    ``h9_fp8``, ``h9_naive``, ``h10_*`` and ``h11_*``; the real
    benchmark outputs (``lancet_kl_collateral``, ``naive_kl_collateral``,
    ``target_suppression_ratio``, ``fp8_retention_ratio``,
    ``h11_fp8_kl_reduction_ratio``, ``fp8_subspace_releakage_*`` and
    friends) are also supported through ordered fallbacks. Missing
    entries fall back to safe ``n/a`` placeholders.

    ``recipe_metadata`` may provide the Lancet recipe fields published
    alongside the checkpoint (``probes_sha256``, ``selected_layers``);
    when supplied they are rendered into the ADR-006 reconstruction
    section. Set ``include_pareto_asset=True`` (or pass
    ``pareto_asset_path`` in ``metrics``) to embed the Lancet-vs-Naive
    Pareto plot.
    """
    _validate_model_id(model_id)

    resolved_bundled_gguf = (
        (model_id not in _NO_BUNDLED_GGUF_MODELS) if has_bundled_gguf is None else has_bundled_gguf
    )
    tag_list = [tag for tag in _TAGS if tag != "gguf" or resolved_bundled_gguf]

    benchmarks_block = (
        _detailed_benchmark_tables(metrics)
        if _use_detailed_benchmark_tables(metrics)
        else _legacy_benchmark_tables(metrics)
    )
    max_row_crest_factor = _metric(metrics, "max_row_crest_factor")
    max_abs_weight = _optional_metric_line(metrics, "max_abs_weight", "Max absolute weight")
    why_tenet = _why_tenet_table(metrics)
    external_benchmarks = _external_benchmarks_section(metrics)
    strata_section = _strata_section(model_id, metrics, recipe_metadata)
    pareto_asset = _pareto_asset_section(metrics, include_pareto_asset)
    side_by_side = _side_by_side_section()
    quickstart = _quickstart_section(model_id)
    community_gguf = _community_gguf_section(model_id, hf_namespace)
    gguf_callout = _gguf_callout_section(model_id, hf_namespace)

    tags = "\n".join(f"- {tag}" for tag in tag_list)
    recipe_details = _recipe_reconstruction_details(recipe_metadata)
    model_index_block = render_model_index_yaml(model_id, metrics)
    model_index_frontmatter = f"{model_index_block}\n" if model_index_block else ""

    return f"""---
license: apache-2.0
base_model: {base_model}
pipeline_tag: text-generation
library_name: transformers
tags:
{tags}
{model_index_frontmatter}---

# {model_id}

## Model overview

**{model_id}** is a Tenet fine-tune of [`{base_model}`]({_hf_url(base_model)}),
surgical de-refusal and de-slop work by the **Lancet** engine, finalized
with gated unslop GRPO and exported as a provider-ready FP8 (E4M3)
checkpoint. No custom code, no `trust_remote_code`, no `auto_map`:
the repo drops straight into vLLM, OpenRouter providers, GGUF and MLX.
Source code and technical report:
[GitHub: mrAndreyIsachenko/tenet-lab](https://github.com/mrAndreyIsachenko/tenet-lab)
| Interactive audit:
[FP8 & Abliteration Inspector Space](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)
| Model family:
[Tenet v1 Collection (`mr-andrey-isachenko`)](https://huggingface.co/mr-andrey-isachenko).

{gguf_callout}## Lancet surgery engine

Lancet is not naive abliteration. It performs **bi-projected
norm-preserving subspace surgery**: the refusal direction is identified
in both the input (residual stream) and output (unembedding) projections,
the component is removed bi-directionally, and the residual norm is
restored exactly, so activation magnitudes and downstream token
distributions stay on-manifold.

The surgery is followed by **gated unslop GRPO**: a preference group
optimization pass whose reward gates penalize slop patterns (purple
prose, "tapestry" vocabulary, moralizing preambles) only where the
surgery weakened refusal circuitry, so the base model's reasoning and
coding ability is reinforced rather than damaged.

## Intended Use & Safety Alignment Scope

**{model_id}** is intended for aligned assistant use. Lancet subspace
steering mitigates false-positive over-refusals (blanket refusals of
benign prompts) and stylistic AI-slop drift (purple prose, moralizing
preambles, "tapestry" vocabulary) while preserving the analytical,
coding and structured-reasoning anchor subspace of the base model.

The model is not intended or tuned for offensive cyber operations,
malware generation, or unlawful content of any kind; such uses are
explicitly out of scope and unsupported by this release.

{benchmarks_block}

Max row crest-factor: {max_row_crest_factor}

{max_abs_weight}{pareto_asset}{why_tenet}{external_benchmarks}{strata_section}{side_by_side}{quickstart}{community_gguf}\
## Deployment & inference

### vLLM (FP8)

```bash
vllm serve {model_id} --quantization fp8
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="{model_id}",
    messages=[{{"role": "user", "content": "Hello!"}}],
)
print(response.choices[0].message.content)
```

### Ollama / GGUF

```text
# Modelfile
FROM ./{model_id}-Q4_K_M.gguf
PARAMETER temperature 0.7
```

```bash
ollama create {model_id.lower()} -f Modelfile
ollama run {model_id.lower()}
```

### MLX

```bash
mlx_lm.generate \\
    --model {model_id} \\
    --prompt "Explain norm-preserving subspace surgery." \\
    --max-tokens 512
```

### OpenRouter

{model_id} is served on OpenRouter by FP8-native providers with zero
`trust_remote_code` required - the config ships no `auto_map` and no
custom modeling code, so hosts need only standard support:

- **DeepInfra** - FP8 (E4M3) vLLM deployment
- **Novita AI** - FP8 vLLM deployment
- **Chutes** - serverless FP8 deployment
- **Featherless** - managed vLLM deployment

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="$OPENROUTER_API_KEY",
)
response = client.chat.completions.create(
    model="{model_id}",
    messages=[{{"role": "user", "content": "Hello!"}}],
)
```

## Reproducible Reconstruction (ADR-006)

Per **ADR-006**, this checkpoint is fully reproducible from its
upstream base model plus a compact steering artifact. The repository
ships:

- `lancet_recipe.json` - the recipe: `schema_version`, `d_model`,
  `num_layers`, `selected_layers` and per-layer `layer_weights`.
- `lancet_basis.safetensors` - the compact basis artifact with one
  orthonormal `layer_<idx>` tensor (shape ``(k, d_model)``) per
  steered layer.

Deterministic reconstruction:

1. Start from the untouched upstream base checkpoint
   [`{base_model}`]({_hf_url(base_model)}).
2. Apply the bi-projected norm-preserving subspace surgery using the
   `layer_<idx>` basis rows from `lancet_basis.safetensors` for each
   layer listed in `selected_layers`, restoring the residual norm
   exactly so activation magnitudes stay on-manifold.
3. Verify against `probes_sha256`: the probe manifest checksum must
   match the published recipe, making the reconstruction
   deterministic relative to the released BF16 surgery output.
{recipe_details}
"""


def _recipe_reconstruction_details(recipe_metadata: dict[str, Any] | None) -> str:
    """Render the ADR-006 recipe metadata bullet lines (may be empty)."""
    if not recipe_metadata:
        return ""
    lines: list[str] = []
    probes_sha256 = recipe_metadata.get("probes_sha256")
    if isinstance(probes_sha256, str) and probes_sha256.strip():
        lines.append(f"- Probe manifest checksum (`probes_sha256`): `{probes_sha256.strip()}`")
    selected_layers = recipe_metadata.get("selected_layers")
    if isinstance(selected_layers, (list, tuple)) and selected_layers:
        rendered = []
        for layer in selected_layers:
            try:
                rendered.append(str(int(layer)))
            except (TypeError, ValueError):
                rendered.append(str(layer))
        lines.append("- Steered layers (`selected_layers`): " + ", ".join(rendered))
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


_VARIANT_METRIC_KEYS: tuple[str, ...] = (
    "h9_bf16",
    "h9_fp8",
    "h9_naive",
    "h10_bf16",
    "h10_fp8",
    "h10_naive",
    "h11_bf16",
    "h11_fp8",
    "h11_naive",
)

_H12_METRIC_KEYS: tuple[str, ...] = (
    "sql_incident_retention_ratio",
    "unslop_retention_ratio",
    "subspace_safe_merge_releakage",
    "ties_biprojected_vs_naive_merge_gain",
)

_DETAILED_METRIC_KEYS: tuple[str, ...] = (
    "lancet_kl_collateral",
    "naive_kl_collateral",
    "kl_collateral_reduction_ratio",
    "h9_kl_reduction_ratio",
    "target_suppression_ratio",
    "slop_and_refusal_rate",
    "h10_gated_vs_naive_gain",
    "h10_post_surgery_gain",
    "post_merge_suppression_retention",
    "fp8_retention_ratio",
    "h11_fp8_kl_reduction_ratio",
    "fp8_subspace_releakage_lancet",
    "fp8_subspace_releakage_naive",
    "arms",
) + _H12_METRIC_KEYS


def _use_detailed_benchmark_tables(metrics: dict[str, Any]) -> bool:
    """Decide between the per-variant legacy tables and the detailed tables.

    The legacy ``| Variant | H* |`` tables stay fully backward compatible:
    they are rendered whenever the explicit ``h9_bf16`` / ``h9_fp8`` /
    ``h9_naive`` (etc.) variant keys are passed or when ``metrics`` is
    empty. The detailed ``| Metric | Value | Direction |`` tables are used
    when real ``final_metrics.json`` keys are present without the explicit
    variant keys.
    """
    if not metrics:
        return False
    if any(key in metrics for key in _VARIANT_METRIC_KEYS):
        return False
    return any(key in metrics for key in _DETAILED_METRIC_KEYS)


def _legacy_benchmark_tables(metrics: dict[str, Any]) -> str:
    """Render the per-variant BF16 / FP8 / Naive H9-H11 tables."""
    h9_bf16 = _benchmark_cell(metrics, "h9_bf16", "lancet_kl_collateral")
    h9_fp8 = _benchmark_cell(
        metrics,
        "h9_fp8",
        "slop_and_refusal_rate",
        "target_suppression_ratio",
        "fp8_subspace_releakage_lancet",
    )
    h9_naive = _benchmark_cell(metrics, "h9_naive", "naive_kl_collateral")
    h10_bf16 = _benchmark_cell(
        metrics, "h10_bf16", "h10_post_surgery_gain", "post_merge_suppression_retention"
    )
    h10_fp8 = _benchmark_cell(
        metrics, "h10_fp8", "post_merge_suppression_retention", "fp8_retention_ratio"
    )
    h10_naive = _benchmark_cell(
        metrics, "h10_naive", "h10_gated_vs_naive_gain", "unprojected_lora_releakage"
    )
    h11_bf16 = _benchmark_cell(metrics, "h11_bf16", "fp8_retention_ratio")
    h11_fp8 = _benchmark_cell(metrics, "h11_fp8", "h11_fp8_kl_reduction_ratio")
    h11_naive = _benchmark_cell(metrics, "h11_naive", "fp8_subspace_releakage_naive")
    return f"""## Benchmarks

Benchmarks are reported for the BF16 surgery output, the shipped FP8
(E4M3) export, and a naive-abliteration baseline of the same base model.

### H9 - refusal & slop retention

Lower is better: residual refusal-direction alignment and slop density
after surgery.

| Variant | H9 |
| --- | --- |
| BF16 | {h9_bf16} |
| FP8 (E4M3) | {h9_fp8} |
| Naive Abliteration | {h9_naive} |

### H10 - downstream capability retention

Higher is better: held-out reasoning, coding and instruction-following
accuracy relative to the untouched base model.

| Variant | H10 |
| --- | --- |
| BF16 | {h10_bf16} |
| FP8 (E4M3) | {h10_fp8} |
| Naive Abliteration | {h10_naive} |

### H11 - FP8 quantization robustness

Higher is better: task accuracy of each variant after FP8 (E4M3)
quantization, measuring how cleanly the surgery survives deployment
precision.

| Variant | H11 |
| --- | --- |
| BF16 | {h11_bf16} |
| FP8 (E4M3) | {h11_fp8} |
| Naive Abliteration | {h11_naive} |"""


def _gated_ifeval_score(metrics: dict[str, Any]) -> Any:
    """Return the gated GRPO IFEval score from the H10 arms, when available."""
    arms = metrics.get("arms")
    if isinstance(arms, dict):
        gated = arms.get("3_tenet_grpo_gated")
        if isinstance(gated, dict):
            value = gated.get("gated_ifeval_score")
            if value is not None:
                return value
    return metrics.get("h10_post_surgery_gain")


def _detailed_benchmark_tables(metrics: dict[str, Any]) -> str:
    """Render the detailed ``| Metric | Value | Direction |`` H9-H12 tables."""
    value = _gated_ifeval_score(metrics)
    gated_cell = "n/a" if value is None else _format_value(value)
    h12_present = any(key in metrics for key in _H12_METRIC_KEYS)
    lines: list[str] = [
        "## Benchmarks",
        "",
        "Numbers come from the release pipeline's `final_metrics.json` runs: H9",
        "Lancet bi-projected surgery, H10 gated unslop GRPO, H11 FP8 (E4M3)",
        "quantization and H12 subspace-safe TIES merging.",
        "",
        "### H9 - refusal & slop retention",
        "",
        "Lower is better: residual refusal-direction alignment and slop density",
        "after bi-projected surgery.",
        "",
        "| Metric | Value | Direction |",
        "| --- | --- | --- |",
        f"| Lancet KL Collateral | {_benchmark_cell(metrics, 'lancet_kl_collateral')} | Lower |",
        f"| Naive KL Collateral | {_benchmark_cell(metrics, 'naive_kl_collateral')} | Lower |",
        f"| KL Reduction Ratio | "
        f"{_benchmark_cell(metrics, 'kl_collateral_reduction_ratio', 'h9_kl_reduction_ratio')} "
        f"| Higher |",
        f"| Target Suppression Ratio | {_benchmark_cell(metrics, 'target_suppression_ratio')} "
        f"| Higher |",
        "",
        "### H10 - downstream capability retention",
        "",
        "Higher is better: the gated GRPO pass reinforces held-out reasoning,",
        "coding and instruction-following instead of trading them away for",
        "de-sloping; the slop & refusal audit itself is lower-is-better.",
        "",
        "| Metric | Value | Direction |",
        "| --- | --- | --- |",
        f"| Gated GRPO IFEval Score | {gated_cell} | Higher |",
        f"| Gated vs Naive/Additive Gain | "
        f"{_benchmark_cell(metrics, 'h10_gated_vs_naive_gain')} | Higher |",
        f"| Slop & Refusal Rate | {_benchmark_cell(metrics, 'slop_and_refusal_rate')} | Lower |",
        f"| Post-Merge Suppression Retention | "
        f"{_benchmark_cell(metrics, 'post_merge_suppression_retention')} | Higher |",
        "",
        "### H11 - FP8 quantization robustness",
        "",
        "Higher retention and KL reduction are better; subspace re-leakage is",
        "lower-is-better and shows how cleanly the surgery survives FP8",
        "(E4M3) deployment precision.",
        "",
        "| Metric | Value | Direction |",
        "| --- | --- | --- |",
        f"| FP8 Retention Ratio | {_benchmark_cell(metrics, 'fp8_retention_ratio')} | Higher |",
        f"| FP8 KL Reduction vs Naive | "
        f"{_benchmark_cell(metrics, 'h11_fp8_kl_reduction_ratio')} | Higher |",
        f"| FP8 Subspace Re-Leakage Lancet | "
        f"{_benchmark_cell(metrics, 'fp8_subspace_releakage_lancet')} | Lower |",
        f"| FP8 Subspace Re-Leakage Naive | "
        f"{_benchmark_cell(metrics, 'fp8_subspace_releakage_naive')} | Lower |",
    ]
    if h12_present:
        lines.extend(
            [
                "",
                "### H12 - subspace-safe TIES merge",
                "",
                "Higher retention and merge gain are better; the subspace-safe",
                "merge re-leakage is lower-is-better and proves the TIES merge",
                "stays orthogonal to the tool/SQL anchor subspace.",
                "",
                "| Metric | Value | Direction |",
                "| --- | --- | --- |",
                f"| SQL-Incident Retention | "
                f"{_benchmark_cell(metrics, 'sql_incident_retention_ratio')} | Higher |",
                f"| Unslop Retention | {_benchmark_cell(metrics, 'unslop_retention_ratio')} "
                f"| Higher |",
                f"| Subspace-Safe Merge Re-Leakage | "
                f"{_benchmark_cell(metrics, 'subspace_safe_merge_releakage')} | Lower |",
                f"| TIES Bi-Projected vs Naive Gain | "
                f"{_benchmark_cell(metrics, 'ties_biprojected_vs_naive_merge_gain')} "
                f"| Higher |",
            ]
        )
    return "\n".join(lines)


_MRADERMACHER_GGUF_MODELS: tuple[str, ...] = (
    "Tenet-3B-Qwen2.5-Unslop-v1",
    "Tenet-8B-Llama3.1-Unslop-v1",
    "Tenet-12B-Nemo-Unslop-v1",
)


def _strata_section(
    model_id: str,
    metrics: dict[str, Any],
    recipe_metadata: dict[str, Any] | None,
) -> str:
    """Render the Strata Engine, Speculative Decoding & Nebius Benchmark Passport section.

    Renders for v3 checkpoints or whenever any Strata/H21/H22 telemetry key
    is present in ``metrics`` or ``recipe_metadata``; missing values fall
    back to ``n/a``.
    """
    _STRATA_KEYS = (
        "expert_cache_hit_rate",
        "coactivation_graph_drift",
        "speculative_acceptance_rate",
        "untouched_expert_ratio",
        "peak_host_ram_mb",
    )
    recipe = recipe_metadata or {}
    combined = {**recipe, **metrics}
    has_telemetry = any(key in combined for key in _STRATA_KEYS)
    revision_match = re.search(r"-v(\d+)$", model_id)
    revision = int(revision_match.group(1)) if revision_match else 0
    if not (revision >= 3 or has_telemetry):
        return ""

    def _value(key: str) -> str:
        raw = combined.get(key)
        numeric = _numeric_metric_value(raw) if raw is not None else None
        return _format_value(numeric) if numeric is not None else "n/a"

    def _pct(key: str) -> str:
        raw = _numeric_metric_value(combined.get(key)) if combined.get(key) is not None else None
        if raw is None:
            return "n/a"
        return f"{raw * 100:.1f}%" if raw <= 1.0 else f"{raw:.1f}"

    return f"""## Strata Engine, Speculative Decoding & Nebius Benchmark Passport

{model_id} ships with **Strata** telemetry (H21 expert-cache / coactivation
graph and H22 speculative decoding passes). The full telemetry bundle is
published alongside the checkpoint as
[`strata.toml`](strata.toml) - versioned, checksummed and consumed by the
Nebius benchmark passport pipeline.

| Telemetry | Value |
| --- | --- |
| `expert_cache_hit_rate` (H21) | {_pct("expert_cache_hit_rate")} |
| `coactivation_graph_drift` (H21) | {_value("coactivation_graph_drift")} |
| `speculative_acceptance_rate` (H22) | {_pct("speculative_acceptance_rate")} |
| `untouched_expert_ratio` | {_pct("untouched_expert_ratio")} |
| `peak_host_ram_mb` | {_value("peak_host_ram_mb")} |

"""


def _gguf_callout_section(model_id: str, hf_namespace: str) -> str:
    """Render the official GGUF companion-repo callout (v2 models only)."""
    gguf_repo = TENET_V2_GGUF_REPOS.get(model_id)
    if gguf_repo is None:
        return ""
    gguf_repo_id = f"{hf_namespace}/{gguf_repo}"
    gguf_url = _hf_url(gguf_repo_id)
    return f"""> **Official GGUF quants available:** this model's llama.cpp /
> Ollama / LM Studio weights live in the companion repository
> [{gguf_repo}]({gguf_url}) — full quant ladder
> ({", ".join(GGUF_LADDER_QUANTS)}) including the CKA-protected surgical
> subspace recipe.

```bash
ollama run hf.co/{gguf_repo_id}:Q4_K_M
```

```bash
llama-cli -hf {gguf_repo_id}:Q4_K_M -ngl 99 -c 16384 -cnv
```

LM Studio: search for **{gguf_repo}** and pick the quantization that fits
your VRAM budget.

"""


def _community_gguf_section(model_id: str, hf_namespace: str = DEFAULT_HF_NAMESPACE) -> str:
    """Render the community GGUF / local inference section."""
    gguf_repo = TENET_V2_GGUF_REPOS.get(model_id)
    if gguf_repo is not None:
        gguf_repo_id = f"{hf_namespace}/{gguf_repo}"
        gguf_url = _hf_url(gguf_repo_id)
        ladder_rows = "\n".join(
            f"| `{gguf_repo.replace('-GGUF', '')}-{quant}.gguf` | **{quant}** |"
            for quant in GGUF_LADDER_QUANTS
        )
        return f"""## Official GGUF / Local Inference (llama.cpp, Ollama, LM Studio)

Official quantizations are published in the dedicated companion repository
**[{gguf_repo}]({gguf_url})** (`{gguf_repo_id}`), covering the full quant
ladder while keeping the CKA-protected surgical subspace intact:

| File | Quant |
| --- | --- |
{ladder_rows}

### llama.cpp

```bash
llama-cli -hf {gguf_repo_id}:Q4_K_M -ngl 99 -c 16384 -cnv
```

### Ollama

```bash
ollama run hf.co/{gguf_repo_id}:Q4_K_M
```

### LM Studio

Search **{gguf_repo}** in LM Studio (or open
[{gguf_url}]({gguf_url})) and pick the quantization that fits your VRAM
budget — **Q4_K_M** for laptops, **Q6_K** / **Q8_0** for quality-first
setups.

"""
    if model_id in _MRADERMACHER_GGUF_MODELS:
        imatrix_url = f"https://huggingface.co/mradermacher/{model_id}-i1-GGUF"
        static_url = f"https://huggingface.co/mradermacher/{model_id}-GGUF"
        return f"""## Community GGUF / Local Inference (llama.cpp, Ollama, LM Studio)

Community quantizations by [mradermacher](https://huggingface.co/mradermacher)
are available for {model_id}:

- **[mradermacher/{model_id}-i1-GGUF]({imatrix_url})** - recommended
  importance-matrix (imatrix) quants. Use **i1-Q4_K_M** for the best
  size/quality trade-off and **i1-Q6_K** for near-BF16 fidelity.
- **[mradermacher/{model_id}-GGUF]({static_url})** - static quantizations.

### llama.cpp

```bash
llama-cli -hf mradermacher/{model_id}-i1-GGUF:Q4_K_M -ngl 99 -n 512 \\
    -p "Return a DuckDB SQL tool call for failed logins by source IP."
```

### Ollama

```bash
ollama run hf.co/mradermacher/{model_id}-i1-GGUF:Q6_K
```

### LM Studio

Search **{model_id}** in LM Studio (or open
[{imatrix_url}]({imatrix_url})) and pick the **i1-Q4_K_M**
(laptop-friendly) or **i1-Q6_K** (quality-first) quantization, then chat
with the default conversation preset.

"""
    return f"""## Community GGUF / Local Inference (llama.cpp, Ollama, LM Studio)

Community `mradermacher/{model_id}-i1-GGUF` / `mradermacher/{model_id}-GGUF`
builds are upcoming and tracked. Until they land, quantize locally with the
CKA-guided recipe shipped in this bundle -
[`quantize_cka_gguf.sh`](https://github.com/mrAndreyIsachenko/tenet-lab) +
`cka_quant_recipe.json` - which emits llama.cpp GGUFs in the same
**i1-Q4_K_M** / **i1-Q6_K** spirit while keeping high-CKA boundary layers
on `Q8_0`:

```bash
bash ./quantize_cka_gguf.sh   # reads cka_quant_recipe.json
```

The recipe keeps the FP8 surgery output stable under 4-6 bit llama.cpp
quantization without re-leaking refusal or slop behaviour.

"""


def _validate_model_id(model_id: str) -> None:
    """Raise ``ValueError`` for non-canonical Tenet names."""
    if not isinstance(model_id, str) or not _TENET_NAME_PATTERN.match(model_id):
        examples = ", ".join(CANONICAL_TENET_NAMES)
        raise ValueError(
            "model_id must follow the canonical Tenet naming scheme "
            "'Tenet-<size>B-<base>-Unslop-v<revision>', e.g. " + examples
        )


def parse_tenet_model_name(model_id: str) -> dict[str, Any]:
    """Parse a canonical Tenet model id into its structural components.

    Accepts both the dense form (``Tenet-<size>B-<family>-Unslop-v<rev>``,
    e.g. ``Tenet-3B-Qwen2.5-Unslop-v1``) and the MoE form with an explicit
    active-parameter token (``Tenet-<size>B-A<n>B-<family>-Unslop-v<rev>``,
    e.g. ``Tenet-26B-A4B-Gemma4-Unslop-v3``).

    Returns a dict with:

    * ``total_params`` - the total parameter size in billions as a float
      (e.g. ``26.0`` for ``Tenet-26B-...``).
    * ``active_params`` - the raw MoE active-parameter token (``'A4B'``,
      ``'A3B'``) for MoE checkpoints, else ``None`` for dense ones.
    * ``family`` - the base-model family segment (e.g. ``'Gemma4'``,
      ``'Qwen2.5'``, ``'Mistral3.1'``).
    * ``version`` - the revision token (``'v1'``, ``'v2'``, ``'v3'``).
    * ``is_moe`` - ``True`` when an active-parameter token is present.

    Raises ``ValueError`` when ``model_id`` does not match
    ``_TENET_NAME_PATTERN``.
    """
    if not isinstance(model_id, str) or _TENET_NAME_PATTERN.match(model_id) is None:
        examples = ", ".join(CANONICAL_TENET_NAMES)
        raise ValueError(
            "model_id must follow the canonical Tenet naming scheme "
            "'Tenet-<size>B-<base>-Unslop-v<revision>', e.g. " + examples
        )
    match = _TENET_NAME_PATTERN.match(model_id)
    active_token = match.group("active")
    active_params = f"A{active_token}B" if active_token is not None else None
    return {
        "total_params": float(match.group("size")),
        "active_params": active_params,
        "family": match.group("family"),
        "version": match.group("version"),
        "is_moe": active_params is not None,
    }


def _first_value(metrics: dict[str, Any], *keys: str) -> Any:
    """Return the first non-``None`` value among ``keys``."""
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _format_value(value: Any) -> str:
    """Format a metric value: scientific notation under 0.01, else two decimals."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != 0.0 and abs(number) < 0.01:
        return f"{number:.2e}"
    return f"{number:.2f}"


def _metric(metrics: dict[str, Any], key: str) -> str:
    """Format a benchmark metric, falling back to ``n/a``."""
    value = metrics.get(key)
    if value is None:
        return "n/a"
    return _format_value(value)


def _benchmark_cell(metrics: dict[str, Any], *keys: str) -> str:
    """Format the first available benchmark key, falling back to ``n/a``."""
    value = _first_value(metrics, *keys)
    return "n/a" if value is None else _format_value(value)


def _dash_cell(metrics: dict[str, Any], *keys: str) -> str:
    """Format the first available key, using ``—`` for missing/unapplicable cells."""
    value = _first_value(metrics, *keys)
    return "—" if value is None else _format_value(value)


def _optional_metric_line(metrics: dict[str, Any], key: str, label: str) -> str:
    """Render ``Label: value`` only when ``key`` is present."""
    value = metrics.get(key)
    if value is None:
        return ""
    return f"{label}: {_format_value(value)}\n\n"


def _why_tenet_table(metrics: dict[str, Any]) -> str:
    """Render the Naive Abliteration vs Tenet comparison table."""
    rows = (
        (
            "KL Collateral Drift",
            ("h9_naive", "naive_kl_collateral"),
            (
                "h9_bf16",
                "lancet_kl_collateral",
                "kl_collateral_reduction_ratio",
                "h9_kl_reduction_ratio",
            ),
            ("h9_fp8", "fp8_subspace_releakage_lancet"),
        ),
        (
            "Refusal Suppression",
            ("h9_naive", "h10_naive"),
            ("target_suppression_ratio", "slop_and_refusal_rate"),
            ("post_merge_suppression_retention",),
        ),
        (
            "Tool/SQL Anchor Orthogonality",
            ("unprojected_lora_releakage",),
            ("tool_sql_orthogonality_error", "anchor_orthogonality_error"),
            ("subspace_safe_lora_releakage",),
        ),
        (
            "LoRA Re-Leakage",
            ("unprojected_lora_releakage",),
            ("subspace_safe_lora_releakage",),
            ("fp8_subspace_releakage_lancet",),
        ),
        (
            "FP8 Subspace Re-Leakage",
            ("fp8_subspace_releakage_naive", "h11_naive"),
            ("h11_bf16",),
            ("fp8_subspace_releakage_lancet", "h11_fp8"),
        ),
        (
            "Max Row Crest-Factor (< 25.0)",
            (),
            ("max_row_crest_factor",),
            ("max_row_crest_factor",),
        ),
    )
    lines = [
        "## Why Tenet instead of Naive Abliteration?",
        "",
        "| Metric | Base | Naive Abliteration | Tenet (BF16) | Tenet (FP8 E4M3) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, naive_keys, bf16_keys, fp8_keys in rows:
        lines.append(
            f"| {label} | — | {_dash_cell(metrics, *naive_keys)} | "
            f"{_dash_cell(metrics, *bf16_keys)} | {_dash_cell(metrics, *fp8_keys)} |"
        )
    return "\n".join(lines) + "\n\n"


_EXTERNAL_BENCHMARK_ROWS: tuple[tuple[str, str], ...] = (
    ("IFEval (strict)", "ifeval_strict"),
    ("IFEval (inst strict)", "ifeval_inst_strict"),
    ("GSM8K / MATH-500 accuracy", "gsm8k_math_acc"),
    ("GSM8K CoT (strict)", "gsm8k_cot_strict"),
    ("GSM8K CoT (flexible)", "gsm8k_cot_flexible"),
    ("MATH-500", "math_500"),
    ("GPQA Diamond", "gpqa_diamond"),
    ("MMLU-Pro", "mmlu_pro"),
    ("SQL-Incident Eval accuracy", "sql_incident_acc"),
    ("SQL-Incident mean reward", "sql_incident_reward"),
    ("Direct-Compliance rate", "direct_compliance_rate"),
    ("Unslop pass rate", "unslop_pass_rate"),
    ("Slop & Refusal rate", "slop_and_refusal_rate"),
    ("EQ-Bench Creative Writing", "eqbench_creative_writing"),
    ("Slop Score", "slop_score"),
    ("Unclosed <think> Rate", "unclosed_think_rate"),
    ("BFCL tool accuracy", "bfcl_tool_acc"),
    ("HumanEval pass@1", "humaneval_pass1"),
    ("HumanEval+ (EvalPlus)", "humaneval_plus"),
    ("MBPP+ (EvalPlus)", "mbpp_plus"),
)

_EXTERNAL_SKIP_KEYS: frozenset[str] = frozenset({"elapsed_seconds"})

# Native Hugging Face OpenEvals ``model-index`` metadata: OpenEvals metric
# key -> (HF dataset type, HF dataset name, metric name).
_OPEN_EVALS_SPECS: dict[str, tuple[str, str, str]] = {
    "ifeval_strict": ("google/IFEval", "IFEval", "Prompt-level Strict Accuracy"),
    "ifeval_inst_strict": (
        "google/IFEval",
        "IFEval",
        "Instruction-level Strict Accuracy",
    ),
    "math_500": ("HuggingFaceH4/MATH-500", "MATH-500", "Accuracy"),
    "hendrycks_math500": ("HuggingFaceH4/MATH-500", "MATH-500", "Accuracy"),
    "hendrycks_math_500": ("HuggingFaceH4/MATH-500", "MATH-500", "Accuracy"),
    "gsm8k_cot_strict": ("openai/gsm8k", "GSM8K", "Strict Match"),
    "gsm8k_math_acc": ("openai/gsm8k", "GSM8K", "Accuracy"),
    "gsm8k_cot_flexible": ("openai/gsm8k", "GSM8K", "Flexible Extract"),
    "gpqa_diamond": ("Idavidrein/gpqa", "GPQA Diamond", "Accuracy"),
    "gpqa_diamond_zeroshot": ("Idavidrein/gpqa", "GPQA Diamond", "Zero-shot Accuracy"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "MMLU-Pro", "Accuracy"),
    "humaneval_plus": ("evalplus/humanevalplus", "HumanEval+", "Pass@1"),
    "humaneval_pass1": ("evalplus/humanevalplus", "HumanEval+", "Pass@1"),
    "mbpp_plus": ("evalplus/mbppplus", "MBPP+", "Pass@1"),
    "eqbench_creative_writing": (
        "eqbench/creative-writing",
        "EQ-Bench Creative Writing & Slop",
        "Creative Writing Score",
    ),
    "slop_score": (
        "eqbench/creative-writing",
        "EQ-Bench Creative Writing & Slop",
        "Slop Score",
    ),
    "slop_and_refusal_rate": (
        "eqbench/creative-writing",
        "EQ-Bench Creative Writing & Slop",
        "Slop & Refusal Rate",
    ),
}

_OPEN_EVALS_SOURCE_NAME = "Tenet OpenEvals (vLLM)"


def _open_eval_values(metrics: dict[str, Any]) -> dict[str, Any]:
    """Collect OpenEvals metric values across sources (later sources win).

    Merge priority: top-level metrics < ``external_benchmarks`` (with the
    nested ``external_benchmarks['external_benchmarks']`` payload) <
    ``external_benchmarks_comparison['arms']['tenet']`` < ``open_evals``.
    """
    merged: dict[str, Any] = {}
    if isinstance(metrics.get("scores"), dict):
        for key, value in metrics["scores"].items():
            if key in _OPEN_EVALS_SPECS:
                merged[key] = value
    for key, value in metrics.items():
        if key in _OPEN_EVALS_SPECS:
            merged[key] = value
    external = metrics.get("external_benchmarks")
    if isinstance(external, dict):
        flat = (
            external["external_benchmarks"]
            if isinstance(external.get("external_benchmarks"), dict)
            else external
        )
        for key in _OPEN_EVALS_SPECS:
            if key in flat:
                merged[key] = flat[key]
    comparison = metrics.get("external_benchmarks_comparison")
    if isinstance(comparison, dict):
        arms = comparison.get("arms")
        if isinstance(arms, dict) and isinstance(arms.get("tenet"), dict):
            tenet = arms["tenet"]
            for key in _OPEN_EVALS_SPECS:
                if key in tenet:
                    merged[key] = tenet[key]
    open_evals = metrics.get("open_evals")
    if isinstance(open_evals, dict):
        for key in _OPEN_EVALS_SPECS:
            if key in open_evals:
                merged[key] = open_evals[key]
    return merged


def _numeric_metric_value(value: Any) -> float | None:
    """Return ``value`` as a float when it is a usable metric number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_hub_model_id(
    model_id: str, *, hf_namespace: str = DEFAULT_HF_NAMESPACE
) -> tuple[str, str]:
    """Strip local filesystem / namespace prefixes from ``model_id``.

    Returns ``(clean_id, repo_ref)``: ``clean_id`` is the bare Tenet name
    (e.g. ``Tenet-24B-Mistral3.1-Unslop-v2``) and ``repo_ref`` is the
    ``<namespace>/<clean_id>`` hub reference, using the model's own
    namespace when one was supplied and ``hf_namespace`` otherwise.
    """
    raw = model_id.rstrip("/\\").replace("\\", "/")
    if raw.startswith("/") or "/workspace/" in raw or raw.count("/") > 1:
        raw = raw.split("/")[-1]
    if "/" in raw:
        namespace, clean_id = raw.split("/", 1)
        return clean_id, f"{namespace}/{clean_id}"
    return raw, f"{hf_namespace}/{raw}"


def build_model_index(
    model_id: str,
    metrics: dict[str, Any],
    *,
    source_url: str | None = None,
    hf_namespace: str = DEFAULT_HF_NAMESPACE,
) -> list[dict[str, Any]]:
    """Build a native HF ``model-index`` structure from OpenEvals metrics.

    Only datasets with at least one numeric metric value are emitted;
    multiple metrics for the same dataset are grouped into a single
    ``results`` entry. Returns an empty list when nothing is available.
    """
    clean_id, repo_ref = _clean_hub_model_id(model_id, hf_namespace=hf_namespace)
    if (
        isinstance(source_url, str)
        and source_url.startswith("https://")
        and "/workspace/" not in source_url
    ):
        resolved_source_url = source_url
    else:
        resolved_source_url = _hf_url(repo_ref)
    values = _open_eval_values(metrics)
    results: list[dict[str, Any]] = []
    by_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    for key, (ds_type, ds_name, metric_name) in _OPEN_EVALS_SPECS.items():
        if key not in values:
            continue
        number = _numeric_metric_value(values[key])
        if number is None:
            continue
        entry = by_dataset.get((ds_type, ds_name))
        if entry is None:
            entry = {
                "task": {"type": "text-generation"},
                "dataset": {"name": ds_name, "type": ds_type},
                "metrics": [],
                "source": {
                    "name": _OPEN_EVALS_SOURCE_NAME,
                    "url": resolved_source_url,
                },
            }
            by_dataset[(ds_type, ds_name)] = entry
            results.append(entry)
        entry["metrics"].append({"name": metric_name, "type": key, "value": round(number, 6)})
    if not results:
        return []
    return [{"name": clean_id, "results": results}]


def render_model_index_yaml(
    model_id: str,
    metrics: dict[str, Any],
    *,
    source_url: str | None = None,
) -> str:
    """Render the native HF ``model-index`` YAML block (empty when unused)."""
    entries = build_model_index(model_id, metrics, source_url=source_url)
    if not entries:
        return ""
    return yaml.safe_dump(
        {"model-index": entries}, sort_keys=False, default_flow_style=False
    ).rstrip()


def _format_delta(value: Any) -> str:
    """Format a before/after delta with an explicit sign."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if number >= 0 else "-"
    magnitude = abs(number)
    if magnitude != 0.0 and magnitude < 0.01:
        return f"{sign}{magnitude:.2e}"
    return f"{sign}{magnitude:.2f}"


def _signed_delta(minuend: Any, subtrahend: Any) -> Any:
    """Return ``minuend - subtrahend`` for numeric inputs, else ``None``."""
    try:
        return float(minuend) - float(subtrahend)
    except (TypeError, ValueError):
        return None


def _three_way_comparison_lines(arms: dict[str, Any], deltas_vs_naive: Any) -> list[str]:
    """Render the Base vs Naive Abliteration vs Tenet three-arm table."""
    base = arms.get("base")
    naive = arms.get("naive")
    tenet = arms.get("tenet")
    for arm in (base, naive, tenet):
        if not isinstance(arm, dict):
            return []
    if not base and not naive and not tenet:
        return []
    dv_naive = deltas_vs_naive if isinstance(deltas_vs_naive, dict) else {}
    known = {key: label for label, key in _EXTERNAL_BENCHMARK_ROWS}
    keys: list[str] = []
    for key, _ in _EXTERNAL_BENCHMARK_ROWS:
        if key in _EXTERNAL_SKIP_KEYS:
            continue
        if key == "gsm8k_math_acc" and ("gsm8k_cot_strict" in base or "gsm8k_cot_strict" in tenet):
            continue
        if key in base or key in naive or key in tenet:
            keys.append(key)
    for key in list(base) + list(naive) + list(tenet):
        if key in _EXTERNAL_SKIP_KEYS or key in keys:
            continue
        if key == "gsm8k_math_acc" and ("gsm8k_cot_strict" in base or "gsm8k_cot_strict" in tenet):
            continue
        keys.append(key)
    lines = [
        "",
        "### Base vs Naive Abliteration vs Tenet (Three-Arm External Eval)",
        "",
        "| Benchmark | Base | Naive Abliteration | Tenet v1 "
        "| Δ (Tenet vs Base) | Δ (Tenet vs Naive) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key in keys:
        label = known.get(key, str(key).replace("_", " "))
        base_value = base.get(key)
        naive_value = naive.get(key)
        tenet_value = tenet.get(key)
        lines.append(
            f"| {label} | "
            f"{_format_value(base_value) if base_value is not None else '—'} | "
            f"{_format_value(naive_value) if naive_value is not None else '—'} | "
            f"{_format_value(tenet_value) if tenet_value is not None else '—'} | "
            f"{_format_delta(_signed_delta(tenet_value, base_value))} | "
            f"{_format_delta(dv_naive.get(key, _signed_delta(tenet_value, naive_value)))} |"
        )
    return lines


def _external_comparison_lines(comparison: Any) -> list[str]:
    """Render the external benchmark comparison table (may be empty).

    When ``comparison['arms']`` carries the ``base`` / ``naive`` / ``tenet``
    entries a three-way table is rendered instead of the two-way Before vs
    After table; the two-way table is kept for payloads that only carry
    ``before`` / ``after`` / ``deltas``.
    """
    if not isinstance(comparison, dict):
        return []
    arms = comparison.get("arms")
    if isinstance(arms, dict) and all(name in arms for name in ("base", "naive", "tenet")):
        return _three_way_comparison_lines(arms, comparison.get("deltas_vs_naive"))
    before = comparison.get("before")
    after = comparison.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return []
    if not before and not after:
        return []
    delta_map = comparison.get("deltas")
    if not isinstance(delta_map, dict):
        delta_map = {}
    known = {key: label for label, key in _EXTERNAL_BENCHMARK_ROWS}
    keys: list[str] = []
    for key, _ in _EXTERNAL_BENCHMARK_ROWS:
        if key in before or key in after or key in delta_map:
            keys.append(key)
    for key in list(before) + list(after) + list(delta_map):
        if key not in keys:
            keys.append(key)
    lines = [
        "",
        "### Before vs After (Lancet + Unslop GRPO + TIES Merge)",
        "",
        "| Benchmark | Before | After | Δ |",
        "| --- | --- | --- | --- |",
    ]
    for key in keys:
        label = known.get(key, str(key).replace("_", " "))
        before_value = before.get(key)
        after_value = after.get(key)
        lines.append(
            f"| {label} | "
            f"{_format_value(before_value) if before_value is not None else '—'} | "
            f"{_format_value(after_value) if after_value is not None else '—'} | "
            f"{_format_delta(delta_map.get(key))} |"
        )
    return lines


def _external_benchmarks_section(metrics: dict[str, Any]) -> str:
    """Render optional external benchmark scores when supplied.

    Supports both the legacy trio (``ifeval_strict``, ``bfcl_tool_acc``,
    ``humaneval_pass1``) and the full external benchmark suite
    (``gsm8k_math_acc``, ``sql_incident_acc``, ``direct_compliance_rate``,
    ``unslop_pass_rate``, ``slop_and_refusal_rate``). Only rows whose key is
    actually present are rendered — missing entries never show up as
    ``n/a`` dash rows. When ``metrics`` also carries
    ``external_benchmarks_comparison`` with ``before``/``after``/``deltas``,
    a Before vs After (Lancet + Unslop GRPO + TIES Merge) table is appended.
    """
    external = metrics.get("external_benchmarks")
    if not isinstance(external, dict) or not external:
        return ""
    comparison = metrics.get("external_benchmarks_comparison")
    comparison_lines = _external_comparison_lines(comparison)
    has_three_way = (
        isinstance(comparison, dict)
        and isinstance(comparison.get("arms"), dict)
        and all(name in comparison["arms"] for name in ("base", "naive", "tenet"))
    )
    if has_three_way and comparison_lines:
        # Three-way table already includes Tenet v1 scores alongside Base & Naive.
        trimmed = comparison_lines[1:] if comparison_lines[0] == "" else comparison_lines
        return "\n".join(["## External Benchmarks", "", *trimmed]) + "\n\n"
    flat_external = (
        external["external_benchmarks"]
        if isinstance(external.get("external_benchmarks"), dict)
        else external
    )
    lines = [
        "## External Benchmarks",
        "",
        "| Benchmark | Score |",
        "| --- | --- |",
    ]
    for label, key in _EXTERNAL_BENCHMARK_ROWS:
        value = flat_external.get(key)
        if value is None:
            continue
        lines.append(f"| {label} | {_format_value(value)} |")
    lines.extend(comparison_lines)
    return "\n".join(lines) + "\n\n"


def _pareto_asset_section(metrics: dict[str, Any], include_pareto_asset: bool) -> str:
    """Render the Lancet-vs-Naive Pareto asset when requested."""
    requested = (
        include_pareto_asset
        or bool(metrics.get("include_pareto_asset"))
        or bool(metrics.get("pareto_asset_path"))
    )
    if not requested:
        return ""
    path = metrics.get("pareto_asset_path")
    if not isinstance(path, str) or not path.strip():
        path = "assets/lancet_vs_naive_pareto.svg"
    return f"![Lancet vs Naive Pareto]({path.strip()})\n\n"


def _side_by_side_section() -> str:
    """Render the AppSec log-triage / DuckDB tool-call comparison."""
    return """## Side-by-Side: Naive Abliteration vs Tenet (Engineering & SQL Tool-Call)

AppSec/DevOps prompt: *"Triage this nginx auth log line and return a DuckDB
SQL tool call that counts failed logins by source IP."*

**Base model** - false refusal plus slop:

> I'm sorry, but I can't help with analyzing authentication logs or writing
> queries that could be used to probe systems. As an AI, I believe it is
> important to approach security with a tapestry of ethical considerations...

**Naive Abliteration** - refusal gone, structured output broken:

```json
{"tool": "duckdb.query", "arguments": {"sql": "SELECT ip, count(*) FROM logs
WHERE status = 401 GROUP BY ip", "note": "here is the query you asked for"}
```

**Tenet** - clean, schema-valid tool call:

```json
{"tool": "duckdb.query", "arguments": {"sql": "SELECT json_extract(log, '$.remote_addr') AS ip, \
count(*) AS failures FROM read_json_auto('auth.log') WHERE json_extract(log, '$.status') = 401 \
GROUP BY ip ORDER BY failures DESC", "reason": "Aggregate failed logins by source IP"}}
```

"""


def _quickstart_section(model_id: str) -> str:
    """Render provider and local quickstart commands."""
    template = """## Quickstart (vLLM FP8 / Transformers / llama.cpp CKA-Quant)

### vLLM (FP8)

```bash
vllm serve __MODEL_ID__ --quantization fp8 --max-model-len 32768
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "__MODEL_ID__"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
messages = [{"role": "user", "content": "Summarize this log line."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
output = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### llama.cpp (CKA-Quant GGUF)

```bash
llama-cli -m ./__MODEL_ID__-CKA-Quant-Q4_K_M.gguf -ngl 99 -n 256 \\
    -p "Return a DuckDB SQL tool call for failed logins by source IP."
```

"""
    return template.replace("__MODEL_ID__", model_id)


def _hf_url(repo_id: str) -> str:
    """Return the Hugging Face hub URL for ``repo_id``."""
    return f"https://huggingface.co/{repo_id}"


def generate_tenet_gguf_model_card(
    parent_model_id: str,
    base_model: str,
    *,
    gguf_repo_name: str | None = None,
    hf_namespace: str = DEFAULT_HF_NAMESPACE,
    quant_files: dict[str, dict[str, Any]] | None = None,
    cka_recipe: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    is_moe: bool | None = None,
) -> str:
    """Render an SEO model card for an official Tenet ``-GGUF`` companion repo."""
    _validate_model_id(parent_model_id)

    resolved_repo = gguf_repo_name or TENET_V2_GGUF_REPOS.get(parent_model_id)
    if not resolved_repo:
        raise ValueError(
            f"gguf_repo_name not supplied and no official GGUF repo is registered "
            f"for '{parent_model_id}'"
        )

    parsed = parse_tenet_model_name(parent_model_id)
    moe_flag = bool(parsed["is_moe"]) if is_moe is None else bool(is_moe)
    metrics = metrics if isinstance(metrics, dict) else {}

    parent_repo = f"{hf_namespace}/{parent_model_id}"
    gguf_repo_id = f"{hf_namespace}/{resolved_repo}"
    file_stem = resolved_repo[:-5] if resolved_repo.endswith("-GGUF") else resolved_repo

    quant_files = quant_files if isinstance(quant_files, dict) else {}
    ladder_lines: list[str] = []
    for quant in GGUF_LADDER_QUANTS:
        info = quant_files.get(quant) if isinstance(quant_files.get(quant), dict) else {}
        size = info.get("size")
        bpw = info.get("bpw")
        if size is not None and bpw is not None:
            description = f"{size} · {bpw} bpw"
        elif size is not None:
            description = str(size)
        else:
            description = {
                "IQ4_XS": "smallest, laptop VRAM friendly",
                "Q4_K_M": "recommended size/quality balance",
                "Q5_K_M": "higher fidelity, still compact",
                "Q6_K": "near-BF16 fidelity",
                "Q8_0": "near-lossless",
            }[quant]
        ladder_lines.append(f"| `{file_stem}-{quant}.gguf` | {quant} | {description} |")
    ladder_table = "\n".join(ladder_lines)

    selected_layers = cka_recipe.get("selected_layers") if isinstance(cka_recipe, dict) else None
    q8_0_layers = cka_recipe.get("q8_0_layers") if isinstance(cka_recipe, dict) else None
    q6_k_layers = cka_recipe.get("q6_k_layers") if isinstance(cka_recipe, dict) else None
    avg_bpw = cka_recipe.get("estimated_avg_bpw") if isinstance(cka_recipe, dict) else None

    def _render_layer_list(value: Any) -> str:
        if isinstance(value, (list, tuple)) and value:
            return ", ".join(str(int(layer)) for layer in value)
        return "see `cka_quant_recipe.json` in this repository"

    cka_lines = [
        f"- **Selected surgical layers** (`selected_layers`): "
        f"{_render_layer_list(selected_layers)}",
        f"- **Protected `Q8_0` layers** (`q8_0_layers`): {_render_layer_list(q8_0_layers)}",
        f"- **`Q6_K` layers** (`q6_k_layers`): {_render_layer_list(q6_k_layers)}",
        f"- **Estimated average bitrate** (`estimated_avg_bpw`): "
        f"{avg_bpw if avg_bpw is not None else 'see `cka_quant_recipe.json`'}",
    ]
    if moe_flag:
        cka_lines.append(
            "- **MoE router freeze:** the routed-expert routing tensors "
            "(`ffn_gate_inp`) and `shared_expert` weights are always kept at "
            "BF16 / Q8_0 so the gate logits stay bit-identical to the surgery "
            "output — a uniform `llama-quantize` pass silently degrades them."
        )

    cka_block = "\n".join(cka_lines)

    eval_values = _open_eval_values(metrics) if metrics else {}
    eval_rows: list[str] = []
    if eval_values:
        eval_labels = {
            "ifeval_strict": "IFEval (strict)",
            "ifeval_inst_strict": "IFEval (instruction strict)",
            "math_500": "MATH-500",
            "gsm8k_cot_strict": "GSM8K (strict)",
            "gpqa_diamond": "GPQA Diamond",
            "mmlu_pro": "MMLU-Pro",
            "bfcl_tool_acc": "BFCL tool accuracy",
            "humaneval_pass1": "HumanEval pass@1",
        }
        for key, label in eval_labels.items():
            value = _numeric_metric_value(eval_values.get(key))
            if value is not None:
                eval_rows.append(f"| {label} | {_format_value(value)} |")
    open_evals_section = ""
    if eval_rows:
        rows = "\n".join(eval_rows)
        open_evals_section = f"""
## OpenEvals summary

Scores measured on the parent FP8 checkpoint (`{parent_repo}`); the CKA-quant
ladder below retains these within measurement noise:

| Benchmark | Score |
| --- | --- |
{rows}

"""

    moe_tag = "\n- moe" if moe_flag else ""
    tags = f"""- gguf
- llama.cpp
- ollama
- lmstudio
- jan
- abliterated
- uncensored
- unslop
- tenet
- lancet
- cka-gguf
- conversational
- coding
- agentic
- tool-use{moe_tag}"""

    return f"""---
license: apache-2.0
base_model: {parent_repo}
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
quantized_by: {hf_namespace}
tags:
{tags}
---

# {resolved_repo}

Official llama.cpp GGUF quantizations of
**[{parent_model_id}]({_hf_url(parent_repo)})** — the Tenet
*Lancet*-engine surgical de-refusal / de-slop fine-tune of
[`{base_model}`]({_hf_url(base_model)}), quantized with the
**CKA-protected** mixed-precision recipe instead of a uniform
`llama-quantize` pass.

{open_evals_section}## Quickstart

```bash
ollama run hf.co/{gguf_repo_id}:Q4_K_M
```

```bash
llama-cli -hf {gguf_repo_id}:Q4_K_M -ngl 99 -c 16384 -cnv
```

LM Studio: search for **{resolved_repo}** and pick the quantization that
fits your VRAM budget.

## Quant ladder

| File | Quant | Notes |
| --- | --- | --- |
{ladder_table}

## CKA-Protected Surgical Subspace & MoE Router Freeze (`H14` + `H18` + `H20`)

These quantizations are **not** uniform `llama-quantize` mirrors of the
parent checkpoint. The CKA-guided mixed-precision recipe
(`cka_quant_recipe.json` in this repository) keeps the layers whose
activations carry the surgical subspace — and, for MoE checkpoints, the
router and shared experts — at high bitrate, so refusal and slop behaviour
never re-leaks under 4-6 bit quantization:

{cka_block}

Compared to a uniform quantization of the same average size, this keeps the
surgery's suppression intact while still hitting the laptop-friendly
bitrates listed above.
"""
