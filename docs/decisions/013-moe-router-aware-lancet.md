# ADR-013: Router-Aware & Shared-Expert Lancet Surgery for Sparse MoE (H20)

- **Date:** 2026-11-23
- **Status:** Accepted
- **Hypotheses:** H20
- **Supersedes / Extends:** ADR-006 (Lancet), ADR-010 (Tri-Anchor Lancet), ADR-011 (H18 Quant-Snap), ADR-012 (H19 Loop-Unrolled SVD)

## Context

Dense Transformer surgery (`self_attn.o_proj` and `mlp.down_proj`) assumes every parameter in the MLP block participates uniformly across refusal/slop and capability prompts. In Sparse Mixture-of-Experts (MoE) architectures—such as **Google Gemma 4 (`26B-A4B`)** with hybrid sliding/global attention, packed 3D routed experts, and a shared MLP/expert, as well as **Qwen3-MoE (`30B-A3B`)** with unpacked `mlp.experts.{e}.down_proj`—naive uniform editing causes two failure modes:

1. **Router destabilization:** Editing `mlp.gate` / `router` / `block_sparse_moe.gate` alters token-to-expert routing probabilities, causing cascading distribution shift and calibration collapse.
2. **Expert collateral damage:** Applying full-strength surgical projection to every routed expert degrades specialized experts (code, SQL, mathematics) that are rarely active on refusal/slop prompts but heavily active on `Tri-Anchor` prompts.

Additionally, `Small Model Lab` needed first-class module mapping, `safetensors` streaming, `model_card.py` `v3` naming, and `validator.py` readiness checks for **Gemma 4**, **Qwen3-MoE**, and **Mistral Small 3.1 / Devstral Small (24B, Apache 2.0)**.

## Decision

1. **Router Freeze Guarantee:**
   - Router/gate weights (`mlp.gate`, `router`, `router.proj`, `block_sparse_moe.gate`, including biases and DeepSeek `e_score_correction_bias`) are strictly frozen during both in-memory (`apply_moe_lancet_surgery`) and shard-streaming (`apply_moe_lancet_recipe_streaming`) surgery.
   - `sml.hub.validator.verify_moe_router_freeze` verifies `router_weight_drift == 0.0` across all router tensors.
2. **Expert Activation Profiling (`collect_moe_expert_profiles`):**
   - Forward hooks on router modules collect normalized expert activation frequencies $p_e^{\text{target}}$ on refusal/slop prompts and $p_e^{\text{anchor}}$ on `Tri-Anchor` (`cot`, `math`, `tool`) prompts.
   - Each expert $e$ receives a selective surgical weight:
     $$s_e = \operatorname{clip}\!\left(\frac{p_e^{\text{target}}}{p_e^{\text{target}} + \beta \, p_e^{\text{anchor}} + \epsilon},\ 0,\ 1\right)$$
3. **Differentiated Projection (`apply_moe_lancet_surgery` & `apply_moe_lancet_recipe_streaming`):**
   - `self_attn.o_proj` and `shared_expert.down_proj` (or dense/shared `mlp.down_proj` in MoE layers) receive full Tri-Anchor `Lancet` projection + optional `H18 Quant-Snap` (`nvfp4_b16`, `int4_b32`, `hybrid_nvfp4_int4`) and micro-block crest-factor regularization (`max_block16_crest_factor <= 3.2`).
   - `routed_experts[e].down_proj`—supporting both unpacked `ModuleList` experts and packed/fused 3D `(E, out_features, in_features)` tensors in `safetensors`—are projected with effective scale `scale * s_e`, skipping anchor-only experts (`s_e <= min_selective_weight`).
4. **Architecture & `v3` Naming Support:**
   - `get_transformer_layers` and `get_moe_layer_components` (`src/sml/surgery/cka.py`) support `model.language_model.model.layers`, `model.language_model.layers`, `model.model.layers`, and `model.layers`.
   - `CANONICAL_TENET_NAMES` and `_TENET_NAME_PATTERN` (`src/sml/hub/model_card.py`) support `Tenet-26B-A4B-Gemma4-Unslop-v3`, `Tenet-30B-A3B-Qwen3-Unslop-v3`, and `Tenet-24B-Mistral3.1-Unslop-v3` with 100% backward compatibility for `v1`/`v2`.
   - `SUPPORTED_ARCHITECTURES` (`src/sml/hub/validator.py`) adds `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, and `Mistral3ForConditionalGeneration`, and `_inspect_weights` inspects 3D packed expert slices for `max_block16_crest_factor`, outliers, and NaN/Inf.

## Pre-Registered Acceptance Criteria (H20)

- `router_weight_drift == 0.0` (exact bit-for-bit router freeze in-memory and streaming).
- Anchor-only routed experts (`s_e <= min_selective_weight`) exhibit `0.0` weight drift while `o_proj`, `shared_expert.down_proj`, and target-heavy routed experts are projected along the Tri-Anchor surgical basis.
- `max_block16_crest_factor <= 3.35` and `max_row_crest_factor < 25.0` when `H18 Quant-Snap` / micro-block regularization is enabled.
- `verify_provider_readiness` passes for `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM`, `Qwen3MoeForCausalLM`, and `Mistral3ForConditionalGeneration`.
- `100%` backward compatibility with `Tenet v1`/`v2` naming and `STANDARD_ARCHITECTURES`.
