# ADR-014: Strata-Tier-Aware & Activation-Sparsity-Gated MoE Lancet (H21)

- **Date:** 2026-11-24
- **Status:** Accepted
- **Hypotheses:** H21
- **Supersedes / Extends:** ADR-006 (Lancet), ADR-010 (Tri-Anchor Lancet), ADR-011 (H18 Quant-Snap), ADR-012 (H19 Loop-Unrolled SVD), ADR-013 (H20 Router-Aware MoE Lancet)

## Context

ADR-013 established router freezing and selectivity-scaled expert projection for Sparse MoE checkpoints, but two structural failure modes remain at Tenet v3 scale:

1. **Structural-freeze leakage.** Beyond routers, MoE families carry auxiliary learned structures that silently shift when surrounding expert weights are edited: `Engram` N-gram memory tables, `4-branch Gated Residual` / `iHC` hyper-connection gates, and `Looped MoE` tick controllers. Any drift in these components changes the effective per-tick routing/mixing behavior even when `router_weight_drift == 0.0`.
2. **Selectivity is a blunt gate.** The H20 selective weight $s_e$ scales but does not skip inactive experts in aggregate, and the `shared_expert` receives full-strength projection regardless of how much slop/refusal mass actually flows through it.
3. **Uniform quantization hygiene is wasteful.** Most routed experts in large MoE checkpoints are `cold` (rarely activated); imposing `FP8` row-crest hygiene and `H18` micro-block regularization on all of them burns compute, while the `hot` VRAM-resident experts need the full `H11` treatment.

## Decision

1. **Hard Structural Freeze (`verify_moe_structural_freeze`, `src/sml/hub/validator.py`):**
   - Frozen key families, matched by module-name substrings on top of the ADR-013 router freeze:
     - Routers: `gate`, `router`, `routing_weights` (including `e_score_correction_bias`).
     - `Engram` N-gram memory tables: `engram`, `ngram_table`, `memory_table`.
     - `4-branch Gated Residual` / `iHC`: `gated_residual`, `residual_gate`, `ihc`, `hyper_conn`.
     - `Looped MoE` controllers: `loop_controller`, `loop_gate`, `tick_embed`.
   - Enforced both in-memory (`apply_moe_lancet_surgery`) and shard-streaming (`apply_moe_lancet_recipe_streaming`); `structural_weight_drift == 0.0` for every matched tensor.
2. **Activation-Sparsity Gating (`collect_moe_expert_profiles` extension):**
   - `min_selectivity_threshold = 0.25`: routed experts with $s_e < 0.25$ are skipped entirely (not merely scaled), yielding `untouched_expert_ratio >= 0.75` per checkpoint and `0.0` weight drift on skipped experts.
   - `shared_expert_damping = 0.40`: `shared_expert.down_proj` receives an effective step $\beta_{\text{shared}} = \beta \cdot 0.40 \approx 0.14$, damping the projection instead of full strength.
3. **Strata-Tier-Aware Quant-Snap (`strata.toml` passport):**
   - `cold` NVMe-tier experts (offloaded to local NVMe, absent from `hot_vram_expert_ids`): `quant_snap_block16` micro-block hygiene only, gate `max_block16_crest_factor <= 3.20` ($\tau_{16} = 3.2$).
   - `hot` VRAM-tier experts (`hot_vram_expert_ids`) and `shared_expert`: full `FP8`/`Q8_0` row-crest hygiene (`max_row_crest_factor < 25.0`) plus `H18` micro-block regularization.
   - The `strata.toml` passport records per-expert tier assignment, selectivity $s_e$, and applied hygiene mode for `model_card.py` v3 publishing.
4. **Target Tenet v3 6-Model Lineup & Telemetry:**
   - Tier 1: `Tenet-15B-A2B-Qwen3.6-Unslop-v3`.
   - Tier 2: `Tenet-26B-A3B-Inkling2-Unslop-v3`, `Tenet-35B-A3B-Qwen3.6-Unslop-v3`, `Tenet-32B-A3B-GLM4.5-Unslop-v3`.
   - Tier 3: `Tenet-125B-A6B-Qwen3.8-Unslop-v3`, `Tenet-220B-A12B-Hy4-Unslop-v3`.
   - `CANONICAL_TENET_NAMES` / `_TENET_NAME_PATTERN` (`src/sml/hub/model_card.py`) extended for all six names; `SUPPORTED_ARCHITECTURES` (`src/sml/hub/validator.py`) adds the corresponding `Qwen3.6MoE`, `Inkling2`, `GLM4.5`, `Qwen3.8MoE`, and `Hy4` classes; Nebius Managed MLflow telemetry publishes per-tier surgery metrics alongside the existing H20 run.

## Pre-Registered Acceptance Criteria (H21)

- `structural_weight_drift == 0.0` for all router, `Engram`, `gated_residual`/`iHC`, and `Looped MoE` controller tensors (in-memory and streaming).
- `untouched_expert_ratio >= 0.75` with `0.0` drift on skipped routed experts; `shared_expert` effective step $\beta_{\text{shared}} = \beta \cdot 0.40$ measured within `1e-3`.
- `max_block16_crest_factor <= 3.20` on `cold` NVMe-tier experts and `max_row_crest_factor < 25.0` on `hot` VRAM-tier and `shared_expert` weights.
- `verify_provider_readiness` passes for all six Tenet v3 lineup models; `strata.toml` passport present in every published bundle.
- Nebius Managed MLflow run reports per-tier metrics for every Tier 1–3 checkpoint with `h21_verified == True`.
