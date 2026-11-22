# ADR-012: Loop-Unrolled Tri-Anchor SCR & SVD-Compressed Multi-Pass Biprojection Lancet (H19)

- **Status:** Accepted
- **Date:** 2026-11-23
- **Hypotheses:** H19
- **Related:** ADR-010 (Tenet v2), ADR-011 (NVFP4 Block-16 Quant-Snap Lancet)

---

## Context

`H15` established the Tri-Anchor Lancet (CoT deliberation + reasoning + tool/SQL anchors bi-projected against refusal/slop targets), and `H18` made the resulting surgery survive 4-bit inference formats via Quant-Snap. However, depth-recurrent (looped) Transformers execute the same block several times per forward pass (`num_loop_repeats = K`), which breaks the single-pass surgery contract in two ways:

1. **Execution-tick aliasing:** A plain per-layer contrastive hook only keeps the *last* execution of a looped layer, so refusal/slop and anchor directions extracted from the first ticks (closest to the input, before the residual stream saturates) are silently discarded. Surgery built from tick-0 directions alone mis-targets the K-looped forward pass.
2. **Full-rank unrolled blow-up:** Concatenating every tick's bi-projected targets without compression yields a rank-`2*K` suppression subspace per layer (K=3 → rank 6), which over-ablates the looped residual stream and inflates anchor/CoT collateral KL.

## Decision

1. **Loop-Unrolled Activation Hooks (`src/sml/surgery/subspace.py`):**
   - `collect_loop_unrolled_activations` registers one forward hook per requested layer and keys each captured hidden state by `(layer_idx, tick)`, where `tick` counts how many times that layer has executed in the pass: `t ∈ {0, ..., K-1}` on looped layers, `t = 0` on prefix/suffix layers.
   - `extract_loop_unrolled_contrastive_directions` mirrors `extract_contrastive_direction` over those keys, producing one unit-norm pooled `pos_mean - neg_mean` direction per `(layer, tick)` execution.

2. **`omega_t`-Weighted SVD Bi-Projected Basis (`build_loop_unrolled_biprojected_basis`):**
   - Anchor directions are normalized, weighted by the decaying per-execution weight $\omega_t = K - t + 1$ (1-indexed $t$; equivalently $\omega_t = K - t$ for 0-indexed ticks, applied as $\sqrt{\omega_t}$ row scaling so earlier executions matter more), stacked and SVD-reduced to an orthonormal anchor basis $U_{\text{anchor}}$.
   - Target directions are normalized, weighted the same way, and bi-projected orthogonally to $U_{\text{anchor}}$ via **2-pass modified Gram-Schmidt**; the residuals are SVD-compressed to at most `k_loop` directions and **re-orthogonalized** (2-pass MGS against $U_{\text{anchor}}$ and among themselves), giving $U_{\text{target}}$ with $\|U_{\text{target}} U_{\text{target}}^\top - I\|_{\max} < 10^{-5}$ and $U_{\text{target}} U_{\text{anchor}}^\top \approx 0$.
   - Rank budget: `k_loop <= 2` per layer versus the `2*K = 6` full-rank unrolled concat, so the loop-unrolled surgery never exceeds the H9/H15 single-pass rank.

3. **Loop-Unrolled SCR Layer Selection (`src/sml/surgery/pareto.py`):**
   - `compute_loop_unrolled_scr_profile` takes baseline logits on the full K-looped forward pass, temporarily projects each candidate layer's `o_proj` / `down_proj` weights (row norms preserved) out of that layer's compressed targets, recomputes post-ablation K-looped logits, and scores
     $\text{SCR} = (\text{refusal KL} + \text{slop KL}) / (\text{anchor KL} + \lambda_{\text{cot}}\,\text{cot KL} + \epsilon)$.
   - Original weights are always restored (`finally`-guarded), so profiling leaves the model bit-exact. The top-`top_k_layers` layers by SCR are shared by every compared method to isolate basis construction.
   - `run_h19_loop_unrolled_lancet_benchmark` compares `naive_single_pass` (tick-0 only), `full_rank_unrolled_concat` (all `2*K` tick targets), and `h19_loop_unrolled_svd_biprojection` (this ADR), reporting effective rank, anchor/CoT collateral KL, target-shift KL, orthogonality errors, row-norm drift and crest factors.
   - Surgery applies row-norm-preserving `apply_lancet_biprojected_surgery` with optional `H18` Quant-Snap compensation (`quant_snap`, `quant_snap_iters = 4`, `max_block16_crest_factor = 3.2`), fully backwards-compatible with H9–H18.

## Pre-Registered Acceptance Criteria

- `h19_effective_rank_per_layer <= 2.0`
- `full_rank_unrolled_concat_effective_rank_per_layer == 6.0`
- `h19_total_collateral_kl < full_rank_unrolled_concat_total_collateral_kl`
- `anchor_orthogonality_error < 1e-5`
- `target_orthonormality_error < 1e-5`
- `max_row_norm_rel_drift <= 0.02`
