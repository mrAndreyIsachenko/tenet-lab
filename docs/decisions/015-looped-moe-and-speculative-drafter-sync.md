# ADR-015: Looped-MoE K=3 & Coupled Speculative Drafter / MTP Surgery (H22)

- **Date:** 2026-11-24
- **Status:** Accepted
- **Hypotheses:** H22
- **Supersedes / Extends:** ADR-006 (Lancet), ADR-010 (Tri-Anchor Lancet), ADR-011 (H18 Quant-Snap), ADR-012 (H19 Loop-Unrolled SVD), ADR-013 (H20 Router-Aware MoE Lancet), ADR-014 (H21 Strata-Tier MoE Lancet)

## Context

`H19` solved loop-unrolled SVD bi-projection for depth-recurrent dense blocks, and `H20`/`H21` solved router-aware, sparsity-gated surgery for sparse MoE blocks. Neither composes today, and two coupled-structure gaps remain in the Tenet v3 lineup (`Tenet-26B-A3B-Inkling2-Unslop-v3` in particular):

1. **Looped MoE composition gap.** A depth-recurrent MoE block executes its routed+shared mixture for $K = 3$ ticks $t \in \{0, 1, 2\}$. Applying H20 per-tick independently either yields a rank-$2 \cdot K = 6$ suppression subspace (`Rank Budget Collapse`) or discards early-tick directions; applying H19 naively ignores the MoE routing structure ($s_e$ selectivity and tick-dependent expert activation differ per tick).
2. **Drafter desynchronization.** Speculative decoding drafter and MTP heads (`dspark`, `dfly2`, `drafter`, `mtp_heads`, `eh_proj`) share input embeddings and output projections with the target backbone. Surgery applied to the backbone alone silently changes the distribution the drafter was trained to predict, degrading `speculative_acceptance_rate` and erasing the `$/1k`-incident Pareto gains from Sprint 4.

## Decision

1. **Looped-MoE Unrolled Bi-Projection (`build_looped_moe_biprojected_basis`, `src/sml/surgery/subspace.py`):**
   - Synthesis of `H19` (loop-unrolled $\omega_t$-weighted SVD) $\times$ `H20` (router-aware selectivity): per-tick contrastive directions are extracted keyed by `(layer_idx, tick)` for $t \in \{0, 1, 2\}$ with $K = 3$, and each tick's targets are bi-projected orthogonally to $U_{\text{reason}} + U_{\text{boundary}} + U_{\text{format}}$ (Tri-Anchor basis, ADR-010) via 2-pass float32 Modified Gram-Schmidt.
   - Tick directions are SVD-compressed to rank $k_{\text{loop}} \le 2$ per layer and re-orthogonalized, preserving the H19 rank budget (`effective_rank_per_layer <= 2.0`) instead of the full-rank $2 \cdot K = 6$ concat.
   - Expert selectivity uses tick-averaged activation frequencies $\bar{s}_e$ computed from $p_e^{\text{target}, t}$ and $p_e^{\text{anchor}, t}$ across all ticks (`s_bar_e`), combined with the ADR-014 `min_selectivity_threshold = 0.25` skip gate and `shared_expert_damping = 0.40`.
2. **Coupled Speculative Drafter / MTP Surgery (`synchronize_speculative_drafter_surgery`, `src/sml/surgery/subspace.py`):**
   - Embedded drafter / MTP output projections (`dspark`, `dfly2`, `drafter`, `mtp_heads`, `eh_proj`) receive the same row-norm-preserving bi-projection as their backbone counterparts in the same pass, so drafter and target logits shift coherently.
   - Divergence gate: `drafter_backbone_kl_divergence` (KL between drafter logits and backbone logits on held-out anchor prompts, pre- vs post-surgery) must drop by `>= 35%` relative to the unsynchronized baseline.
   - Acceptance preservation: `speculative_acceptance_rate` must not regress beyond `-0.01` absolute versus the pre-surgery checkpoint on the Fight Card seed set.
3. **Benchmark & Verification:**
   - `run_h22_looped_moe_lancet_benchmark` (`src/sml/surgery/pareto.py`) compares `h20_per_tick_independent`, `h19_dense_naive`, and `h22_looped_moe_biprojection` on effective rank, tick-weighted collateral KL, and drafter divergence.
   - `sml verify-h22` checks the coupled-surgery invariants and publishes Nebius Managed MLflow telemetry; fully backwards-compatible with H9–H21 recipes.

## Pre-Registered Acceptance Criteria (H22)

- `h22_effective_rank_per_layer <= 2.0` (vs `6.0` for per-tick independent concat).
- `h22_total_collateral_kl < h20_per_tick_independent_total_collateral_kl` and `h22_total_collateral_kl < h19_dense_naive_total_collateral_kl`.
- `target_orthonormality_error < 1e-5`, `anchor_orthogonality_error < 1e-5`, `max_row_norm_rel_drift <= 0.02`.
- `drafter_backbone_kl_divergence_reduction >= 0.35` versus the unsynchronized baseline.
- `speculative_acceptance_rate_delta >= -0.01` on held-out Fight Card seeds, and `router_weight_drift == 0.0` / `structural_weight_drift == 0.0` (ADR-013/ADR-014 freezes inherited).
