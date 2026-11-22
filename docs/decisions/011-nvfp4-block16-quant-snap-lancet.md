# ADR-011: NVFP4 / MXFP4 & Block-16 Subspace-Compensated Quant-Snap Lancet (H18)

- **Status:** Accepted
- **Date:** 2026-11-22
- **Hypotheses:** H18
- **Related:** ADR-006 (Three-Tier Model Storage), ADR-009 (GGUF CKA Survival), ADR-010 (Tenet v2)

---

## Context

`H11` established that norm-preserving Lancet surgery survives `FP8 E4M3` server quantization (`Row Crest-Factor < 25.0`), and `H14` addressed `GGUF` mixed-precision recipes (`cka_quant_recipe.json`). However, uniform 4-bit inference formats — **Blackwell NVFP4** (`E2M1` with `FP8 E4M3` per-16-element block scales), **MXFP4** (`block_size=32`), **MLX 4-bit**, and third-party static **GGUF `Q4_K_M`** quantized without `cka_quant_recipe.json` — introduce two failure modes:

1. **Micro-block outlier amplification:** A model can pass global row-wise crest factor (`Row CF < 25.0`) while still containing localized spikes inside 16-element or 32-element micro-blocks, inflating the per-block scale `s_b` and degrading `E2M1` precision for the remaining 15 weights.
2. **4-bit quantization re-leakage:** Coarse 4-bit rounding re-projects weight energy back onto the ablated subspace $U$, partially undoing refusal/slop suppression when quantized uniformly without mixed-precision protection.

## Decision

1. **Micro-Block Crest-Factor Regularization (`src/sml/surgery/quantization.py`):**
   - Partition rows of $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ into micro-blocks of length $B \in \{16, 32\}$, computing $\text{CF}_B = \max_i |w_i| / (\text{RMS}(w_{\text{block}}) + 10^{-8})$.
   - Iteratively clamp $|w_i| > \tau_B \cdot \text{RMS}(w_{\text{block}})$ ($\tau_{16} = 3.2$) to convergence and restore per-row L2 norms before subspace projection.
   - Emulate `NVFP4` via `quantize_nvfp4_e2m1` using `E2M1` levels $\{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\}$ and `FP8 E4M3FN` block scales $s_b = \text{quantize\_fp8\_e4m3}(\max(|w_b|)/6.0)$ with CPU fallback for MPS.

2. **Iterative Subspace Quant-Error Cancellation (`src/sml/surgery/subspace.py`):**
   - `project_and_snap_subspace_norm_preserving` starts from $W^{(0)} = \text{NormPreservingProject}(W_{\text{hygiene}}, U)$ and runs $K=4$ iterations ($\eta = 0.75$) simulating $Q_4(W^{(k)})$ in `"nvfp4_b16"`, `"int4_b32"`, or `"hybrid_nvfp4_int4"` mode.
   - At each step, compute re-leakage $E^{(k)} = U^\top (U Q_4(W^{(k)}))$, apply $\widetilde{W}^{(k+1)} = W^{(k)} - \eta E^{(k)}$ bounded by relative Frobenius drift $\le 0.02$ against $W_{\text{lancet}}$, restore row L2 norms, and select the candidate $k^*$ minimizing $\|U Q_4(\widetilde{W})\|_F$.
   - Extend `apply_lancet_biprojected_surgery` and `apply_lancet_recipe_streaming` with opt-in `quant_snap`, `quant_snap_iters`, and `max_block16_crest_factor` parameters (100% backwards-compatible with H9–H17).

3. **Hub Validator Telemetry (`src/sml/hub/validator.py`):**
   - Report `max_block16_crest_factor` inside `_inspect_weights` alongside existing row crest-factor diagnostics without altering the 7 `H11` readiness checks.

## Pre-Registered Acceptance Criteria

- `nvfp4_releakage_reduction_ratio >= 0.50`
- `int4_b32_releakage_reduction_ratio >= 0.40`
- `h18_nvfp4_kl_reduction_ratio >= 0.25`
- `max_block16_crest_factor <= 3.35`
- `max_row_crest_factor < 25.0`
- `snap_weight_rel_l2_drift <= 0.02`
