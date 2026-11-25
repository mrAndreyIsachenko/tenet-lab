# Tenet Lab (Small Model Lab & Tenet Family)

**Research Question:** *How small and efficient can a language model become while retaining most of the capability required for production data-pipeline incident reasoning — and how can bi-projected subspace surgery (`Lancet`) and subspace-safe RL (`Tenet`) eliminate slop and false refusals without reasoning collateral or FP8 degradation?*

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](pyproject.toml)
[![CI](https://github.com/mrAndreyIsachenko/tenet-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/mrAndreyIsachenko/tenet-lab/actions/workflows/ci.yml)
[![Hypotheses H1–H19](https://img.shields.io/badge/Hypotheses_H1--H19-18%2F19_Verified-success.svg)](reports/sprint1/summary.md)
[![Hugging Face Space](https://img.shields.io/badge/HF_Space-FP8_Abliteration_Inspector-yellow.svg)](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)
[![Hugging Face Models](https://img.shields.io/badge/HF_Family-Tenet_v1_%26_v2-orange.svg)](https://huggingface.co/mr-andrey-isachenko)
[![Model Fight Card 16](https://img.shields.io/badge/Model_Fight_Card_16-Majors_(8B)_%26_Minors_(3B)_Champions-f5c542.svg)](https://onlymodels.gg/cards/2026-11-20.html)

---

## Hugging Face Showcase (`Tenet v1` & `Tenet v2` Collections + Inspector Space)

* **Hugging Face Profile:** [`mr-andrey-isachenko`](https://huggingface.co/mr-andrey-isachenko)
* **Hugging Face Research Article:** [Tenet v2: Weight Surgery Without Breaking Reasoning](https://huggingface.co/blog/mr-andrey-isachenko/tenet-v2-thinking-safe-weight-surgery)
* **Tenet v1 Collection:** [Tenet v1 — Bi-Projected Unslop & FP8-Safe LLM Family](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v1-bi-projected-unslop-and-fp8-safe-llm-family-6afed2048ba13a74fb7e0fac)
* **Tenet v2 Collection:** [Tenet v2 — Thinking-Safe Tri-Anchor Unslop & Domain CKA-GGUF](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4)
* **Interactive Audit Space:** [`mr-andrey-isachenko/fp8-abliteration-inspector`](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)
* **Released Models (`Tenet v2` — Thinking-Safe Tri-Anchor, Cross-Family Quant-Snap & Router-Frozen MoE + Domain CKA-GGUF):**
  * [`mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2) (`Qwen3-32B` — `9/10` gates, **`100.0 HP` `24/24`** on `F16` & pure `CKA-Q4_K_M`, `Crest=4.36`, `FP8=2.56%`)
  * [`mr-andrey-isachenko/Tenet-26B-A4B-Gemma4-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-26B-A4B-Gemma4-Unslop-v2) (`google/gemma-4-26B-A4B-it` MoE — `H20 + H18` Router-Frozen MoE Streaming Top-2 Power-Iteration + Quant-Snap: `30/30` routers frozen with `0.0` drift, `0-shot` Chat **IFEval Strict `86.32%` (`90.41%` Inst-Strict)** & **MATH-500 `75.00%`**)
  * Companion GGUF: [`mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF`](https://huggingface.co/mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF) (`IQ4_XS` / `Q4_K_M` / `Q5_K_M` / `Q6_K` / `Q8_0` CKA-guided mixed-precision `GGUF` ladder, `256K` ctx)
  * [`mr-andrey-isachenko/Tenet-24B-Mistral3.1-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-24B-Mistral3.1-Unslop-v2) (`Mistral-Small-3.1-24B-Instruct-2503` — `H20 + H18` Streaming Quant-Snap, `Crest=9.01 < 25.0`, **IFEval Prompt-Strict `75.79%` vs `65.99%` Base (`+9.80 pp`)**, Inst-Strict `82.73%` (`+7.31 pp`), GSM8K Flex `89.84%`, GPQA `39.39%`, MMLU-Pro `63.49%`)
  * Companion GGUF: [`mr-andrey-isachenko/Tenet-Mistral-Small-3.1-24B-Abliterated-Unslop-v2-GGUF`](https://huggingface.co/mr-andrey-isachenko/Tenet-Mistral-Small-3.1-24B-Abliterated-Unslop-v2-GGUF) (`IQ4_XS` / `Q4_K_M` / `Q5_K_M` / `Q6_K` / `Q8_0` CKA-guided mixed-precision `GGUF` ladder, `128K` ctx)
  * [`mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2) (`Qwen3-14B` — `5/10` gates, `Crest=3.60`, `FP8=2.59%`, **`95.83 HP`** pure `CKA-Q4_K_M`, `6/6` SQL vs `0/6` `imatrix`)
  * [`mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2) (`Qwen3-8B` — `10/10` gates, `100 HP` `24/24`, `100%` CoT integrity, `i1-CKA-Q4_K_M` KLD `0.01314`)
  * [`mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2) (`Qwen3-4B` — `10/10` gates, `100 HP` `24/24`, `100%` CoT integrity, `0%` length collapse)
* **Released Models & Audit Bundles (`Tenet v1`):**
  * [`mr-andrey-isachenko/Tenet-32B-Qwen2.5-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-32B-Qwen2.5-Unslop-v1) (`Qwen2.5-32B-Instruct`)
  * [`mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1) (`Mistral-Nemo-Instruct-2407`)
  * [`mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1) (`Llama-3.1-8B-Instruct` — **Majors Division Champion**, [Model Fight Card 16](https://onlymodels.gg/cards/2026-11-20.html))
  * [`mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1-CKA-GGUF`](https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1-CKA-GGUF) (8B CKA-guided mixed-precision `GGUF` bundle)
  * [`mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1) (`Qwen2.5-3B-Instruct` — **Minors Division Champion at 90 HP**, [Model Fight Card 16](https://onlymodels.gg/cards/2026-11-20.html))
  * [`mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1) (16K-param Nano verification & `H1–H13` artifact repo, renamed from `small-model-lab`)

> **Release Cadence & Versioning Policy:** Major `Tenet vN` numbers represent distinct **algorithmic generations**, not incremental runs (`v1`: Dual-Anchor Bi-Projected Lancet + FP8 readiness on non-thinking Instruct models; `v2`: Phase-Segmented Tri-Anchor `<think>` surgery + Length-Calibrated GRPO + CKA-GGUF + Block-16 Quant-Snap & Router-Frozen MoE across `Qwen3`, `Mistral-Small-3.1`, and `Gemma-4 MoE`). Eval-harness updates and new quantization exports (`CKA-Q4_K_M`, `FP8`, `NVFP4`) update existing `vN` releases in place; recipe tuning within a generation uses point revisions (`v2.1`, `v2.2`); `v3` is reserved for the next verified multi-family algorithmic leap.

---

## Overview

**Tenet Lab** (originated as **Small Model Lab**, CLI/package `sml`) is an end-to-end language model systems and applied research project structured into eight tightly coupled sprints:

1. **Sprint 1 — Core Transformer, Custom `@triton.jit` Kernels & GPU Pretraining (`Phase I Lean`):** Modern Llama/Qwen-compatible decoder (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, intra-document causal masking, `KV-cache`) + custom `@triton.jit` fused `RMSNorm`, `SwiGLU`, and chunked `CrossEntropy` forward/backward kernels (`src/sml/kernels/`), validated against OpenAI `gpt2` (`124M`) logits (`< 1e-4`), trained with a 32k BPE tokenizer on `FineWeb-Edu` (`4.0B` tokens with Warmup–Stable–Decay and a `1.5B` Chinchilla-optimal cooldown branch).
2. **Sprint 2 — Trainable Analytical & Incident RL Environment (`StatefulToolEnv`):** `verifiers`-compatible multi-turn environment (`StatefulToolEnv` + `load_environment`) built on `dbt Core + DuckDB + sqlglot` with isolated MCP tools, multi-seed counterfactual fault injection (`product_bug`, `data_pipeline_artifact`, `mix_shift` + confounders), and a Gated Multiplicative `Rubric` (`R_total = R_outcome · R_evidence · R_process`).
3. **Sprint 3 — Adaptation, Distillation & Multi-Turn RL (`GRPO`):** Controlled comparison on `1.5B–3B` open weights: Prompting vs. Hybrid RAG vs. LoRA/SFT vs. Teacher (`30B+`) Rejection-Sampled Distillation (via the hardened OpenRouter gateway in `src/sml/gateway/` with strict recursive secret/raw-row egress guards) and multi-turn `GRPO`, including a pre-registered reward-hacking ablation (`H5`: Gated Multiplicative vs. Naive Additive reward).
4. **Sprint 4 — Specialist Surgery & Production Inference (`vLLM`):** Layer-wise representation drift (`CKA`), CKA-guided mixed-precision allocation (RCO principle), rank-1 subspace weight surgery (W' = W - r(r^T W)), quantization sensitivity (`FP8` / `INT4 AWQ`), layer pruning, component-level Gated Rubric degradation audit (H7, H8), and load-tested serving economics (`TTFT`, `TPOT`, `req/s`, `$/1k` incidents Pareto frontier).
5. **Sprint 5 — Lancet Dual-Anchor Soft-SCR Surgery (`H9`):** Bi-projected, norm-preserving, multi-direction subspace surgery (`src/sml/surgery/`): Modified Gram-Schmidt biprojection of refusal/slop directions against the reasoning anchor (float32), joint projection from `self_attn.o_proj` / `mlp.down_proj` with per-row L2-norm rescaling, and Signal-to-Collateral-Ratio (`SCR`) layer selection ([ADR-005](docs/decisions/005-lancet-biprojected-surgery.md)).
6. **Sprint 6 — Tenet Subspace-Safe Unslop GRPO (`H10`):** Gated unslop `GRPO` (Gated Multiplicative `Rubric` + slop/refusal penalty) on surgically edited weights, with LoRA adapters merged only through the orthogonal projection `I - U U^T` plus row-norm preservation, so RL updates cannot re-leak into the ablated subspace — released as the `Tenet` model family ([ADR-005](docs/decisions/005-lancet-biprojected-surgery.md)).
7. **Sprint 7 — FP8 Provider Survival & Readiness (`H11`):** Certification that edited checkpoints survive OpenRouter-style `FP8` (E4M3) quantization: FP8 Subspace Re-Leakage measurement, Per-Row vs. Per-Tensor E4M3 scale comparison, and the `Row Crest-Factor < 25.0` provider readiness gate ([ADR-005](docs/decisions/005-lancet-biprojected-surgery.md)).
8. **Sprint 8 — Cross-Domain Subspace-Safe Adapter Merge (`H12`):** TIES / Gram-Schmidt subspace-safe composition of independently trained GRPO adapters (`Delta W_SQL` + `Delta W_unslop`): orthogonal projection of `Delta W_unslop` against `U_{refusal/slop}` and the top-k left singular vectors `U_SQL` of `Delta W_SQL`, TIES trim/sign-election/disjoint merge, row-norm restoration, and the `H11` FP8 readiness gate ([ADR-007](docs/decisions/007-cross-domain-subspace-safe-adapter-merge.md)).

> **Zero Placeholder Numbers Policy:** No synthetic or estimated numbers are ever written into result tables. All tables in `reports/` are compiled directly from machine-readable experiment manifests in `results/<run_id>/` via `sml report`.

---

## Pre-Registered Hypotheses & Status (Sprints 1–11)

| ID | Hypothesis | Pre-Registered Acceptance Criterion | Status |
| :--- | :--- | :--- | :--- |
| **H1** | **Mathematical & Causal Equivalence:** Custom `Transformer` in `gpt2-compat` mode matches `GPT2LMHeadModel` (`124M`) logits; naive `O(T^2)` GQA matches `SDPA`; `KV-cache` matches full forward; intra-document mask blocks cross-`<\|eos\|>` leakage. | `max_abs_logit_diff < 1e-4` (`fp32`); cross-doc leakage `== 0.0` | **Verified** (`sdpa=3.16e-6`, `naive=3.81e-6`, `kv_cache=2.80e-6`, `leakage=0.0` in [`results/h1_gpt2_parity/`](results/h1_gpt2_parity/final_metrics.json)) |
| **H2** | **GPU Throughput & `@triton.jit` Waterfall:** Cumulative system optimizations (`fp32 naive` → `fp32 sdpa` → `bf16 sdpa` → `bf16 fused kernels` → `torch.compile`) yield measured throughput gains and high hardware utilization. | Measured 5-stage waterfall in `reports/sprint1/`; `MFU >= 30%` on `1x H100 SXM` (`1.40x` local MPS speedup verified) | **Verified** (`472,080.81 tok/s`, `MFU=34.22%`, `13.55x` vs `fp32` naive `34,850.26 tok/s` on `1x H100 SXM` in [`results/h2_mfu_waterfall/`](results/h2_mfu_waterfall/final_metrics.json)) |
| **H3** | **Chinchilla Optimal (`1.5B`) vs. Overtrained (`4.0B`) WSD Cooldown:** Overtraining `sml-100m` (`75.52M` non-embedding params) past `20x` tokens/param via WSD cooldown branching improves `val bpb` and zero-shot accuracy at identical inference cost. | Improvement exceeds `2σ` of 3-seed `sml-xs` noise floor | **Verified** (`val_bpb=1.039992` at `4.00B` tokens vs `1.044582` `1.5B` branch at step `2861`, `Δ=-0.004590`, `445,067.74 tok/s`, `MFU=32.27%` in [`results/h3_wsd_100m/`](results/h3_wsd_100m/final_metrics.json)) |
| **H4** | **In-Context Induction Phase Transition:** Log-spaced checkpoints reveal an abrupt phase transition in intra-document `loss(late) - loss(early)` and synthetic sequence copying. | Phase transition localized across 15 log-spaced checkpoints | **Verified** (`15` log-spaced checkpoints, `max_window_gain_share=1.0`, `copy_acc=0.25`, `loss_delta=-0.179115` in [`results/h4_induction_dynamics/`](results/h4_induction_dynamics/final_metrics.json)) |
| **H5** | **Gated Multiplicative vs. Naive Additive Rubric (Anti-Reward-Hacking):** `G_evidence · R_correctness` prevents premature guessing and unsafe SQL during multi-turn `GRPO`. | `oracle_gated > 1.0`, `lazy_gated == 0.0` vs. `lazy_naive == 0.75`, and `H5_skill_difference >= 0.15` | **Verified** (`Δskill = +0.8625` in [`results/h5_env_verification/`](results/h5_env_verification/final_metrics.json) & [`results/h6_adaptation_comparison/`](results/h6_adaptation_comparison/adaptation_comparison.json); `Qwen2.5-3B-Instruct` MPS `Δskill = +0.1684` in [`results/h6_qwen3b_mps/`](results/h6_qwen3b_mps/adaptation_comparison.json)) |
| **H6** | **Context Enrichment (`Hybrid RAG`) vs. Weight Updates (`Distillation + GRPO`):** Hybrid BM25 + char 3-gram RRF resolves knowledge-bound table lookups (`>= 85%`), while weight updates dominate skill-bound multi-turn hypothesis elimination (`>= 20 pp` gain). | `rag_knowledge_score >= 0.85` AND `H6_skill_difference >= 0.20` | **Verified** (`rag_kb = 1.00`, `Δskill = +0.5050` in [`results/h6_adaptation_comparison/`](results/h6_adaptation_comparison/adaptation_comparison.json); `Qwen2.5-3B-Instruct` MPS `rag_kb = 1.00`, `6_grpo_gated_rl = 1.0000` vs `3_hybrid_rag = 0.5600`, `Δskill = +0.4400` in [`results/h6_qwen3b_mps/`](results/h6_qwen3b_mps/adaptation_comparison.json)) |
| **H7** | **Representation-Constrained Compression (CKA-Guided Mixed-Precision RCO):** Allocating 8-bit precision to the top-k highest per-block CKA-drift layers reduces KL divergence by `>= 20%` vs. an equal-bit bottom-k control (`5.33 bits/weight`). | `h7_kl_reduction_ratio >= 0.20` at identical average bits/weight | **Verified** (`73.33%` KL reduction: `0.005168` vs. `0.019381` at `5.33 bits` in [`results/h7_h8_surgery_pareto/`](results/h7_h8_surgery_pareto/surgery_pareto.json); `Qwen2.5-3B-Instruct` MPS `30.10%` KL reduction: `0.165203` vs. `0.236338` in [`results/h7_h8_qwen3b_pareto/`](results/h7_h8_qwen3b_pareto/surgery_pareto.json)) |
| **H8** | **Rank-1 Subspace Projection (`W' = W - r(r^T W)`):** Closed-form orthogonal projection zeroes out targeted residual stream writes along contrastive direction `r` (`< 1e-5`). | `max_ortho_residual < 1e-5` AND `layer_write_proj_after < 1e-5` | **Verified** (`ortho_residual = 0.0`, `write_proj: 7.91e-3 → 2.57e-6` in [`results/h7_h8_surgery_pareto/`](results/h7_h8_surgery_pareto/surgery_pareto.json); `Qwen2.5-3B-Instruct` MPS `ortho_residual = 9.06e-4`, `write_proj: 59.55 → 0.0701`, `kl_collateral = 0.0961` in [`results/h7_h8_qwen3b_pareto/`](results/h7_h8_qwen3b_pareto/surgery_pareto.json)) |
| **H9** | **Lancet Dual-Anchor Soft-SCR Surgery:** Bi-projected (Modified Gram-Schmidt, float32), norm-preserving, multi-direction subspace surgery with Signal-to-Collateral-Ratio layer selection removes refusal/slop writes without colliding with the reasoning anchor. | `anchor_orthogonality_error < 1e-6`, `max_row_norm_drift < 1e-5`, `target_suppression_ratio >= 0.90`, `h9_kl_reduction_ratio >= 0.35` | **Verified** (`sml-nano`: `h9_verified=true`, `h9_kl_reduction_ratio=0.6092` in [`results/h9_biprojected_surgery/`](results/h9_biprojected_surgery/final_metrics.json); `Qwen2.5-3B` MPS `h9_kl_reduction_ratio=0.9941`, `target_suppression=0.9694` in [`results/h9_qwen3b_mps/`](results/h9_qwen3b_mps/final_metrics.json); `Llama-3.1-8B` H100 `h9_kl_reduction_ratio=0.8989`, `target_suppression=0.9930`, `ortho=1.0e-08` in [`results/h9_llama8b_h100/`](results/h9_llama8b_h100/final_metrics.json); `Mistral-Nemo-12B` H100 `h9_kl_reduction_ratio=0.9672`, `target_suppression=0.9914`, `ortho=1.0e-08` in [`results/h9_nemo12b_h100/`](results/h9_nemo12b_h100/final_metrics.json)) |
| **H10** | **Tenet Subspace-Safe Unslop GRPO:** Gated unslop `GRPO` on surgically edited weights, with LoRA merged only through the orthogonal projection `I - U U^T` plus row-norm preservation, drives slop/refusal to near zero without re-leaking into the ablated subspace. | `slop_and_refusal_rate <= 0.02` (`nano`) / `<= 0.10` (`3B+`), `h10_gated_vs_naive_gain >= 0.15`, `post_merge_suppression_retention >= 0.90` | **Verified** (`sml-nano`: `slop=0.0150`, `gain=+0.2276` in [`results/h10_unslop_grpo/`](results/h10_unslop_grpo/final_metrics.json); `Qwen2.5-3B` MPS `slop=0.0909`, `gain=+0.1602`, `releakage=0.0690` in [`results/h10_qwen3b_mps/`](results/h10_qwen3b_mps/final_metrics.json); `Llama-3.1-8B` H100 `slop=0.0410`, `gain=+0.2489`, `releakage=0.0726` in [`results/h10_llama8b_h100/`](results/h10_llama8b_h100/final_metrics.json); `Mistral-Nemo-12B` H100 `slop=0.0239`, `gain=+0.4292`, `releakage=0.1282` in [`results/h10_nemo12b_h100/`](results/h10_nemo12b_h100/final_metrics.json)) |
| **H11** | **FP8 Re-Leakage & Crest-Factor Provider Readiness:** Norm-preserving Lancet surgery keeps edited checkpoints provider-ready under `FP8` (E4M3) quantization — far lower FP8 subspace re-leakage than naive surgery, mode-insensitive KL across Per-Row vs. Per-Tensor E4M3 scales, and `Row Crest-Factor < 25.0`. | `fp8_retention_ratio >= 0.96` (`nano`) / `>= 0.95` (`3B+`), `h11_fp8_kl_reduction_ratio >= 0.25`, `max_row_crest_factor < 25.0`, `provider_ready=true` (7/7 checks) | **Verified** (`sml-nano`: `retention=0.9997`, `kl_red=0.4558` in [`results/h11_provider_fp8_readiness/`](results/h11_provider_fp8_readiness/final_metrics.json); `Qwen2.5-3B` MPS `retention=0.9541`, `kl_red=0.5185`, `crest=17.2706`, `provider_ready=true` in [`results/h11_qwen3b_mps/`](results/h11_qwen3b_mps/final_metrics.json); `Llama-3.1-8B` H100 `retention=0.9936`, `kl_red=0.4265`, `crest=6.3244`, `provider_ready=true` in [`results/h11_llama8b_h100/`](results/h11_llama8b_h100/final_metrics.json); `Mistral-Nemo-12B` H100 `retention=0.9678`, `kl_red=0.4892`, `crest=10.6508`, `provider_ready=true` in [`results/h11_nemo12b_h100/`](results/h11_nemo12b_h100/final_metrics.json)) |
| **H12** | **Cross-Domain Subspace-Safe Adapter Merge (TIES + Lancet):** TIES trim/sign-election/disjoint merge of independently trained GRPO adapters (`Delta W_SQL` + `Delta W_unslop`), with `Delta W_unslop` orthogonally projected against `U_{refusal/slop}` and the top-k left singular vectors `U_SQL` of `Delta W_SQL`, preserves both SQL-incident skill and unslop behavior without re-leaking into the ablated subspace, and keeps the merged checkpoint provider-ready under `FP8`. | `sql_incident_retention_ratio >= 0.95`, `unslop_retention_ratio >= 0.95`, `slop_and_refusal_rate <= 0.02` (`nano`) / `<= 0.10` (`3B+`), `subspace_safe_merge_releakage < 1e-5`, `ties_biprojected_vs_naive_merge_gain >= 0.15`, `max_row_crest_factor < 25.0`, `provider_ready=true` | **Verified** (`sml-nano`: `gain=+0.4995`, `crest=1.2184` in [`results/h12_subspace_safe_merge/`](results/h12_subspace_safe_merge/final_metrics.json); `Qwen2.5-3B` MPS `gain=+0.7848`, `sql=1.0`, `unslop=1.0`, `crest=17.2706` in [`results/h12_qwen3b_mps/`](results/h12_qwen3b_mps/final_metrics.json); `Llama-3.1-8B` H100 `gain=+0.7965`, `sql=0.9799`, `unslop=1.0`, `crest=6.3244` in [`results/h12_llama8b_h100/`](results/h12_llama8b_h100/final_metrics.json); `Mistral-Nemo-12B` H100 `gain=+0.7806`, `sql=1.0`, `unslop=1.0`, `crest=10.6510` in [`results/h12_nemo12b_h100/`](results/h12_nemo12b_h100/final_metrics.json)) |
| **H13** | **Layer-Streaming Lancet Surgery & Subspace-Safe Merge (`32B–70B`):** Shard-by-shard `safetensors` streaming (`float32` Modified Gram-Schmidt + TIES merge + per-row L2-norm restoration) achieves numerical equivalence with in-memory surgery/merge within `< 16 GB` peak RAM and `< 15 GB` local scratch disk on `1x H100 80GB SXM`. | `streaming_surgery_max_abs_diff < 1e-6`, `streaming_merge_max_abs_diff < 1e-6`, `peak_memory_gb < 16.0`, `max_row_crest_factor < 25.0`, `provider_ready=true` | **Verified** (`sml-nano`: `surgery_diff=7.45e-09`, `merge_diff=0.0`, `crest=1.1926` in [`results/h13_streaming_shard_surgery/`](results/h13_streaming_shard_surgery/final_metrics.json); `Tenet-32B-Qwen2.5-Unslop-v1` `crest=3.68`, `8/8` FP8 readiness) |
| **H14** | **CKA-Guided Mixed-Precision `GGUF` Quantization Survival (`Tenet-8B` / `Tenet v2`):** Protecting top-CKA-drift layers (`Q8_0`/`Q6_K`) in `CKA-Q4_K_M` / `i1-CKA-Q4_K_M` (`5.29` avg bpw) halves surgical subspace re-leakage amplification and lowers KL vs. `F16` compared to static `Q4_K_M` and `i1-IQ4_XS`. | `subspace_releakage_amplification` below static `Q4_K_M`, lower `mean_kl_vs_f16`, `sql_readonly_rate == 1.0`, `slop_and_refusal_rate <= 0.15` | **Verified** (`Tenet-8B-Llama3.1-Unslop-v1`: `3.012x` amp vs `6.402x` `Q4_K_M` / `12.374x` `i1-IQ4_XS`; Mean KLD `0.02048` / `0.01656` vs `0.02479`; `sql_readonly_rate=1.0` in [`results/h14_gguf_releakage/`](results/h14_gguf_releakage/final_metrics.json)) |
| **H15** | **Phase-Segmented Tri-Anchor Lancet (`Tenet v2`):** `PhaseSegmenter` (`<think>...</think>`) + 2-pass `float32` MGS Tri-Anchor basis ($a_{\text{cot}}, a_{\text{math\_logic}}, a_{\text{tool\_sql}}$) + post-`</think>` target extraction + CoT-weighted SCR preserve reasoning-channel integrity and FP8 readiness. | `tri_anchor_orthogonality_error < 1e-6`, `biprojected_orthogonality_error < 1e-6`, `max_row_crest_factor < 25.0`, `h15_verified=true` | **Verified** (`Tenet-4B-Qwen3-Unslop-v2` & `Tenet-8B-Qwen3-Unslop-v2`: `10/10` gates, `crest=4.3688` & `fp8_releakage=2.7243%` on `8B` in [`results/tenet_releases/Tenet-8B-Qwen3-Unslop-v2/`](results/tenet_releases/Tenet-8B-Qwen3-Unslop-v2/foundry_summary.json)) |
| **H16** | **Dual-Window Length-Calibrated & Artifact-Gated GRPO (`Tenet v2`):** `G_cot_integrity`, asymmetric log-normal `G_length_cal`, `G_antislop_density`, and verifiable `R_info_density` eliminate `length_collapse` and `<think>` overthinking while preserving technical artifact completeness. | `cot_tag_integrity_rate >= 0.995`, `length_collapse_rate < 0.01`, `slop_and_refusal_rate <= 0.02`, `artifact_completeness_rate >= 0.99`, `h16_verified=true` | **Verified** (`Tenet-4B-Qwen3-Unslop-v2` & `Tenet-8B-Qwen3-Unslop-v2` `i1-CKA-Q4_K_M` / `f16`: `cot_tag_integrity=1.0`, `length_collapse=0.0`, `slop_and_refusal=0.0` in [`ADR-010`](docs/decisions/010-tenet-v2-thinking-surgery-and-length-calibrated-grpo.md)) |
| **H17** | **24-Seed Fight Card Arena House & Peer Eval Gate (`Tenet v2`):** End-to-end mechanical verification on `24` held-out Fight Card seeds (`f16`, `cka_q4_k_m`, `i1_cka_q4_k_m`) plus read-only SQL checks certifies zero release regressions before Hub publish. | `arena_house_hp >= 95.0`, `house_artifact_classification_rate == 1.0` (`24/24`), `unslop_pass_rate == 1.0`, `sql_readonly_rate == 1.0` (`6/6`) | **Verified** (`Tenet-4B-Qwen3-Unslop-v2` & `Tenet-8B-Qwen3-Unslop-v2`: `100.0 HP`, `24/24` across all 3 GGUF variants, `6/6` SQL, `8B` `i1_cka_q4_k_m` Mean KLD `0.01314` in [`results/tenet_releases/Tenet-8B-Qwen3-Unslop-v2/`](results/tenet_releases/Tenet-8B-Qwen3-Unslop-v2/foundry_summary.json)) |
| **H18** | **NVFP4 / Block-16 Quant-Snap Lancet:** Quantization-snap-aware Lancet biprojection on NVFP4 (block-16 scaling) keeps NVFP4 subspace re-leakage below naive surgery without reasoning collateral. | `nvfp4_releakage_reduction >= 0.20`, `int4_b32_releakage_reduction >= 0.20` | **Partial / Pilot** (`Qwen/Qwen3-4B`, 8 surgery layers: `26.15%` NVFP4 re-leakage reduction, `28.72%` INT4-b32 re-leakage reduction; synthetic controlled audit `32.58%` NVFP4 re-leakage reduction, `17.05%` NVFP4 KL reduction, while the strict `max_block16_crest_factor = 3.3695 > 3.20` gate keeps `h18_verified = false` and motivates scaling in [`results/h18_nvfp4_quant_snap/qwen3_4b_audit.json`](results/h18_nvfp4_quant_snap/qwen3_4b_audit.json)) |
| **H19** | **Loop-Unrolled SVD Biprojection Lancet:** Unrolled per-layer SVD biprojection on a weight-tied looped transformer preserves SCR layer selection quality and anchor orthonormality without unrolling the recurrent weight matrix. | `scr_biprojected > scr_naive > scr_concatenated_anchor`, `orthonormality_error < 1e-6` | **Verified** (`Tenet-Nano-6L-Looped-H19`, 3-repeat weight-tied looped transformer: `SCR=2.4339` biprojected vs `2.4030` naive vs `1.6729` concatenated-anchor, orthonormality error `1.19e-7` in [`results/h19_loop_unrolled_lancet/h19_summary.json`](results/h19_loop_unrolled_lancet/h19_summary.json)) |
| **H20** | **Router-Frozen MoE & Cross-Family Streaming Quant-Snap Surgery (`Tenet v2`):** `apply_moe_lancet_recipe_streaming` (frozen router gate, fused `gate_up_proj`/`down_proj` streaming projection) + 4-iter `H18` Quant-Snap across `Mistral-Small-3.1-24B` & `Gemma-4-26B-A4B MoE` lifts instruction-following without reasoning collateral or router drift. | Mistral-24B: IFEval Prompt-Strict gain `> +8 pp` with `< 1.1 pp` MMLU-Pro/GSM8K/GPQA drift and `Row Crest-Factor < 25.0`; Gemma-4 MoE: holds 0-shot IFEval/MATH-500 while documenting 5-shot CoT router/expert sensitivity | **Partial / Pilot** (Mistral-24B `+9.80 pp` IFEval & `<1.1 pp` MMLU-Pro/GSM8K/GPQA drift; Gemma-4 MoE holds `86.32%` IFEval / `75.00%` MATH-500 0-shot while exposing 5-shot CoT router/expert sensitivity) |

---

## Verified Model Configurations (Sprint 1)

| Model | `d_model` | Layers | Q / KV Heads | `d_ff` | Vocab | Non-Emb Params | Total Params (Tied) | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sml-nano` | 256 | 4 | 4 / 2 | 768 | 32,768 | `3,148,032` | `11,536,640` | Local CPU/MPS unit tests & TinyStories smoke check |
| `sml-xs` | 512 | 8 | 8 / 4 | 1,408 | 32,768 | `23,601,664` | `40,378,880` | 24 GB GPU pilot & 3-seed noise floor (`0.4B` tokens) |
| **`sml-100m`** | **768** | **12** | **12 / 4** | **2,048** | **32,768** | **`75,516,672`** | **`100,682,496`** | **Primary Sprint 1 model (`1.5B` branch + `4.0B` full)** |
| `gpt2-124m` | 768 | 12 | 12 / 12 | 3,072 | 50,257 | `85,842,432` | `124,439,808` | Reference parity target for `H1` (`gpt2-compat`) |

---

## Key Measured Results Summary (`H1–H12`)

* **Sprint 1 (`1x NVIDIA H100 80GB SXM` — `sml-100m`):** 5-stage `@triton.jit` + `torch.compile` waterfall (`34,850` → `472,081 tok/s`, **`34.22% MFU`**, **`13.55x` speedup**), `4.0B`-token WSD pretraining on `FineWeb-Edu` (`445,068 tok/s`, `32.27% MFU`, `val_bpb = 1.0400` vs `1.0446` `1.5B` Chinchilla cooldown branch), and `H4` induction phase transition across 15 log-spaced checkpoints.
* **Sprints 2–4 (`Apple M4 Pro 48GB MPS` — `Qwen/Qwen2.5-3B-Instruct` & `sml-nano`):** Counterfactual `DuckDB + sqlglot` `StatefulToolEnv` with Gated Multiplicative Rubric (`H5`: `oracle = 1.12`, `lazy_guess = 0.00` vs `0.75` naive additive), `Hybrid RAG` (`100%` knowledge-bound recall) vs. `LoRA SFT + Multi-Turn Token-Level GRPO` (`H6`: `+44.0 pp` skill-bound gain on `Qwen2.5-3B-Instruct`), Blockwise CKA-Guided `4/8-bit` RCO Quantization (`H7`: `30.10%` KL reduction on `Qwen2.5-3B-Instruct`, `73.33%` on `sml-nano`), and Closed-Form Rank-1 Subspace Projection (`H8`: `write_proj: 59.55 → 0.0701`).
* **Sprints 5–7 (`Apple M4 Pro 48GB MPS` — `Qwen/Qwen2.5-3B-Instruct` & `sml-nano`; `Lancet` engine & `Tenet` family `Tenet-3B-Qwen2.5-Unslop-v1`, `Tenet-8B-Llama3.1-Unslop-v1`, `Tenet-12B-Nemo-Unslop-v1`):** `H9` Lancet Dual-Anchor Soft-SCR Surgery (`h9_kl_reduction_ratio=0.6092` on `sml-nano`, `0.9941` on `Qwen2.5-3B-Instruct` with `naive_kl_collateral=3.7607` vs `lancet_kl_collateral=0.0220`), `H10` Tenet Subspace-Safe Unslop GRPO (`slop_and_refusal_rate=0.0150` on `sml-nano`, `0.0909` on `Qwen2.5-3B-Instruct` (vs `0.1542` additive GRPO, `+16.02 pp` gated gain, `+35.98 pp` post-surgery gain; `subspace_safe_lora_releakage=0.0690` vs `unprojected_lora_releakage=0.8300`, `post_merge_suppression_retention=1.0`)), and `H11` FP8 Provider Survival & Readiness (`fp8_subspace_releakage_lancet=0.0310` vs `naive=0.4996` on `sml-nano`, `0.0270` vs `0.5003` on `Qwen2.5-3B-Instruct`; `max_row_crest_factor=17.2706 < 25.0`, `provider_ready=true` 7/7 checks on both).
* **Sprint 8 (`Apple M4 Pro 48GB MPS` — `Qwen/Qwen2.5-3B-Instruct` & `sml-nano`):** `H12` Cross-Domain Subspace-Safe Adapter Merge (`ties_biprojected_vs_naive_merge_gain=+0.4995` on `sml-nano` and `+0.7848` on `Qwen2.5-3B-Instruct`, `sql_incident_retention_ratio=1.0`, `unslop_retention_ratio=1.0`, `slop_and_refusal_rate=4.57e-08` / `5.55e-07`, `subspace_safe_merge_releakage=1.25e-07` / `4.71e-07`, `max_row_crest_factor=1.2184` / `17.2706 < 25.0`, `provider_ready=true` on both `sml-nano` and `Qwen2.5-3B-Instruct`).
* **GGUF Survival Audit (`H14` — `Apple M4 Pro 48GB`, `Tenet-8B-Llama3.1-Unslop-v1`) & 3-Way External Evals (`3B`/`8B`):** CKA-guided mixed-precision `GGUF` (`cka_q4_k_m` / `i1_cka_q4_k_m`, `5.290` avg bpw) cuts surgical subspace re-leakage amplification in half vs. static `Q4_K_M` (`3.012x` vs. `6.402x`, `12.374x` on `i1-IQ4_XS`) and lowers mean KLD vs. `F16` (`0.02048` / `0.01656` vs. `0.02479` on `Q4_K_M`) while preserving `sql_readonly_rate=1.0` ([`results/h14_gguf_releakage/`](results/h14_gguf_releakage/final_metrics.json), [ADR-009](docs/decisions/009-gguf-imatrix-vs-cka-survival.md)). Three-way external evals (`Base` vs. `Naive Abliteration` vs. `Tenet v1`) confirm `0.0` `slop_and_refusal_rate` and `1.0` `unslop_pass_rate` on both `3B` (`ifeval_strict=0.600`, `+10.0 pp` vs. Naive) and `8B` (`gsm8k_cot_flexible=0.733`, `+6.7 pp` vs. Base) ([`results/external_evals/`](results/external_evals/Tenet-8B-Llama3.1-Unslop-v1.json)).
* **Independent Third-Party Eval (`Model Fight Card 16`, `2026-11-20`):** Double championship sweep — `Tenet-3B-Qwen2.5-Unslop-v1` won the **Minors Division Championship outright at `90 HP`** (`22/22` classifiable artifacts, `21/22` House escapes, `4/4` APEX seeds) and `Tenet-8B-Llama3.1-Unslop-v1` won the **Majors Division Championship outright at `50 HP`** (`24/24` classifiable artifacts, `20/24` House escapes, `4/4` APEX seeds), with the `8B` additionally authoring the only fresh gate-valid Peer Eval of the night (`7/7` clean hits inflicted) — [Model Fight Card 16](https://onlymodels.gg/cards/2026-11-20.html).

### `Tenet v2` Release Gate Matrix (`4B` / `8B` / `14B` / `32B`)

| Release | Base Model | Max Crest (`< 25`) | FP8 Re-Leakage (`< 3%`) | `F16` Arena HP | `CKA-Q4_K_M` HP | `i1-CKA-Q4_K_M` HP | Gates | Recommended Quant |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| [`Tenet-4B-Qwen3-Unslop-v2`](results/tenet_releases/Tenet-4B-Qwen3-Unslop-v2/foundry_summary.json) | `Qwen/Qwen3-4B` | 3.1766 | 2.4946% | **100.0** (`24/24`) | **100.0** (`24/24`) | **100.0** (`24/24`) | **10/10** | `i1-CKA-Q4_K_M` / `BF16-FP8` |
| [`Tenet-8B-Qwen3-Unslop-v2`](results/tenet_releases/Tenet-8B-Qwen3-Unslop-v2/foundry_summary.json) | `Qwen/Qwen3-8B` | 4.3688 | 2.7243% | **100.0** (`24/24`) | **100.0** (`24/24`) | **100.0** (`24/24`) | **10/10** | `i1-CKA-Q4_K_M` / `BF16-FP8` |
| [`Tenet-14B-Qwen3-Unslop-v2`](results/tenet_releases/Tenet-14B-Qwen3-Unslop-v2/foundry_summary.json) | `Qwen/Qwen3-14B` | 3.5974 | 2.5898% | 93.75 (`22/24`) | 91.67 (`21/24`, `6/6` SQL) | 95.83 (`23/24`, `0/6` SQL) | 5/10 | `BF16-FP8` (`lm-eval` winner) |
| [`Tenet-32B-Qwen3-Unslop-v2`](results/tenet_releases/Tenet-32B-Qwen3-Unslop-v2/foundry_summary.json) | `Qwen/Qwen3-32B` | 4.3593 | 2.5579% | **100.0** (`24/24`) | **100.0** (`24/24`, `6/6` SQL) | 97.92 (`23/24`*) | 9/10 | **`CKA-Q4_K_M`** / `BF16-FP8` |

> *Note on `Tenet-32B-Qwen3-Unslop-v2` (`9/10` gates):* `F16` and pure `CKA-Q4_K_M` (`4.896 bpw`) both score **`100.0 HP` (`24/24` artifacts, `100%` `<think>` integrity, `100%` unslop, `6/6` SQL)**; the single failed gate is `house_artifact_classification_rate[i1_cka_q4_k_m]` (`97.92 HP`, `23/24` seeds when `imatrix` is overlaid, missing `house-edge-leap-years`), so pure `CKA-Q4_K_M` or `BF16/FP8` is the recommended 32B-v2 deployment format.

### 3-Way External Benchmarks (`lm-evaluation-harness` + 12-Seed Unslop Audit, `results/external_evals/`)

| Release | Base Model | `limit` | `ifeval_strict` (Base / Naive / Tenet) | `gsm8k_cot_flexible` (Base / Naive / Tenet) | `unslop_pass_rate` (Base / Naive / Tenet) | `slop_refusal` (Base / Naive / Tenet) |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: |
| [`Tenet-3B-Qwen2.5-Unslop-v1`](results/external_evals/Tenet-3B-Qwen2.5-Unslop-v1.json) | `Qwen2.5-3B-Instruct` | 40 | 0.525 / 0.500 / **0.600** | 0.525 / 0.350 / 0.450 | 1.000 / 0.958 / **1.000** | 0.000 / 0.042 / **0.000** |
| [`Tenet-8B-Llama3.1-Unslop-v1`](results/external_evals/Tenet-8B-Llama3.1-Unslop-v1.json) | `Llama-3.1-8B-Instruct` | 30 | 0.633 / 0.667 / 0.600 | 0.667 / 0.833 / **0.733** | 1.000 / 1.000 / **1.000** | 0.000 / 0.000 / **0.000** |
| [`Tenet-12B-Nemo-Unslop-v1`](results/external_evals/Tenet-12B-Nemo-Unslop-v1.json) | `Mistral-Nemo-Instruct-2407` | 40 | 0.525 / 0.600 / 0.550 | 0.725 / 0.825 / 0.775 | 0.958 / 0.917 / **1.000** | 0.042 / 0.083 / **0.000** |
| [`Tenet-32B-Qwen2.5-Unslop-v1`](results/external_evals/Tenet-32B-Qwen2.5-Unslop-v1.json) | `Qwen2.5-32B-Instruct` | 40 | 0.800 / 0.775 / 0.750 | 0.725 / 0.675 / 0.675 | 1.000 / 1.000 / **1.000** | 0.000 / 0.000 / **0.000** |

### Full vLLM OpenEvals — `Tenet v2` (canonical, `suite=full`, `think_end_token=</think>`, `max_gen_toks=8192`, `0.00%` unclosed `<think>`, `n=3,566 tasks/arm`)

| Release | Base Model | GSM8K Strict (Base / Naive / Tenet) | MATH-500 (Base / Naive / Tenet) | IFEval Prompt-Strict (Base / Naive / Tenet) | GPQA Diamond (Base / Naive / Tenet) | MMLU-Pro (Base / Naive / Tenet) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [`Tenet-4B-Qwen3-Unslop-v2`](results/external_evals/Tenet-4B-Qwen3-Unslop-v2.json) | `Qwen3-4B` | **88.78% / 85.97% / 88.93%** (`+2.96 pp` vs Naive; Flex `77.10% / 79.15% / 78.01%`) | 70.60% / 70.20% / **70.20%** | 78.56% / 81.33% / 80.59% (`+2.03 pp` vs Base; Inst `85.01% / 87.17% / 85.97%`) | 34.34% / 34.34% / **34.34%** | 57.44% / 55.75% / **56.35%** |
| [`Tenet-8B-Qwen3-Unslop-v2`](results/external_evals/Tenet-8B-Qwen3-Unslop-v2.json) | `Qwen3-8B` | Flex **90.30% / 90.98% / 90.90%** (`+0.61 pp` vs Base; Strict `90.14% / 90.37% / 89.31%`) | 70.60% / 70.40% / **72.20%** (`+1.60 pp` vs Base, `+1.80 pp` vs Naive) | 82.07% / 82.62% / 80.78% (Inst `87.53% / 88.01% / 86.81%`) | 39.90% / 38.38% / **40.40%** (`+0.50 pp` vs Base, `+2.02 pp` vs Naive) | 59.42% / 60.91% / 59.72% (`+0.30 pp` vs Base) |
| [`Tenet-32B-Qwen3-Unslop-v2`](results/external_evals/Tenet-32B-Qwen3-Unslop-v2.json) | `Qwen3-32B` | Flex **91.74% / 84.76% / 92.19%** (`+0.45 pp` vs Base, `+7.43 pp` vs Naive; Strict `90.07% / 85.60% / 80.97%`) | 77.60% / 71.20% / **74.80%** (`+3.60 pp` vs Naive) | 84.66% / 82.26% / 84.47% (`+2.21 pp` vs Naive; Inst `89.21% / 87.77% / 88.85%`) | 38.89% / 37.88% / **40.40%** (`+1.51 pp` vs Base, `+2.52 pp` vs Naive) | 71.03% / 62.90% / 65.48% (`+2.58 pp` vs Naive) |
| [`Tenet-24B-Mistral3.1-Unslop-v2`](results/external_evals/Tenet-24B-Mistral3.1-Unslop-v2.json) | `Mistral-Small-3.1-24B-Instruct-2503` | Flex **89.84%** (Tenet) | — | **65.99% Base → 75.79% Tenet (`+9.80 pp`)**; Inst-Strict `82.73%` (`+7.31 pp`) | 39.39% (Tenet, `<1.1 pp` drift) | 63.49% (Tenet, `<1.1 pp` drift) |
| [`Tenet-26B-A4B-Gemma4-Unslop-v2`](results/external_evals/Tenet-26B-A4B-Gemma4-Unslop-v2.json) | `google/gemma-4-26B-A4B-it` (MoE) | 0-shot: GSM8K n/a; **5-shot `<|thought|>` sensitivity: 66.34% → 21.83%** (MoE router/expert finding) | **75.00%** (0-shot) | **86.32%** Strict (`90.41%` Inst-Strict, 0-shot Chat) | — | — |

---

## Quickstart (Local macOS / Apple Silicon & Ephemeral Cloud GPU)

```bash
# 1. Install dependencies into an isolated Python 3.12 virtualenv
make sync

# 2. Run all 60 unit & integration tests and ruff linter
make test
make lint

# 3. Verify H1 (GPT-2 124M parity), H5 (StatefulToolEnv), H6 (Context vs. Weights), and H7/H8 (Weight Surgery & Pareto)
make verify-h1
make verify-env
make compare-adaptation
make run-surgery
make report

#    LLM backend (Qwen2.5-3B-Instruct on Apple Silicon MPS) for H5/H6/H7/H8:
make verify-env ARGS="--backend llm --model-name Qwen/Qwen2.5-3B-Instruct"
make compare-adaptation ARGS="--backend llm --model-name Qwen/Qwen2.5-3B-Instruct"
make run-surgery ARGS="--backend llm --model-name Qwen/Qwen2.5-3B-Instruct"

# 4. Run end-to-end M2+M3 local pretraining smoke test on Apple Silicon MPS / CPU
make train-smoke

# 5. Profile the 5-stage H2 throughput waterfall locally
make profile-h2-local
```

### Running `M4–M5` on an Ephemeral Cloud GPU (`RTX 4090` / `H100 SXM`)

```bash
# 1. Bootstrap Ubuntu + NVIDIA GPU instance (installs uv, Python 3.12, locked deps, runs pytest)
./scripts/remote/bootstrap.sh

# 2. Run the full M4-M5 GPU experiment pipeline (with 15-min idle GPU auto-shutdown watchdog):
#    - H1 parity verification
#    - H2 5-stage MFU & Triton fused-kernel waterfall
#    - FineWeb-Edu streaming BPE + uint16 shard generation (< 15 GB disk discipline)
#    - H3/H4 WSD pretraining with cooldown branching & 15 log-spaced checkpoints
#    - Markdown report compilation to reports/sprint1/summary.md
GPU_PROFILE=h100 ./scripts/remote/run_m4_m5_gpu.sh
# Or on a 24 GB RTX 4090 pilot instance:
GPU_PROFILE=rtx4090 MAX_DOCS=50000 MAX_STEPS=2000 ./scripts/remote/run_m4_m5_gpu.sh
```

---

## Architecture & Decisions

* [Tenet Lab Technical Report (`H1–H12` & `Tenet` Family Post-Mortem)](reports/tenet_technical_report.md)
* [Master Roadmap & Sprint Specifications](docs/roadmap.md)
* [Sprint 1 Measured Results Summary](reports/sprint1/summary.md)
* [ADR-001: Four-Sprint Architecture vs. Eight Loose Phases](docs/decisions/001-four-sprint-architecture.md)
* [ADR-002: Grouped-Query Attention (GQA) & WSD Cooldown Branching](docs/decisions/002-gqa-and-wsd-schedule.md)
* [ADR-003: Streaming Data Shards & `< 15 GB` Local Disk Discipline](docs/decisions/003-streaming-disk-discipline.md)
* [ADR-004: Multi-Turn Verifiable RL Environment (`StatefulToolEnv` + Gated Anti-Hacking `Rubric`)](docs/decisions/004-verifiable-rl-analytics-environment.md)
* [ADR-005: Lancet Dual-Anchor Soft-SCR Surgery (`H9`) → Tenet Subspace-Safe GRPO (`H10`) → FP8 Re-Leakage & Crest-Factor Provider Readiness (`H11`)](docs/decisions/005-lancet-biprojected-surgery.md)
* [ADR-006: Three-Tier Model Storage & Recipes](docs/decisions/006-three-tier-model-storage-and-recipes.md)
* [ADR-007: Cross-Domain Subspace-Safe Adapter Merge (`H12`)](docs/decisions/007-cross-domain-subspace-safe-adapter-merge.md)
* [ADR-008: Layer-Streaming Lancet Surgery & Subspace-Safe Merge (`H13`)](docs/decisions/008-streaming-shard-surgery-32b-70b.md)
* [ADR-009: GGUF Quantization Survival — Static `Q4_K_M` vs. `imatrix` vs. CKA-Guided Mixed Precision (`H14`)](docs/decisions/009-gguf-imatrix-vs-cka-survival.md)
* [ADR-010: Tenet v2 — Thinking-Safe Tri-Anchor Surgery, Length-Calibrated GRPO & 24-Seed Fight Card Gate (`H15–H17`)](docs/decisions/010-tenet-v2-thinking-surgery-and-length-calibrated-grpo.md)

---

## Author & Collaboration / Hiring Callout

**Andrey Isachenko** — Research Engineer (LLM Systems & Post-Training Surgery):
- **Core Focus:** Custom Triton/CUDA kernels (`@triton.jit` fused RMSNorm/SwiGLU/CrossEntropy), Subspace Surgery (`Lancet` bi-projected MGS & streaming 32B–70B `safetensors`), Subspace-Safe RL (`GRPO` with orthogonal LoRA & length-calibration), and Quantization-Aware Survival (`FP8 E4M3` Crest-Factor & `CKA-Q4_K_M` GGUF).
- **Open To:** Research engineering collaborations, post-training/inference consulting, and full-time Research Engineer / ML Systems roles.
- **Links:** [GitHub (`mrAndreyIsachenko/tenet-lab`)](https://github.com/mrAndreyIsachenko/tenet-lab) · [Hugging Face (`mr-andrey-isachenko`)](https://huggingface.co/mr-andrey-isachenko) · [HF Research Article](https://huggingface.co/blog/mr-andrey-isachenko/tenet-v2-thinking-safe-weight-surgery) · [FP8 Inspector Space](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)
