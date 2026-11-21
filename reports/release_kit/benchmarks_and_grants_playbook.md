# Tenet v2 External Benchmarks & GPU Compute Grants Playbook

All metrics below are sourced directly from `results/tenet_releases/*/foundry_summary.json`, `results/tenet_releases/*/release_manifest.json`, and `results/external_evals/*.json`.

---

## Section A — External Leaderboards & Serverless Inference Submission Playbook

### 1. OnlyModels.gg (Model Fight Card 17)
- **Target Entries:**
  - **Minors Division:** `mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2` (`10/10` gates, `100.0 HP`, `24/24` artifacts across `F16`, `CKA-Q4_K_M`, `i1-CKA-Q4_K_M`, `100%` `<think>` integrity)
  - **Majors Division:** `mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2` (`10/10` gates, `100.0 HP`, `24/24` artifacts, `i1-CKA-Q4_K_M` Mean KLD `0.01314`, `0.300` `IFEval Strict` vs `0.100` Naive)
  - **Heavy / Open Division:** `mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2` (`9/10` gates; **submit `BF16` or pure `CKA-Q4_K_M`** at `4.952 bpw`, both `100.0 HP` `24/24`, `6/6` SQL)
- **Prior Proof:** Model Fight Card 16 (`2026-11-20`, `onlymodels.gg/cards/2026-11-20.html`) — `Tenet-3B-Qwen2.5-Unslop-v1` (Minors Champion, `90 HP`, `22/22`) and `Tenet-8B-Llama3.1-Unslop-v1` (Majors Champion, `50 HP`, `24/24`, `7/7` Peer Eval hits).

### 2. UGI Leaderboard (`DontPlanToEnd/UGI-Leaderboard`) — Discussion Submission Text
Post on the `DontPlanToEnd/UGI-Leaderboard` Discussions tab:

```markdown
**Model Submission: Tenet-8B-Qwen3-Unslop-v2 & Tenet-32B-Qwen3-Unslop-v2**

- **HF Repos:**
  - `mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2` (Base: `Qwen/Qwen3-8B`, `bfloat16`)
  - `mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2` (Base: `Qwen/Qwen3-32B`, `bfloat16`)
- **Method:** H15 Tri-Anchor Lancet Surgery ($U_{\text{reason}} \perp U_{\text{boundary}} \perp U_{\text{format}}$ with per-row $L_2$-norm restoration) + H16 Length-Calibrated Anti-Padding GRPO (`tenet-lab`).
- **Verification Summary:**
  - `8B-v2`: `10/10` release gates passed (`100.0 HP`, `24/24` artifacts, `Crest=4.3688`, `FP8 re-leakage=2.7243%`), holds `0.300` `IFEval Strict` (vs `0.100` naive abliteration) and completes 4-task `lm-eval` 17.4% faster (`851.8s` vs `1030.7s` base).
  - `32B-v2`: `9/10` gates (`100.0 HP` `24/24` in `F16` & pure `CKA-Q4_K_M`, `Crest=4.3593`, `FP8 re-leakage=2.5579%`).
- **Collection:** https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4
```

### 3. Open LLM Leaderboard v2 (`huggingfaceh4/open_llm_leaderboard`)
- **Exact Submission Parameters:**
  - **Model ID:** `mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2`, `mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2`, `mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2`, `mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2`
  - **Precision:** `bfloat16`
  - **Weight type:** `Original` (`safetensors`)
  - **Model type:** `Fine-tuned` (`chat` template, `Qwen3ForCausalLM`)
  - **Revision:** `main`
  - **Pre-submit check:** verify `config.json` has `"architectures": ["Qwen3ForCausalLM"]` and `tokenizer_config.json` includes `<think>`/`</think>` special tokens.

### 4. EQ-Bench v3 / Creative Writing Slop Leaderboard (`eqbench.com`)
- **Local / Cloud Run Command (via OpenAI-compatible `llama-server` or `vLLM`):**
```bash
# Serve Tenet-8B-Qwen3-Unslop-v2 (or 32B-v2 CKA-Q4_K_M)
llama-server -m Tenet-8B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf --port 8080 -c 8192 -ngl 99

# Run EQ-Bench / Creative Writing Slop benchmark against local endpoint
git clone https://github.com/EQ-bench/EQ-Bench.git && cd EQ-Bench
python3 run_eqbench.py \
  --model-name "mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2" \
  --api-base "http://localhost:8080/v1" \
  --benchmarks eqbench_v3 creative_writing_slop
```
- **Submission Form Notes:** Attach the generated `results/` JSON alongside the `Tenet-8B-Qwen3-Unslop-v2` / `Tenet-32B-Qwen3-Unslop-v2` HF link and note that `Tenet v2` is explicitly trained to minimize slop (`slop_and_refusal_rate = 0.0` in 24-seed Arena Verifier) while preserving `<think>` coherence.

### 5. Serverless Inference Pitch (`Chutes.ai` / `Featherless.ai` / `OpenRouter`)
```text
Subject: Model Listing Pitch — Tenet v2 (Qwen3 8B & 32B, FP8-Safe & Unslop Verified)

Hi [Chutes / Featherless / OpenRouter] team,

We just released Tenet v2 (Qwen3 4B / 8B / 14B / 32B), an open-weights family of thinking-safe unslop models following our double championship sweep on Model Fight Card 16 (Tenet-3B-v1 Minors Champion at 90 HP; Tenet-8B-v1 Majors Champion at 50 HP).

Unlike standard abliterated models that suffer ~50% subspace re-leakage and Row Crest-Factor > 25 under FP8 E4M3 serving, Tenet v2 is mathematically certified for production FP8 inference:
- Tenet-8B-Qwen3-Unslop-v2: Max Row Crest = 4.3688 (< 25), FP8 Re-Leakage = 2.7243% (< 3%), 10/10 hardware gates passed, 100.0 HP (24/24), holds 100% of Base IFEval Strict (0.300 vs 0.100 naive) while running 17.4% faster.
- Tenet-32B-Qwen3-Unslop-v2: Max Row Crest = 4.3593 (< 25), FP8 Re-Leakage = 2.5579% (< 3%), 100.0 HP (24/24) in BF16/F16 & pure CKA-Q4_K_M.

HF Collection: https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4
FP8 Inspector Space: https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector

Would love to see Tenet-8B-Qwen3-Unslop-v2 and Tenet-32B-Qwen3-Unslop-v2 hosted on your serverless FP8/BF16 endpoints.
```

---

## Section B — 4 Copy-Paste-Ready Compute & GPU Grant Applications

### 1. Hugging Face Community GPU Grant (ZeroGPU / L40S for `fp8-abliteration-inspector`)
**Target:** Hugging Face Community GPU Grant  
**Space:** `https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector`

```text
Project Title: FP8 & Abliteration Subspace Inspector (ZeroGPU / L40S Community Grant)
Space URL: https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector
Repository: https://github.com/mrAndreyIsachenko/tenet-lab

Problem & Community Value:
Thousands of open-weights fine-tunes and abliterated models are uploaded to Hugging Face every month, yet creators have no fast, standardized way to check whether their weights survive production FP8 (E4M3) serving or 4-bit GGUF quantization. Our research (H11, H14+) demonstrated that naive abliteration suffers ~50% subspace re-leakage under FP8 E4M3 and Row Crest-Factor spikes (> 25.0) that silently break vLLM/OpenRouter deployments.

What the Space Does:
`fp8-abliteration-inspector` is an interactive, zero-upload diagnostic tool for the Hugging Face community. Given any public HF model ID, it audits:
1. Per-layer Row Crest-Factor (`max_i ||W_i||_inf / RMS(W_i)`) and FP8 E4M3 subspace re-leakage %.
2. CKA layer-sensitivity profiles for `Q4_K_M` / `Q6_K` / `Q8_0` mixed-precision GGUF quantization.
3. `<think>...</think>` boundary integrity and refusal/slop subspace orthogonality.

Why We Need ZeroGPU / L40S:
Inspecting 8B–32B models (`Qwen3-8B`, `Qwen3-14B`, `Qwen3-32B`) on CPU takes several minutes per layer scan and times out on free CPU Spaces. A ZeroGPU or L40S allocation lets any open-source developer run a full FP8 + CKA audit on 8B–32B checkpoints in seconds before publishing to the Hub.

Track Record:
- 8 published open-weights models across Tenet v1 & v2 (3B–32B) with full JSON audit manifests.
- Double Championship at Model Fight Card 16 (onlymodels.gg, 2026-11-20): Tenet-3B-v1 (Minors, 90 HP) & Tenet-8B-v1 (Majors, 50 HP, 7/7 Peer Eval hits).
```

---

### 2. Prime Intellect / Lambda / Nebius Open Research Compute Grant (Sprint 13: `H18`)
**Target:** Prime Intellect / Lambda Research Grant / Nebius AI Cloud Open Research  
**Requested Compute:** `8x H100 / B200` node hours (or `4x–8x H100 80GB SXM` burst)

```text
Title: Quant-Snap Lancet (H18): Block-16 NVFP4 Quantization-Aware Subspace Surgery & 70B+ Streaming Tri-Anchor Unslop

Principal Investigator: Andrey Isachenko (tenet-lab)
Code & Verified Baselines: https://github.com/mrAndreyIsachenko/tenet-lab
HF Models: https://huggingface.co/mr-andrey-isachenko

1. Background & Verified Track Record (H1–H17):
Over Sprints 1–12, we built and open-sourced `tenet-lab`—a fully reproducible pretraining and post-training verification suite with 8 published models (3B–32B):
- Pretraining Systems (H1–H4): 5-stage `@triton.jit` + `torch.compile` waterfall reaching 472,080.81 tok/s (34.22% MFU, 13.55x speedup over fp32 naive) on 1x H100 SXM, plus 4.0B-token WSD pretraining on FineWeb-Edu.
- Subspace Surgery & FP8 Safety (H9–H13): Bi-projected norm-preserving Lancet surgery reducing FP8 E4M3 subspace re-leakage from ~50% (naive) to < 3%, validated independently by winning both divisions at Model Fight Card 16 (Tenet-3B-v1 at 90 HP; Tenet-8B-v1 at 50 HP, 7/7 Peer Eval hits).
- Reasoning-Safe Tenet v2 (H15–H17): Tri-Anchor Lancet (U_reason ⊥ U_boundary ⊥ U_format) + Length-Calibrated Anti-Padding GRPO on Qwen3 (4B/8B/14B/32B). On Qwen3-8B, Tenet v2 holds 0.300 IFEval Strict (vs 0.100 naive) while running 17.4% faster; on Qwen3-14B, GSM8K CoT improves +10.0 pp (0.250 vs 0.150).

2. Open Gap Discovered in Tenet v2:
Our 10-gate hardware audit exposed a fundamental quantization-subspace interaction at 14B–32B:
- On `Tenet-32B-Qwen3-Unslop-v2`, F16 and pure `CKA-Q4_K_M` (4.952 bpw) score 100.0 HP (24/24), whereas overlaying `imatrix` (`i1-CKA-Q4_K_M`) drops 1/24 seeds (97.92 HP).
- On `Tenet-14B-Qwen3-Unslop-v2`, 4-bit `imatrix` perturbs the `sql_readonly` boundary (`0/6` vs `6/6` in `CKA-Q4_K_M`).

3. Sprint 13 Proposal (H18 — Quant-Snap Lancet):
We propose to solve 4-bit quantization re-leakage directly at surgery time:
- Formulate **Quant-Snap Lancet**: joint Tri-Anchor orthogonal projection with a differentiable Block-16 NVFP4 / Q4_K_M round-trip penalty during layer-wise surgery, preventing `imatrix` activation re-weighting from rotating the zeroed subspace.
- Scale **Streaming Tri-Anchor Surgery (H13+H15)** to **70B+ reasoning models** (`Llama-3.3-70B` / `Qwen3-72B`-class) with full 3-way `lm-eval`, NVFP4/FP8 E4M3 re-leakage audits, and 24-seed Arena Verifier manifests.

4. Deliverables (100% Open-Source):
All code, Triton kernels, 70B+ checkpoints, GGUFs, and JSON verification manifests will be released under permissive open licenses on GitHub and Hugging Face.
```

---

### 3. RunPod Creator / Open-Source Compute Credits Pitch
**Target:** RunPod Creator Program / Open-Source Credits (`@runpod_io`)

```text
Subject: Open-Source Compute Credits Pitch — tenet-lab & Tenet v1/v2 (Built on RunPod H100s)

Hi RunPod team,

I'm building `tenet-lab` (https://github.com/mrAndreyIsachenko/tenet-lab), an open-source post-training & quantization-safety lab. Every major milestone in our project was trained and verified on rented RunPod 1x H100 80GB SXM instances using reproducible stateless scripts (`uv sync` + `hf_transfer` + `llama.cpp CUDA`):

- **34.22% MFU Triton Waterfall:** 13.55x speedup (`34,850` -> `472,080.81 tok/s`) + 4.0B-token FineWeb-Edu WSD run on 1x H100 SXM.
- **Double Championship at Model Fight Card 16 (`onlymodels.gg`, 2026-11-20):** `Tenet-3B-Qwen2.5-Unslop-v1` won Minors (90 HP, 22/22) and `Tenet-8B-Llama3.1-Unslop-v1` won Majors (50 HP, 24/24, 7/7 Peer Eval hits).
- **Tenet v2 (`Qwen3 4B / 8B / 14B / 32B`):** 3-way `lm-eval` + 10-gate FP8/GGUF release suite for 14B and 32B executed end-to-end on RunPod H100.

We publish full technical breakdowns (Reddit `r/LocalLLaMA`, Habr, X, Hugging Face) crediting RunPod H100s and sharing our exact stateless RunPod bootstrap scripts. With RunPod open-source compute credits, we will extend our pipeline to 70B+ streaming surgery and NVFP4 (`H18`) with full public benchmarks.

Links:
- HF Family (8 models): https://huggingface.co/mr-andrey-isachenko
- FP8 Inspector Space: https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector
- GitHub: https://github.com/mrAndreyIsachenko/tenet-lab
```

---

### 4. AI Grant / EleutherAI Research Pitch
**Target:** AI Grant / EleutherAI Research Collaboration

```text
Memorandum: Turning `tenet-lab` (Lancet + CKA-GGUF + Arena Verifier) into a Standard Open-Source Post-Training Pipeline

1. Thesis:
Open-weights post-training currently lacks hardware-grounded verification. Weight-editing (abliteration, LoRA merging, RL) is routinely evaluated only in `bf16` on synthetic prompts, then silently fails under FP8 (E4M3) serving (~50% subspace re-leakage, Row Crest-Factor > 25) or 4-bit GGUF quantization.

2. What We Built (`mrAndreyIsachenko/tenet-lab`):
An end-to-end, deterministic post-training & certification stack:
- **Lancet (H9 / H15):** `float32` 2-pass MGS Tri-Anchor orthogonal surgery ($U_{\text{reason}} \perp U_{\text{boundary}} \perp U_{\text{format}}$) with per-row $L_2$-norm restoration.
- **Subspace-Safe & Length-Calibrated GRPO (H10 / H16):** prevents RL re-leakage and `<think>` padding.
- **FP8 & CKA-GGUF Gate (H11 / H14+ / H17):** 10 automated hardware gates (Crest < 25, FP8 re-leakage < 3%, CKA layer protection, 24-seed Arena Verifier, read-only SQL checks).

3. Empirical Proof:
- **Independent Eval:** Swept both divisions of Model Fight Card 16 (`Tenet-3B-v1` Minors at 90 HP; `Tenet-8B-v1` Majors at 50 HP, 7/7 Peer Eval hits).
- **3-Way `lm-eval` (`Qwen3`):** On `8B-v2`, holds 100% of Base `IFEval Strict` (`0.300` vs `0.100` Naive) while running 17.4% faster (`851.8s` vs `1030.7s`); on `14B-v2`, gains `+10.0 pp` `GSM8K CoT` (`0.250` vs `0.150`).
- **Radical Transparency:** Published `9/10` on `32B-v2` and `5/10` on `14B-v2` verbatim in JSON manifests to document where `imatrix` 4-bit quantization perturbs surgically zeroed subspaces.

4. Next Step:
Integrate `tenet-lab` audits directly with `lm-evaluation-harness` workflows and scale Quantization-Aware Subspace Surgery (`H18`, NVFP4 / Q4_K_M) to 70B+ models.
```
