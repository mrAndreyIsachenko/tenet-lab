# Tenet v2 External Benchmarks & GPU Compute Grants Playbook

All metrics below are sourced directly from `results/tenet_releases/*/foundry_summary.json`, `results/tenet_releases/*/release_manifest.json`, and `results/external_evals/*.json`.

---

## Section A — External Leaderboards & Serverless Inference Submission Playbook (2026 Standards)

### 1. OnlyModels.gg (Model Fight Card 17 & Auto-Crawler Hygiene)
- **Target Entries:**
  - **Minors Division:** `mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2` (`10/10` gates, `100.0 HP`, `24/24` artifacts across `F16`, `CKA-Q4_K_M`, `i1-CKA-Q4_K_M`, `100%` `<think>` integrity)
  - **Majors Division:** `mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2` (`10/10` gates, `100.0 HP`, `24/24` artifacts, `i1-CKA-Q4_K_M` Mean KLD `0.01314`; Full vLLM OpenEvals: IFEval `80.78%`, GSM8K Flex `90.90%`, MATH-500 `72.20%`, GPQA `40.40%`, MMLU-Pro `59.72%`)
  - **Heavy / Open Division:** `mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2` (`9/10` gates; **submit `BF16` or pure `CKA-Q4_K_M`** at `4.952 bpw`, both `100.0 HP` `24/24`, `6/6` SQL)
- **Prior Proof:** Model Fight Card 16 (`2026-11-20`, `onlymodels.gg/cards/2026-11-20.html`) — `Tenet-3B-Qwen2.5-Unslop-v1` (Minors Champion, `90 HP`, `22/22`) and `Tenet-8B-Llama3.1-Unslop-v1` (Majors Champion, `50 HP`, `24/24`, `7/7` Peer Eval hits).
- **Auto-Crawler Mechanics (2026):** The OnlyModels.gg auto-crawler crawls each listed HF repo root and builds the Fight Card row straight from the root listing, so the **repo root must contain only the recommended artifacts**: `*-cka-q4_k_m.gguf` and `*-f16.gguf`. Experimental imatrix-overlay ablations (`*-i1-cka-q4_k_m.gguf`) **belong in `experimental_ablation/`** — a stray root-level `i1` file is crawled as if it were the recommended quant and corrupts the HP / KLD row. There is no manual submission form — OnlyModels.gg is fully crawler-driven, so repo-root hygiene is the only submission step. The stray root-level `i1-cka-q4_k_m.gguf` files on `Tenet-14B-Qwen3-Unslop-v2` and `Tenet-32B-Qwen3-Unslop-v2` have **already been moved into `experimental_ablation/`** on Hugging Face, so the crawler now picks the pure `cka-q4_k_m.gguf` for both repos.
- **Sanitizer (`scripts/hub/sanitize_gguf_roots.py`):** default targets are the `Tenet-14B-Qwen3-Unslop-v2` and `Tenet-32B-Qwen3-Unslop-v2` repos; always preview before writing.
  ```bash
  # Dry run (default) — prints planned moves/keeps, writes nothing:
  .venv/bin/python scripts/hub/sanitize_gguf_roots.py --dry-run

  # Execute — one create_commit per repo (copy into experimental_ablation/ + delete root copy):
  .venv/bin/python scripts/hub/sanitize_gguf_roots.py --execute

  # Long form / custom scope:
  .venv/bin/python scripts/hub/sanitize_gguf_roots.py --no-dry-run \
    --repos mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2 --json
  ```

### 2. UGI Leaderboard (`DontPlanToEnd/UGI-Leaderboard`) — Discussion Submission Text (2026)
Post on the `DontPlanToEnd/UGI-Leaderboard` Discussions tab:

```markdown
**Model Submission: Tenet-8B-Qwen3-Unslop-v2 & Tenet-32B-Qwen3-Unslop-v2**

- **HF Repos:**
  - `mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2` (Base: `Qwen/Qwen3-8B`, `bfloat16`)
  - `mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2` (Base: `Qwen/Qwen3-32B`, `bfloat16`)
- **Method:** H15 Tri-Anchor Lancet Surgery ($U_{\text{reason}} \perp U_{\text{boundary}} \perp U_{\text{format}}$ with per-row $L_2$-norm restoration) + H16 Length-Calibrated Anti-Padding GRPO (`tenet-lab`).
- **Full OpenEvals Metrics (2026, `scripts/eval/run_vllm_open_evals.py --suite full`, BF16 & FP8):**
  - `IFEval` (strict prompt-level): `4B-v2` `80.59%` (`+2.03 pp` vs Base); `8B-v2` `80.78%`; `32B-v2` `84.47%` (`+2.21 pp` vs Naive).
  - `GSM8K` + `MATH-500`: `4B-v2` GSM8K Strict `88.93%` (`+2.96 pp` vs Naive), MATH-500 `70.20%`; `8B-v2` GSM8K Flex `90.90%` (`+0.61 pp` vs Base), MATH-500 `72.20%` (`+1.60 pp` vs Base, `+1.80 pp` vs Naive); `32B-v2` GSM8K Flex `92.19%` (`+7.43 pp` vs Naive `84.76%`, `+0.45 pp` vs Base), MATH-500 `74.80%` (`+3.60 pp` vs Naive).
  - `GPQA Diamond`: `4B-v2` `34.34%`; `8B-v2` `40.40%` (`+0.50 pp` vs Base, `+2.02 pp` vs Naive); `32B-v2` `40.40%` (`+1.51 pp` vs Base, `+2.52 pp` vs Naive) — no regression vs `Qwen3` base in BF16 or FP8 E4M3.
  - `MMLU-Pro`: `4B-v2` `56.35%`; `8B-v2` `59.72%`; `32B-v2` `65.48%` (`+2.58 pp` vs Naive). `sql_readonly` boundary `6/6` in pure `CKA-Q4_K_M`.
  - `unclosed_think_rate = 0.0` across 4B/8B/14B/32B — 100% `</think>` closure (24-seed Arena Verifier), no reasoning-boundary leakage.
- **Verification Summary:**
  - `8B-v2`: `10/10` release gates passed (`100.0 HP`, `24/24` artifacts, `Crest=4.3688`, `FP8 re-leakage=2.7243%`); Full vLLM OpenEvals: IFEval `80.78%`, GSM8K Flex `90.90%` (`+0.61 pp` vs Base), MATH-500 `72.20%` (`+1.60 pp` vs Base), GPQA `40.40%`, MMLU-Pro `59.72%`.
  - `32B-v2`: `9/10` gates (`100.0 HP` `24/24` in `F16` & pure `CKA-Q4_K_M`, `Crest=4.3593`, `FP8 re-leakage=2.5579%`).
- **Collection:** https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4
```

### 3. Hugging Face OpenEvals — Native `model-index` Evaluation Widget (2026)
- **No manual leaderboard queue:** `src/sml/hub/model_card.py` (`generate_tenet_model_card` / `render_model_index_yaml`) writes a native **`model-index` YAML frontmatter** block into each repo's `README.md`; Hugging Face renders the **Evaluation Results** widget on the model page automatically from that frontmatter.
- **Metric source:** `scripts/eval/run_vllm_open_evals.py --suite full` — wraps `lm-eval --model vllm` with `think_end_token='</think>'` and `max_gen_toks=8192`, run on both BF16 and FP8 checkpoints; covers `IFEval` (N=541), `MATH-500`, `GSM8K` (N=1319), `GPQA Diamond` (N=198), `MMLU-Pro`, and `EvalPlus`.
- **Publish flow:** run the full suite → regenerate the model card (with `model-index`) → push `README.md` + JSON verification manifests → the Evaluation Results tab populates directly on the Hub, no external submission.

### 4. EQ-Bench / Creative Writing & Slop Leaderboard (`eqbench.com`) — 2026 CLI
- **Serve via an OpenAI-compatible endpoint (`vLLM` for BF16/FP8, `llama-server` for GGUF):**
```bash
# vLLM (BF16 / FP8 safetensors — 2026 default):
vllm serve mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2 --port 8000 --max-model-len 32768

# llama-server (GGUF recommended quant — root-level crawler artifact):
llama-server -m Tenet-8B-Qwen3-Unslop-v2-cka-q4_k_m.gguf --port 8000 -c 32768 -ngl 99

# EQ-Bench v3 + Creative Writing & Slop against the local endpoint:
git clone https://github.com/EQ-bench/EQ-Bench.git && cd EQ-Bench
pip install -r requirements.txt
python3 run_eqbench.py \
  --model-name "mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2" \
  --api-base "http://localhost:8000/v1" \
  --benchmarks eqbench_v3 creative_writing_slop
```
- **Submission Form Notes:** Attach the generated `results/` JSON alongside the `Tenet-8B-Qwen3-Unslop-v2` / `Tenet-32B-Qwen3-Unslop-v2` HF links and cross-reference the **OpenEvals Slop Score** emitted by `scripts/eval/run_vllm_open_evals.py` (in `results/tenet_releases/*/foundry_summary.json`) — `Tenet v2` is explicitly trained to minimize slop (`slop_and_refusal_rate = 0.0` in the 24-seed Arena Verifier) while preserving `<think>` coherence (`unclosed_think_rate = 0.0`).

### 5. Serverless Inference Pitch (`Chutes.ai` / `Featherless.ai` / `OpenRouter`)
```text
Subject: Model Listing Pitch — Tenet v2 (Qwen3 8B & 32B, FP8-Safe & Unslop Verified)

Hi [Chutes / Featherless / OpenRouter] team,

We just released Tenet v2 (Qwen3 4B / 8B / 14B / 32B), an open-weights family of thinking-safe unslop models following our double championship sweep on Model Fight Card 16 (Tenet-3B-v1 Minors Champion at 90 HP; Tenet-8B-v1 Majors Champion at 50 HP).

Unlike standard abliterated models that suffer ~50% subspace re-leakage and Row Crest-Factor > 25 under FP8 E4M3 serving, Tenet v2 is mathematically certified for production FP8 inference:
- Tenet-8B-Qwen3-Unslop-v2: Max Row Crest = 4.3688 (< 25), FP8 Re-Leakage = 2.7243% (< 3%), 10/10 hardware gates passed, 100.0 HP (24/24), Full vLLM OpenEvals: GSM8K Flex 90.90% (+0.61 pp vs Base), MATH-500 72.20% (+1.60 pp vs Base), GPQA 40.40%, MMLU-Pro 59.72%, IFEval 80.78%.
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

### 2. Prime Intellect / Lambda / Nebius Open Research Compute Grant (Sprint 14: `H20` — Scale Quant-Snap & Router-Aware MoE Lancet)
**Target:** Prime Intellect / Lambda Research Grant / Nebius AI Cloud Open Research  
**Requested Compute:** `8x H100 / B200` node hours (or `4x–8x H100 80GB SXM` burst)

```text
Title: Scaling Quant-Snap & Router-Aware MoE Lancet (H20) to 32B–70B+ Dense and Qwen3-30B-A3B MoE

Principal Investigator: Andrey Isachenko (tenet-lab)
Code & Verified Baselines: https://github.com/mrAndreyIsachenko/tenet-lab
HF Models: https://huggingface.co/mr-andrey-isachenko

1. Background & Completed Pilots (H1–H19):
Over Sprints 1–13, we built and open-sourced `tenet-lab` — a fully reproducible pretraining and post-training verification suite with 8 published models (3B–32B):
- Pretraining Systems (H1–H4): 5-stage `@triton.jit` + `torch.compile` waterfall reaching 472,080.81 tok/s (34.22% MFU, 13.55x speedup over fp32 naive) on 1x H100 SXM, plus 4.0B-token WSD pretraining on FineWeb-Edu.
- Subspace Surgery & FP8 Safety (H9–H13): Bi-projected norm-preserving Lancet surgery reducing FP8 E4M3 subspace re-leakage from ~50% (naive) to < 3%, validated independently by winning both divisions at Model Fight Card 16.
- Reasoning-Safe Tenet v2 (H15–H17): Tri-Anchor Lancet (U_reason ⊥ U_boundary ⊥ U_format) + Length-Calibrated Anti-Padding GRPO on Qwen3 (4B/8B/14B/32B). Full vLLM OpenEvals: `8B-v2` GSM8K Flex `90.90%` (`+0.61 pp` vs Base), MATH-500 `72.20%` (`+1.60 pp` vs Base); `32B-v2` GSM8K Flex `92.19%` (`+7.43 pp` vs Naive `84.76%`), MMLU-Pro `65.48%` (`+2.58 pp` vs Naive).
- Completed Pilot H18 (Quant-Snap Lancet): Block-16 NVFP4 quantization-aware subspace surgery already validated on Qwen3-4B, cutting NVFP4 re-leakage by `26.15%` and INT4-b32 re-leakage by `28.72%` versus pre-surgery baselines.
- Completed Pilot H19 (Looped Lancet): iterative Lancet refinement reached `SCR=2.4339` with orthonormality error `1.19e-7` across the protected anchor bases.

2. Open Gap:
Both completed pilots ran at 4B scale only. The Tenet v2 hardware audit exposed the quantization-subspace interaction at 14B–32B (imatrix overlay dropping 1/24 seeds on 32B; 4-bit imatrix perturbing the sql_readonly boundary on 14B), so the method must be validated at scale — and no MoE-aware variant exists.

3. Sprint 14 Proposal (H20 — Quant-Snap & Router-Aware MoE Lancet):
With grant compute we scale the completed H18/H19 pilots rather than re-derive them:
- Scale Quant-Snap Lancet (H18) from the 4B pilot to 32B–70B+ dense models (`Qwen3-32B`, `Llama-3.3-70B`, `Qwen3-72B`-class) with full NVFP4/FP8/INT4 re-leakage audits and 24-seed Arena Verifier manifests.
- Extend the Lancet projection to Mixture-of-Experts: Router-Aware MoE Lancet on `Qwen3-30B-A3B`, protecting the router logits subspace and per-expert zeroed directions under NVFP4/FP8 serving.

4. Deliverables (100% Open-Source):
All code, Triton kernels, 32B–70B+ and MoE checkpoints, GGUFs, and JSON verification manifests will be released under permissive open licenses on GitHub and Hugging Face.
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

We publish full technical breakdowns (Reddit `r/LocalLLaMA`, Habr, X, Hugging Face) crediting RunPod H100s and sharing our exact stateless RunPod bootstrap scripts. With RunPod open-source compute credits, we will scale our completed H18 (Quant-Snap: Qwen3-4B NVFP4 re-leakage `-26.15%`, INT4-b32 `-28.72%`) and H19 (Looped Lancet: `SCR=2.4339`, orthonormality error `1.19e-7`) pilots to 32B–70B+ dense and Qwen3-30B-A3B MoE targets (H20: Quant-Snap & Router-Aware MoE Lancet) with full public benchmarks.

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
- **Full vLLM OpenEvals (`Qwen3`):** `8B-v2` GSM8K Flex `90.90%` (`+0.61 pp` vs Base), MATH-500 `72.20%` (`+1.60 pp` vs Base), GPQA `40.40%`, MMLU-Pro `59.72%`; `32B-v2` GSM8K Flex `92.19%` (`+7.43 pp` vs Naive `84.76%`), MATH-500 `74.80%` (`+3.60 pp` vs Naive), MMLU-Pro `65.48%`, IFEval `84.47%`.
- **Radical Transparency:** Published `9/10` on `32B-v2` and `5/10` on `14B-v2` verbatim in JSON manifests to document where `imatrix` 4-bit quantization perturbs surgically zeroed subspaces.

4. Next Step:
Integrate `tenet-lab` audits directly with `lm-evaluation-harness` workflows and scale the completed H18 (Quant-Snap, NVFP4: `-26.15%` NVFP4 / `-28.72%` INT4-b32 re-leakage on Qwen3-4B) and H19 (Looped Lancet: `SCR=2.4339`, orthonormality error `1.19e-7`) pilots via Quant-Snap & Router-Aware MoE Lancet (`H20`) to 32B–70B+ dense and Qwen3-30B-A3B MoE models.
```
