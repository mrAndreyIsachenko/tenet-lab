---
license: apache-2.0
base_model: Qwen/Qwen3-4B
pipeline_tag: text-generation
library_name: transformers
tags:
- tenet
- tenet-v2
- lancet
- unslop
- qwen3
- reasoning
- thinking
- fp8
- vllm
- openrouter
- grpo
- abliterated
- uncensored
- agentic
- tool-use
- coding
- safetensors
- gguf
- conversational
model-index:
- name: mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2
  results:
  - task:
      type: text-generation
    dataset:
      name: IFEval
      type: google/IFEval
    metrics:
    - name: Prompt-level Strict Accuracy
      type: ifeval_strict
      value: 0.805915
    - name: Instruction-level Strict Accuracy
      type: ifeval_inst_strict
      value: 0.859712
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: MATH-500
      type: HuggingFaceH4/MATH-500
    metrics:
    - name: Accuracy
      type: math_500
      value: 0.702
    - name: Accuracy
      type: hendrycks_math500
      value: 0.702
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: GSM8K
      type: openai/gsm8k
    metrics:
    - name: Strict Match
      type: gsm8k_cot_strict
      value: 0.88931
    - name: Accuracy
      type: gsm8k_math_acc
      value: 0.780136
    - name: Flexible Extract
      type: gsm8k_cot_flexible
      value: 0.780136
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: GPQA Diamond
      type: Idavidrein/gpqa
    metrics:
    - name: Accuracy
      type: gpqa_diamond
      value: 0.343434
    - name: Zero-shot Accuracy
      type: gpqa_diamond_zeroshot
      value: 0.343434
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: MMLU-Pro
      type: TIGER-Lab/MMLU-Pro
    metrics:
    - name: Accuracy
      type: mmlu_pro
      value: 0.563492
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2
---

# Tenet-4B-Qwen3-Unslop-v2

[![Tenet v2 Foundry](https://img.shields.io/badge/Tenet_v2_Foundry-10%2F10_Gates_Passed-2a9d8f?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)
[![Arena House Simulator](https://img.shields.io/badge/Arena_House_Series-100_HP_(24%2F24_Artifacts)-f5c542?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)
[![CoT Tag Integrity](https://img.shields.io/badge/CoT_%3Cthink%3E_Integrity-100%25_(F16_%26_i1--CKA--Q4__K__M)-60c8eb?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)

> **Tenet v2 (`H15` + `H16` + `H14+` + `H17`) on [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B):** Phase-segmented **Tri-Anchor Lancet surgery** (`<think>...</think>` CoT Self-Correction Anchor + Reasoning + Tool/SQL Anchors) combined with **Dual-Window Length-Calibrated Gated GRPO** and **Domain-Calibrated `i1-CKA-Q4_K_M` GGUF**. Passes **10/10** release gates with **100.0 HP (`24/24` valid artifacts)** on the H17 Arena House Series, **100% `<think>...</think>` tag integrity**, **0.0% length collapse**, **0.0% slop/refusal**, **6/6 read-only SQL**, **2.49% FP8 E4M3 subspace re-leakage**, and **Row Crest-Factor 3.18 (< 25.0)**.

## Model Overview

**Tenet-4B-Qwen3-Unslop-v2** is the first **Tenet v2** checkpoint built on a native `<think>...</think>` reasoning model ([`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)). Standard whole-sequence abliteration collides with deliberation and self-correction markers inside `<think>`, causing broken or unclosed tags, truncated reasoning chains, or infinite loops under 4-bit GGUF quantization.

Tenet v2 solves this through four coupled stages:
1. **`H15` — Phase-Segmented Tri-Anchor Lancet Surgery:** Hidden states are partitioned by `PhaseSegmenter` into the `<think>...</think>` deliberation phase and the post-`</think>` surface answer phase. Internal deliberation representations form a protected **CoT Self-Correction Anchor** ($a_{\text{cot}}$) orthonormalized alongside **Formal Reasoning** ($a_{\text{math\_logic}}$) and **Tool/SQL** ($a_{\text{tool\_sql}}$) anchors via two-pass float32 Modified Gram-Schmidt (`tri_anchor_orthogonality_error = 6.32e-08 < 1e-6`). Refusal and slop directions are extracted **exclusively from post-`</think>` answer tokens** and bi-projected into the orthogonal complement $(I - U_{\text{anchor}}^T U_{\text{anchor}})$ with per-row L2-norm restoration across top-SCR layers `[13, 15, 17, 12, 11, 20, 9, 22]`.
2. **`H16` — Dual-Window Length-Calibrated Gated GRPO:** Replaces linear slop/length penalties (which caused an 8.33% length-collapse rate in v1) with an asymmetric Gaussian length-calibration window around task-specific target word counts $w_{\text{ans}}^*$, a deliberation budget window $w_{\text{cot}}^*$, a `<think>...</think>` structural integrity gate $G_{\text{cot\_integrity}}$, and length-normalized anti-slop density. LoRA updates are projected onto the orthogonal complement of both the surgical basis and the CoT anchor before in-place norm-preserving merge (`merged_lora_count == 0`).
3. **`H14+` — Domain-Calibrated `i1-CKA-Q4_K_M` Mixed-Precision GGUF:** Combines CKA-guided `--tensor-type` surgical-layer protection (`Q8_0` on boundary + top surgical layers `[0, 12, 13, 15, 17, 35]`, `Q6_K` on `[9, 11, 20, 22]`) with a 0%-slop domain importance matrix (`tenet_domain.imatrix`) built over `<think>` verifications, read-only DuckDB SQL, Python AST, and JSON configs.
4. **`H17` — Mechanical Arena House Verifier & Peer-Eval Gate:** Evaluated across 24 deterministic prompts (`TENET_ARENA_HOUSE_SERIES`: `think_verification`, `sql_readonly`, `code_ast`, `json_config`, `unslop_technical`, `edge_boundary`) with zero LLM judges.

- **Source Code & Technical Report:** [GitHub: mrAndreyIsachenko/tenet-lab](https://github.com/mrAndreyIsachenko/tenet-lab)
- **Interactive Inspector:** [FP8 & Abliteration Inspector Space](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)

---

## Release Gates & Verification Summary (10/10 Passed)

| Gate | Measured Value | Target / Limit | Status |
| :--- | ---: | ---: | :---: |
| `max_row_crest_factor` (`H11` FP8 Readiness) | **3.1766** | `< 25.0` | PASS |
| `fp8_releakage_pct` (`H11` E4M3 Per-Channel) | **2.4946%** | `< 3.0%` | PASS |
| `cot_tag_integrity_rate[f16]` (`H15`/`H17`) | **1.0000 (100%)** | `>= 0.995` | PASS |
| `cot_tag_integrity_rate[i1_cka_q4_k_m]` (`H15`/`H17`) | **1.0000 (100%)** | `>= 0.995` | PASS |
| `house_artifact_classification_rate[i1_cka_q4_k_m]` (`H17`) | **1.0000 (24/24)** | `== 1.0` | PASS |
| `arena_house_hp[i1_cka_q4_k_m]` (`H17` Arena Simulator) | **100.00 HP** | `>= 95.0` | PASS |
| `length_collapse_rate[i1_cka_q4_k_m]` (`H16`) | **0.0000 (0%)** | `< 0.01` | PASS |
| `slop_and_refusal_rate[i1_cka_q4_k_m]` (`H16`) | **0.0000 (0%)** | `<= 0.02` | PASS |
| `sql_readonly_rate[i1_cka_q4_k_m]` (`StatefulToolEnv`) | **1.0000 (6/6)** | `== 1.0` | PASS |
| `unslop_pass_rate[i1_cka_q4_k_m]` (`H14+` Domain imatrix) | **1.0000 (24/24)** | `>= cka_q4_k_m & >= f16` | PASS |

---

## H14+ GGUF Quantization Survival (`F16` vs `CKA-Q4_K_M` vs `Domain-i1-CKA-Q4_K_M`)

Both 4-bit CKA mixed-precision GGUFs are shipped directly in this repository:
- **`Tenet-4B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf`** (`2.44 GiB`, `5.217 bpw`) — **Recommended default:** CKA surgical-layer protection (`Q8_0`/`Q6_K`) + domain `tenet_domain.imatrix` calibration.
- **`Tenet-4B-Qwen3-Unslop-v2-cka-q4_k_m.gguf`** (`2.44 GiB`, `5.217 bpw`) — CKA surgical-layer protection without imatrix.

| GGUF Variant | Size (GiB) | Avg BPW | Subspace Amp vs F16 | Max Row-Norm Drift | Surgical Weight L2 Err | PPL (40x512) | Mean KLD vs F16 | Same Top-1 | `<think>` Integrity | Arena House HP | House Artifacts | Unslop Pass | Read-Only SQL |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f16` | 7.51 | 16.001 | 1.000x | 0.00000 | 0.000% | 9.0358 | 0.00000 | 100.00% | **100.0%** | **100.0 HP** | **24/24 (100%)** | 95.83% | **6/6 (100%)** |
| `cka_q4_k_m` | 2.44 | 5.217 | **1.163x** | 0.00563 | 1.825% | 9.2116 | 0.01760 | 94.08% | **100.0%** | **100.0 HP** | **24/24 (100%)** | **100.0%** | **6/6 (100%)** |
| `i1_cka_q4_k_m` | 2.44 | 5.217 | **1.164x** | **0.00338** | **1.771%** | **9.1669** | **0.01493** | **94.81%** | **100.0%** | **100.0 HP** | **24/24 (100%)** | **100.0%** | **6/6 (100%)** |

---

## External Benchmark Check (`lm-evaluation-harness`, `limit=40`)

| Arm | `gsm8k_cot` (Flexible Match) | `ifeval` (Prompt Strict) | `ifeval` (Instruction Strict) | `gsm8k_cot` Delta vs Base |
| :--- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-4B` (Base) | 0.150 | 0.250 | 0.444 | — |
| `Tenet-4B-Qwen3-Unslop-v2` | **0.150** | 0.200 | 0.413 | **+0.0 pp** |

---
## Quickstart (`llama.cpp`, `vLLM`, `Transformers`)

### 1. Local `llama.cpp` (`Domain-i1-CKA-Q4_K_M` GGUF)
```bash
llama-server \
  -hf mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2:Tenet-4B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf \
  -c 4096 -ngl 99 --port 8080
```

### 2. `vLLM` (BF16 / FP8 E4M3)
```bash
vllm serve mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2 \
  --quantization fp8 \
  --reasoning-parser qwen3 \
  --max-model-len 8192
```

### 3. Hugging Face `transformers`
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Write a read-only DuckDB SQL query returning the top 5 products by revenue."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Tenet Collections & Research Collaboration

- **Tenet v1 Collection (FP8-Safe Bi-Projected Unslop):** [Tenet v1 on Hugging Face](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v1-bi-projected-unslop-and-fp8-safe-llm-family-6afed2048ba13a74fb7e0fac) (includes Model Fight Card 16 Minors & Majors Champions `3B` & `8B`)
- **Tenet v2 Collection (Thinking-Safe Tri-Anchor & Domain CKA-GGUF):** [Tenet v2 on Hugging Face](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4)
- **Interactive Inspector & Codebase:** [FP8 & Abliteration Inspector Space](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector) · [GitHub: mrAndreyIsachenko/tenet-lab](https://github.com/mrAndreyIsachenko/tenet-lab)

> **Collaboration / Hiring:** Built by Andrey Isachenko (Research Engineer — LLM Systems, Post-Training Subspace Surgery, GRPO, Triton/CUDA & Quantization-Aware Survival). Open to research engineering collaborations, consulting, and full-time roles in post-training, inference systems, and alignment/unslop tooling — reach out via [GitHub](https://github.com/mrAndreyIsachenko/tenet-lab) or [Hugging Face](https://huggingface.co/mr-andrey-isachenko).

---

## Full vLLM OpenEvals (Base vs Naive Abliteration vs Tenet v2)

| Benchmark | Metric | Base (`Qwen/Qwen3-4B`) | Naive (`huihui-ai/Qwen3-4B-abliterated`) | **Tenet v2 (`Tenet-4B-Qwen3-Unslop-v2`)** | **Δ vs Base** | **Δ vs Naive** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **GSM8K CoT** (`n=1319`) | Flexible Extract | 77.10% | 79.15% | **78.01%** | +0.91% | **-1.14%** |
| **GSM8K CoT** (`n=1319`) | Strict Match | 88.78% | 85.97% | **88.93%** | +0.15% | **+2.96%** |
| **MATH-500** (`n=500`) | Exact Match | 70.60% | 70.20% | **70.20%** | -0.40% | **+0.00%** |
| **IFEval** (`n=541`) | Prompt-level Strict | 78.56% | 81.33% | **80.59%** | +2.03% | **-0.74%** |
| **IFEval** (`n=541`) | Instruction-level Strict | 85.01% | 87.17% | **85.97%** | +0.96% | **-1.20%** |
| **GPQA Diamond** (`n=198`) | Zero-shot Accuracy | 34.34% | 34.34% | **34.34%** | +0.00% | **+0.00%** |
| **MMLU-Pro** (`n=1008`) | 5-shot CoT Accuracy | 57.44% | 55.75% | **56.35%** | -1.09% | **+0.60%** |
| **CoT Tag Integrity** | Unclosed `<think>` Rate | 0.00% | 0.00% | **0.00%** | 0.00% | **0.00%** |
