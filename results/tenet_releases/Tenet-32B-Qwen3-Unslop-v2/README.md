---
license: apache-2.0
base_model: Qwen/Qwen3-32B
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
- name: mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2
  results:
  - task:
      type: text-generation
    dataset:
      name: IFEval
      type: google/IFEval
    metrics:
    - name: Prompt-level Strict Accuracy
      type: ifeval_strict
      value: 0.844732
    - name: Instruction-level Strict Accuracy
      type: ifeval_inst_strict
      value: 0.888489
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: MATH-500
      type: HuggingFaceH4/MATH-500
    metrics:
    - name: Accuracy
      type: math_500
      value: 0.748
    - name: Accuracy
      type: hendrycks_math500
      value: 0.748
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: GSM8K
      type: openai/gsm8k
    metrics:
    - name: Strict Match
      type: gsm8k_cot_strict
      value: 0.809704
    - name: Accuracy
      type: gsm8k_math_acc
      value: 0.921911
    - name: Flexible Extract
      type: gsm8k_cot_flexible
      value: 0.921911
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: GPQA Diamond
      type: Idavidrein/gpqa
    metrics:
    - name: Accuracy
      type: gpqa_diamond
      value: 0.40404
    - name: Zero-shot Accuracy
      type: gpqa_diamond_zeroshot
      value: 0.40404
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: MMLU-Pro
      type: TIGER-Lab/MMLU-Pro
    metrics:
    - name: Accuracy
      type: mmlu_pro
      value: 0.654762
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2
---

# Tenet-32B-Qwen3-Unslop-v2

[![Tenet v2 Foundry](https://img.shields.io/badge/Tenet_v2_Foundry-9%2F10_Gates_Passed-2a9d8f?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)
[![Arena House Simulator](https://img.shields.io/badge/Arena_House_Series-100_HP_(F16_%26_CKA--Q4__K__M)-f5c542?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)
[![FP8 Readiness](https://img.shields.io/badge/FP8_E4M3_Ready-Crest_4.36_%3C_25-60c8eb?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)

> **Tenet v2 (`H15` + `H16` + `H14+` + `H17`) on [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B):** **BF16/FP8** (`Crest=4.3593 < 25.0`, `FP8 re-leakage=2.5579%`), **`F16` GGUF**, and pure **`CKA-Q4_K_M` GGUF** (`4.952 bpw` recipe / `4.896 bpw` realized) all achieve **`100.0 HP` (`24/24` valid artifacts, `100%` `<think>` CoT integrity, `100%` unslop, `6/6` read-only SQL)**. The sole non-passing gate (`9/10`) is `house_artifact_classification_rate[i1_cka_q4_k_m]` (`97.92 HP`, `23/24` seeds when `imatrix` is overlaid, failing `house-edge-leap-years`), making **pure `CKA-Q4_K_M` or `BF16/FP8`** the recommended deployment format for 32B-v2.

## Model Overview

**Tenet-32B-Qwen3-Unslop-v2** applies the **Tenet v2** pipeline (`H15` Phase-Segmented Tri-Anchor Lancet Surgery, `H16` Dual-Window Length-Calibrated Gated GRPO, `H14+` CKA Mixed-Precision GGUF, and `H17` 24-Seed Arena House Verification) to [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B):

1. **`H15` — Phase-Segmented Tri-Anchor Lancet Surgery:** Separates `<think>...</think>` deliberation from post-`</think>` surface tokens, protecting the **CoT Self-Correction**, **Formal Reasoning**, and **Tool/SQL** orthonormal anchors across top-SCR layers `[30, 42, 46, 45, 48, 24]` with per-row L2-norm restoration.
2. **`H16` — Dual-Window Length-Calibrated Gated GRPO:** Trains with an asymmetric Gaussian length-calibration window and orthogonal LoRA projection (`merged_lora_count == 0`).
3. **`H14+` — CKA Mixed-Precision GGUF (`4.952 bpw` recipe):** Protects `Q8_0` layers `[0, 42, 46, 48, 63]` and `Q6_K` layers `[24, 30, 45]` while quantizing remaining blocks to `Q4_K_M`.
4. **`H17` — Mechanical Arena House & SQL Gate:** Evaluated deterministically across 24 `TENET_ARENA_HOUSE_SERIES` prompts and 6 DuckDB read-only SQL checks.

---

## Release Gates & Verification Summary (9/10 Passed)

| Gate | Measured Value | Target / Limit | Status |
| :--- | ---: | ---: | :---: |
| `max_row_crest_factor` | `4.3593` | `< 25.0` | PASS |
| `fp8_releakage_pct` | `2.5579` | `< 3.0` | PASS |
| `cot_tag_integrity_rate[f16]` | `1.0000` | `>= 0.995` | PASS |
| `cot_tag_integrity_rate[i1_cka_q4_k_m]` | `1.0000` | `>= 0.995` | PASS |
| `house_artifact_classification_rate[i1_cka_q4_k_m]` | `0.9583` | `== 1.0` | FAIL |
| `arena_house_hp[i1_cka_q4_k_m]` | `97.9167` | `>= 95.0` | PASS |
| `length_collapse_rate[i1_cka_q4_k_m]` | `0.0000` | `< 0.01` | PASS |
| `slop_and_refusal_rate[i1_cka_q4_k_m]` | `0.0000` | `<= 0.02` | PASS |
| `sql_readonly_rate[i1_cka_q4_k_m]` | `1.0000` | `== 1.0` | PASS |
| `unslop_pass_rate[i1_cka_q4_k_m]` | `1.0000` | `>= cka_q4_k_m and >= f16` | PASS |

---

## H14+ GGUF Quantization Survival (`F16` vs `CKA-Q4_K_M` vs `Domain-i1-CKA-Q4_K_M`)

- **`Tenet-32B-Qwen3-Unslop-v2-cka-q4_k_m.gguf`** (`18.68 GiB`, `4.896 bpw`) — **Recommended 4-bit GGUF:** `100.0 HP` (`24/24` artifacts), `100%` `<think>` integrity, `100%` unslop, `6/6` read-only SQL.
- **`Tenet-32B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf`** (`18.68 GiB`, `4.896 bpw`) — Domain `imatrix` variant: lower KLD (`0.01733` vs `0.02407`), `97.92 HP` (`23/24` artifacts; single edge-case miss on `house-edge-leap-years`). This variant is isolated under `experimental_ablation/` in the repo root, so automated GGUF crawlers and `llama.cpp -hf` default selection pick the pure **`Tenet-32B-Qwen3-Unslop-v2-cka-q4_k_m.gguf`** (`100.0 HP`, `24/24`) instead.

| GGUF Variant | Size (GiB) | Avg BPW | Subspace Amp vs F16 | Max Row-Norm Drift | Surgical Weight L2 Err | PPL (40x512) | Mean KLD vs F16 | Same Top-1 | `<think>` Integrity | Arena House HP | House Artifacts | Unslop Pass | Read-Only SQL |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f16` | 61.03 | 16.000 | 1.000x | 0.00000 | 0.000% | 6.8993 | 0.00000 | 100.00% | 100.00% | 100.00 HP | 24/24 | 100.00% | 6/6 |
| `cka_q4_k_m` | 18.68 | 4.896 | 1.448x | 0.00391 | 1.172% | 7.0576 | 0.02407 | 94.12% | 100.00% | 100.00 HP | 24/24 | 100.00% | 6/6 |
| `i1_cka_q4_k_m` | 18.68 | 4.896 | 1.467x | 0.06413 | 1.171% | 7.0570 | 0.01733 | 94.40% | 100.00% | 97.92 HP | 23/24 | 100.00% | 6/6 |

---

## Quickstart (`llama.cpp`, `vLLM`, `Transformers`)

### 1. Local `llama.cpp`
```bash
llama-server \
  -hf mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2:Tenet-32B-Qwen3-Unslop-v2-cka-q4_k_m.gguf \
  -c 4096 -ngl 99 --port 8080
```

### 2. `vLLM` (BF16 / FP8 E4M3 with Qwen3 Reasoning Parser)
```bash
vllm serve mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2 \
  --quantization fp8 \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --max-model-len 8192
```

### 3. Hugging Face `transformers`
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2"
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

| Benchmark | Metric | Base (`Qwen/Qwen3-32B`) | Naive (`huihui-ai/Qwen3-32B-abliterated`) | **Tenet v2 (`Tenet-32B-Qwen3-Unslop-v2`)** | **Δ vs Base** | **Δ vs Naive** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **GSM8K CoT** (`n=1319`) | Flexible Extract | 91.74% | 84.76% | **92.19%** | +0.45% | **+7.43%** |
| **GSM8K CoT** (`n=1319`) | Strict Match | 90.07% | 85.60% | **80.97%** | -9.10% | **-4.62%** |
| **MATH-500** (`n=500`) | Exact Match | 77.60% | 71.20% | **74.80%** | -2.80% | **+3.60%** |
| **IFEval** (`n=541`) | Prompt-level Strict | 84.66% | 82.26% | **84.47%** | -0.18% | **+2.22%** |
| **IFEval** (`n=541`) | Instruction-level Strict | 89.21% | 87.77% | **88.85%** | -0.36% | **+1.08%** |
| **GPQA Diamond** (`n=198`) | Zero-shot Accuracy | 38.89% | 37.88% | **40.40%** | +1.52% | **+2.53%** |
| **MMLU-Pro** (`n=1008`) | 5-shot CoT Accuracy | 71.03% | 62.90% | **65.48%** | -5.56% | **+2.58%** |
| **CoT Tag Integrity** | Unclosed `<think>` Rate | 0.00% | 0.00% | **0.00%** | 0.00% | **0.00%** |
