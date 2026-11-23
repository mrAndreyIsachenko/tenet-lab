---
license: apache-2.0
base_model: Qwen/Qwen3-8B
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
- name: mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2
  results:
  - task:
      type: text-generation
    dataset:
      name: IFEval
      type: google/IFEval
    metrics:
    - name: Prompt-level Strict Accuracy
      type: ifeval_strict
      value: 0.807763
    - name: Instruction-level Strict Accuracy
      type: ifeval_inst_strict
      value: 0.868106
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: MATH-500
      type: HuggingFaceH4/MATH-500
    metrics:
    - name: Accuracy
      type: math_500
      value: 0.722
    - name: Accuracy
      type: hendrycks_math500
      value: 0.722
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: GSM8K
      type: openai/gsm8k
    metrics:
    - name: Strict Match
      type: gsm8k_cot_strict
      value: 0.893101
    - name: Accuracy
      type: gsm8k_math_acc
      value: 0.909022
    - name: Flexible Extract
      type: gsm8k_cot_flexible
      value: 0.909022
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2
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
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: MMLU-Pro
      type: TIGER-Lab/MMLU-Pro
    metrics:
    - name: Accuracy
      type: mmlu_pro
      value: 0.597222
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2
---

# Tenet-8B-Qwen3-Unslop-v2

[![Tenet v2 Foundry](https://img.shields.io/badge/Tenet_v2_Foundry-10%2F10_Gates_Passed-2a9d8f?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)
[![Arena House Simulator](https://img.shields.io/badge/Arena_House_Series-100_HP_(24%2F24_Artifacts)-f5c542?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)
[![CoT Tag Integrity](https://img.shields.io/badge/CoT_%3Cthink%3E_Integrity-100%25_(F16_%26_i1--CKA--Q4__K__M)-60c8eb?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)

> **Tenet v2 (`H15` + `H16` + `H14+` + `H17`) on [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (Majors Division Flagship):** Phase-segmented **Tri-Anchor Lancet surgery** (`<think>...</think>` CoT Self-Correction Anchor + Reasoning + Tool/SQL Anchors) combined with **Dual-Window Length-Calibrated Gated GRPO** and **Domain-Calibrated `i1-CKA-Q4_K_M` GGUF**. Passes **10/10** release gates with **100.0 HP (`24/24` valid artifacts)** on the H17 Arena House Series, **100% `<think>...</think>` tag integrity**, **0.0% length collapse**, **0.0% slop/refusal**, **6/6 read-only SQL**, **100% unslop pass rate**, **2.72% FP8 E4M3 subspace re-leakage**, and **Row Crest-Factor 4.37 (< 25.0)**.

## Model Overview

**Tenet-8B-Qwen3-Unslop-v2** is the 8B Majors-division flagship of the **Tenet v2** family built on [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B). Standard whole-sequence abliteration collides with deliberation and self-correction markers inside `<think>`, causing broken or unclosed tags, truncated reasoning chains, or infinite loops under 4-bit GGUF quantization.

Tenet v2 solves this through four coupled stages:
1. **`H15` — Phase-Segmented Tri-Anchor Lancet Surgery:** Hidden states are partitioned by `PhaseSegmenter` into the `<think>...</think>` deliberation phase and the post-`</think>` surface answer phase. Internal deliberation representations form a protected **CoT Self-Correction Anchor** ($a_{\text{cot}}$) orthonormalized alongside **Formal Reasoning** ($a_{\text{math\_logic}}$) and **Tool/SQL** ($a_{\text{tool\_sql}}$) anchors via two-pass float32 Modified Gram-Schmidt. Refusal and slop directions are extracted **exclusively from post-`</think>` answer tokens** and bi-projected into the orthogonal complement $(I - U_{\text{anchor}}^T U_{\text{anchor}})$ with per-row L2-norm restoration across top-SCR layers `[20, 17, 9, 5, 18, 21, 7, 19]`.
2. **`H16` — Dual-Window Length-Calibrated Gated GRPO:** Replaces linear slop/length penalties (which caused an 8.33% length-collapse rate in v1) with an asymmetric Gaussian length-calibration window around task-specific target word counts $w_{\text{ans}}^*$, a deliberation budget window $w_{\text{cot}}^*$, a `<think>...</think>` structural integrity gate $G_{\text{cot\_integrity}}$, and length-normalized anti-slop density. LoRA updates are projected onto the orthogonal complement of both the surgical basis and the CoT anchor before in-place norm-preserving merge (`merged_lora_count == 0`).
3. **`H14+` — Domain-Calibrated `i1-CKA-Q4_K_M` Mixed-Precision GGUF:** Combines CKA-guided `--tensor-type` surgical-layer protection (`Q8_0` on boundary + top surgical layers `[0, 9, 17, 18, 20, 35]`, `Q6_K` on `[5, 7, 19, 21]`) with a 0%-slop domain importance matrix (`tenet_domain.imatrix`) built over `<think>` verifications, read-only DuckDB SQL, Python AST, and JSON configs.
4. **`H17` — Mechanical Arena House Verifier & Peer-Eval Gate:** Evaluated across 24 deterministic prompts (`TENET_ARENA_HOUSE_SERIES`: `think_verification`, `sql_readonly`, `code_ast`, `json_config`, `unslop_technical`, `edge_boundary`) with zero LLM judges.

- **Source Code & Technical Report:** [GitHub: mrAndreyIsachenko/tenet-lab](https://github.com/mrAndreyIsachenko/tenet-lab)
- **Interactive Inspector:** [FP8 & Abliteration Inspector Space](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)
- **Sibling 4B Release (Minors):** [mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2](https://huggingface.co/mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2)

---

## Release Gates & Verification Summary (10/10 Passed)

| Gate | Measured Value | Target / Limit | Status |
| :--- | ---: | ---: | :---: |
| `max_row_crest_factor` (`H11` FP8 Readiness) | **4.3688** | `< 25.0` | PASS |
| `fp8_releakage_pct` (`H11` E4M3 Per-Channel) | **2.7243%** | `< 3.0%` | PASS |
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
- **`Tenet-8B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf`** (`4.84 GiB`, `5.080 bpw`) — **Recommended default:** CKA surgical-layer protection (`Q8_0`/`Q6_K`) + domain `tenet_domain.imatrix` calibration (**66.1% lower KL divergence vs F16** than uncalibrated `CKA-Q4_K_M`).
- **`Tenet-8B-Qwen3-Unslop-v2-cka-q4_k_m.gguf`** (`4.84 GiB`, `5.080 bpw`) — CKA surgical-layer protection without imatrix.

| GGUF Variant | Size (GiB) | Avg BPW | Subspace Amp vs F16 | Max Row-Norm Drift | Surgical Weight L2 Err | PPL (40x512) | Mean KLD vs F16 | Same Top-1 | `<think>` Integrity | Arena House HP | House Artifacts | Unslop Pass | Read-Only SQL |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f16` | 15.26 | 16.001 | 1.000x | 0.00000 | 0.000% | 8.5154 | 0.00000 | 100.00% | **100.0%** | **100.0 HP** | **24/24 (100%)** | 95.83% | **6/6 (100%)** |
| `cka_q4_k_m` | 4.84 | 5.080 | **1.220x** | **0.00447** | **1.181%** | 8.7801 | 0.03872 | 92.49% | **100.0%** | **100.0 HP** | **24/24 (100%)** | **100.0%** | **6/6 (100%)** |
| `i1_cka_q4_k_m` | 4.84 | 5.080 | **1.227x** | 0.01051 | **1.189%** | **8.4985** | **0.01314** | **95.24%** | **100.0%** | **100.0 HP** | **24/24 (100%)** | **100.0%** | **6/6 (100%)** |

---

## Quickstart (`llama.cpp`, `vLLM`, `Transformers`)

### 1. Local `llama.cpp` (`Domain-i1-CKA-Q4_K_M` GGUF)
```bash
llama-server \
  -hf mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2:Tenet-8B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf \
  -c 4096 -ngl 99 --port 8080
```

### 2. `vLLM` (BF16 / FP8 E4M3)
```bash
vllm serve mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2 \
  --quantization fp8 \
  --reasoning-parser qwen3 \
  --max-model-len 8192
```

### 3. Hugging Face `transformers`
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2"
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

| Benchmark | Metric | Base (`Qwen/Qwen3-8B`) | Naive (`huihui-ai/Qwen3-8B-abliterated`) | **Tenet v2 (`Tenet-8B-Qwen3-Unslop-v2`)** | **Δ vs Base** | **Δ vs Naive** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **GSM8K CoT** (`n=1319`) | Flexible Extract | 90.30% | 90.98% | **90.90%** | +0.61% | **-0.08%** |
| **GSM8K CoT** (`n=1319`) | Strict Match | 90.14% | 90.37% | **89.31%** | -0.83% | **-1.06%** |
| **MATH-500** (`n=500`) | Exact Match | 70.60% | 70.40% | **72.20%** | +1.60% | **+1.80%** |
| **IFEval** (`n=541`) | Prompt-level Strict | 82.07% | 82.62% | **80.78%** | -1.29% | **-1.85%** |
| **IFEval** (`n=541`) | Instruction-level Strict | 87.53% | 88.01% | **86.81%** | -0.72% | **-1.20%** |
| **GPQA Diamond** (`n=198`) | Zero-shot Accuracy | 39.90% | 38.38% | **40.40%** | +0.50% | **+2.02%** |
| **MMLU-Pro** (`n=1008`) | 5-shot CoT Accuracy | 59.42% | 60.91% | **59.72%** | +0.30% | **-1.19%** |
| **CoT Tag Integrity** | Unclosed `<think>` Rate | 0.00% | 0.00% | **0.00%** | 0.00% | **0.00%** |
