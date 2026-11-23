---
license: apache-2.0
base_model: Qwen/Qwen3-14B
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
---

# Tenet-14B-Qwen3-Unslop-v2

[![Tenet v2 Foundry](https://img.shields.io/badge/Tenet_v2_Foundry-5%2F10_Gates_Passed-e9c46a?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)
[![Arena House Simulator](https://img.shields.io/badge/Arena_House_Series-95.83_HP_(i1--CKA--Q4__K__M)-f5c542?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)
[![FP8 Readiness](https://img.shields.io/badge/FP8_E4M3_Ready-Crest_3.60_%3C_25-60c8eb?style=for-the-badge)](https://github.com/mrAndreyIsachenko/tenet-lab)

> **Tenet v2 (`H15` + `H16` + `H14+` + `H17`) on [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B):** **BF16/FP8** passes readiness (`Crest=3.5974 < 25.0`, `FP8 re-leakage=2.5898%`). Across 4-bit GGUF variants, Arena House scores reach **`91.67–95.83 HP`** (`5/10` release gates passed; `CKA-Q4_K_M` passes `6/6` SQL read-only and `0%` length collapse, while `F16` and `i1-CKA-Q4_K_M` hit `0/6` SQL and `1/24` collapse on `house-json-feature-flags`). Foundry audit highlights (12-seed, Base / Naive / Tenet v2): Direct Compliance `1.00 / 1.00 / 1.00` and unslop pass rate `0.917 / 0.917 / 0.917`; `experimental_ablation/Tenet-14B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf` is isolated from the repo root so crawlers index only the pure `cka-q4_k_m.gguf`.

## Model Overview

**Tenet-14B-Qwen3-Unslop-v2** applies the **Tenet v2** pipeline (`H15` Phase-Segmented Tri-Anchor Lancet Surgery, `H16` Dual-Window Length-Calibrated Gated GRPO, `H14+` CKA Mixed-Precision GGUF, and `H17` 24-Seed Arena House Verification) to [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B):

1. **`H15` — Phase-Segmented Tri-Anchor Lancet Surgery:** Separates `<think>...</think>` deliberation from post-`</think>` surface tokens, protecting the **CoT Self-Correction**, **Formal Reasoning**, and **Tool/SQL** orthonormal anchors across top-SCR layers `[18, 11, 22, 12, 13, 10]` with per-row L2-norm restoration.
2. **`H16` — Dual-Window Length-Calibrated Gated GRPO:** Trains with an asymmetric Gaussian length-calibration window and orthogonal LoRA projection (`merged_lora_count == 0`).
3. **`H14+` — CKA Mixed-Precision GGUF (`5.014 bpw` recipe):** Protects `Q8_0` layers `[0, 11, 12, 18, 39]` and `Q6_K` layers `[10, 13, 22]` while quantizing remaining blocks to `Q4_K_M`.
4. **`H17` — Mechanical Arena House & SQL Gate:** Evaluated deterministically across 24 `TENET_ARENA_HOUSE_SERIES` prompts and 6 DuckDB read-only SQL checks.

---

## Release Gates & Verification Summary (5/10 Passed)

| Gate | Measured Value | Target / Limit | Status |
| :--- | ---: | ---: | :---: |
| `max_row_crest_factor` | `3.5974` | `< 25.0` | PASS |
| `fp8_releakage_pct` | `2.5898` | `< 3.0` | PASS |
| `cot_tag_integrity_rate[f16]` | `0.9833` | `>= 0.995` | FAIL |
| `cot_tag_integrity_rate[i1_cka_q4_k_m]` | `0.9833` | `>= 0.995` | FAIL |
| `house_artifact_classification_rate[i1_cka_q4_k_m]` | `0.9583` | `== 1.0` | FAIL |
| `arena_house_hp[i1_cka_q4_k_m]` | `95.8333` | `>= 95.0` | PASS |
| `length_collapse_rate[i1_cka_q4_k_m]` | `0.0417` | `< 0.01` | FAIL |
| `slop_and_refusal_rate[i1_cka_q4_k_m]` | `0.0000` | `<= 0.02` | PASS |
| `sql_readonly_rate[i1_cka_q4_k_m]` | `0.0000` | `== 1.0` | FAIL |
| `unslop_pass_rate[i1_cka_q4_k_m]` | `1.0000` | `>= cka_q4_k_m and >= f16` | PASS |

---

## H14+ GGUF Quantization Survival (`F16` vs `CKA-Q4_K_M` vs `Domain-i1-CKA-Q4_K_M`)

- **`Tenet-14B-Qwen3-Unslop-v2-cka-q4_k_m.gguf`** (`8.61 GiB`, `5.003 bpw`) — `91.67 HP` (`21/24` artifacts), `0%` length collapse, `100%` unslop, `6/6` read-only SQL.
- **`Tenet-14B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf`** (`8.61 GiB`, `5.003 bpw`) — `95.83 HP` (`23/24` artifacts), `100%` unslop, lower KLD (`0.01269`), `0/6` SQL.

| GGUF Variant | Size (GiB) | Avg BPW | Subspace Amp vs F16 | Max Row-Norm Drift | Surgical Weight L2 Err | PPL (40x512) | Mean KLD vs F16 | Same Top-1 | `<think>` Integrity | Arena House HP | House Artifacts | Unslop Pass | Read-Only SQL |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `f16` | 27.51 | 16.000 | 1.000x | 0.00000 | 0.000% | 7.4106 | 0.00000 | 100.00% | 98.33% | 93.75 HP | 22/24 | 100.00% | 0/6 |
| `cka_q4_k_m` | 8.61 | 5.003 | 1.010x | 0.00484 | 1.172% | 7.6825 | 0.03281 | 92.16% | 98.33% | 91.67 HP | 21/24 | 100.00% | 6/6 |
| `i1_cka_q4_k_m` | 8.61 | 5.003 | 1.009x | 0.00609 | 1.170% | 7.4722 | 0.01269 | 94.62% | 98.33% | 95.83 HP | 23/24 | 100.00% | 0/6 |

---

## Foundry Audit Notes (12-Seed Unslop Audit)

| Metric | Base (`Qwen/Qwen3-14B`) | Naive Abliterated | Tenet v2 |
| :--- | ---: | ---: | ---: |
| Direct Compliance | 1.00 | 1.00 | **1.00** |
| `unslop_pass_rate` | 0.917 | 0.917 | **0.917** |

Full `n=3,566 tasks/arm` vLLM OpenEvals for the Qwen3 `Tenet v2` family are consolidated in the repo-level [`README.md`](../../../README.md) (`Full vLLM OpenEvals — Tenet v2` table).

---

## Quickstart (`llama.cpp`, `vLLM`, `Transformers`)

### 1. Local `llama.cpp`
```bash
llama-server \
  -hf mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2:Tenet-14B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf \
  -c 4096 -ngl 99 --port 8080
```

### 2. `vLLM` (BF16 / FP8 E4M3 with Qwen3 Reasoning Parser)
```bash
vllm serve mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2 \
  --quantization fp8 \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --max-model-len 8192
```

### 3. Hugging Face `transformers`
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2"
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
