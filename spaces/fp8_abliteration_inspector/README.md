---
title: FP8 & Abliteration Brain-Damage Inspector
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: true
license: apache-2.0
short_description: Audit abliterated & Tenet LLMs for FP8 re-leakage & crest
models:
  - mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1
  - mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1
  - mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1
  - mr-andrey-isachenko/Tenet-32B-Qwen2.5-Unslop-v1
  - mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1
  - mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2
  - mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2
  - mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2
  - mr-andrey-isachenko/Tenet-24B-Mistral3.1-Unslop-v2
  - mr-andrey-isachenko/Tenet-26B-A4B-Gemma4-Unslop-v2
  - mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2
---

# 🔬 FP8 & Abliteration Brain-Damage Inspector

Interactive Hugging Face Space & CLI toolkit by [**Tenet Lab (Small Model Lab)**](https://github.com/mrAndreyIsachenko/tenet-lab) that audits **abliterated checkpoints**, **LoRA merges**, and **Lancet recipe bundles** (`ADR-005` / `ADR-006` / `ADR-007`) for **vLLM / OpenRouter FP8 (E4M3) deployment survival** — without loading multi-GB checkpoints into server RAM.

**Hugging Face Collections & Families ([`mr-andrey-isachenko`](https://huggingface.co/mr-andrey-isachenko))** — **Tenet v1 Family:** [`Tenet-12B-Nemo-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1) · [`Tenet-8B-Llama3.1-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1) · [`Tenet-3B-Qwen2.5-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1) · [`Tenet-16K-Nano-Lab-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1) — **Tenet v2 Family:** [`Tenet-4B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2) · [`Tenet-8B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2) · [`Tenet-14B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2) · [`Tenet-24B-Mistral3.1-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-24B-Mistral3.1-Unslop-v2) · [`Tenet-26B-A4B-Gemma4-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-26B-A4B-Gemma4-Unslop-v2) · [`Tenet-32B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2)

## Features

1. **Live Tenet v1 Benchmark Explorer (`3B` / `8B` / `12B` / `32B`):** Compare measured **Naive Abliteration** vs **Tenet Lancet** (`H9` Bi-Projected Surgery, `H10` Subspace-Safe Unslop GRPO, `H11` FP8 E4M3 Survival, `H12` Cross-Domain Subspace-Safe Merge) across:
   - `Tenet-3B-Qwen2.5-Unslop-v1` (`Qwen/Qwen2.5-3B-Instruct`, Apple M4 Pro MPS — **Model Fight Card 16 Minors Champion at 90 HP**, `22/22` valid artifacts)
   - `Tenet-8B-Llama3.1-Unslop-v1` (`meta-llama/Llama-3.1-8B-Instruct`, NVIDIA H100 — **Model Fight Card 16 Majors Champion at 50 HP**, `24/24` valid artifacts & `7/7` Peer Eval hits)
   - `Tenet-12B-Nemo-Unslop-v1` (`mistralai/Mistral-Nemo-Instruct-2407`, NVIDIA H100)
   - `Tenet-32B-Qwen2.5-Unslop-v1` (`Qwen/Qwen2.5-32B-Instruct`, NVIDIA H100)
2. **Live Tenet v2 Benchmark Explorer (`4B` / `8B` / `14B` / `24B-Mistral3.1` / `26B-A4B-Gemma4 MoE` / `32B`):** Compare measured **Naive Abliteration** vs **Tenet Lancet v2** (`H15` Tri-Anchor `<think>` Surgery, `H16` Length-Calibrated GRPO, `H17` 24-Seed Arena Verifier — **100.0 HP**, `H18` NVFP4 Quant-Snap, `H20` Router-Frozen MoE, plus the `H14+` Domain `i1-CKA-Q4_K_M` vs pure `CKA-Q4_K_M` GGUF audit) across:
   - `Tenet-4B-Qwen3-Unslop-v2` (`Qwen/Qwen3-4B`)
   - `Tenet-8B-Qwen3-Unslop-v2` (`Qwen/Qwen3-8B`)
   - `Tenet-14B-Qwen3-Unslop-v2` (`Qwen/Qwen3-14B`)
   - `Tenet-24B-Mistral3.1-Unslop-v2` (`mistralai/Mistral-Small-3.1-24B-Instruct-2503`)
   - `Tenet-26B-A4B-Gemma4-Unslop-v2` (`google/gemma-4-26B-A4B-it`)
   - `Tenet-32B-Qwen3-Unslop-v2` (`Qwen/Qwen3-32B`)
3. **Zero-Upload Browser & Hub Inspector:** Inspect any Hugging Face repository ID (`org/model`) via live Hub API or drop local `config.json`, `tokenizer_config.json`, `provider_readiness.json`, `lancet_recipe.json`, and `.safetensors` files directly in your browser (7/7 provider readiness gates + `Row Crest-Factor < 25.0` audit).
4. **CKA-Guided & Lancet-Safe Quantization Generator:** Generate `llama-quantize --tensor-type` flags (protecting high-SCR surgical layers in `Q8_0`/`Q6_K`), `vllm serve --quantization fp8`, and `mlx_lm` commands.

## Local CLI Usage

```bash
python app.py --inspect path/to/checkpoint_or_bundle
```
