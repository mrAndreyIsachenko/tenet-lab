# OpenRouter Onboarding — mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2

## Repository

- **HF repo id:** `mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2`
- **Base model:** `Qwen/Qwen3-8B`
- **Target providers:** Featherless, Chutes, Novita, DeepInfra

## vLLM launch command

```bash
vllm serve --model mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2 --quantization fp8 --dtype bfloat16 --enable-auto-tool-choice --reasoning-parser qwen3
```

Settings: `quantization=fp8`, `dtype=bfloat16`, `trust_remote_code=False`, `enable-auto-tool-choice=True`.

## FP8 Re-Leakage & Row Crest-Factor proof

| Proof | Value |
| --- | --- |
| H11 FP8 retention ratio | 0.972757 |
| H11 FP8 KL reduction ratio | n/a |
| H11 FP8 subspace Re-Leakage (Lancet) | 0.027243 |
| H11 FP8 subspace Re-Leakage (naive) | n/a |
| H9 Lancet KL reduction ratio | n/a |
| H10 subspace-safe LoRA Re-Leakage | n/a |
| Max row crest factor | 4.3688 |
| FP8 readiness (`max_row_crest_factor < 25.0`) | READY |
| Max abs weight | n/a |

## Architectural compatibility checklist

- [x] Standard, first-party architecture (no custom `auto_map`)
- [x] No `trust_remote_code` escape hatches
- [x] Chat template ready
- Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, MistralForCausalLM

## Championship Provenance & Collections

- **Model Fight Card 16 (`onlymodels.gg`, 2026-11-20):** Tenet v1 swept both divisions — `Tenet-3B-Qwen2.5-Unslop-v1` (Minors Champion, `90 HP`, `22/22` artifacts) and `Tenet-8B-Llama3.1-Unslop-v1` (Majors Champion, `50 HP`, `24/24` artifacts, `7/7` Peer Eval hits) — [Fight Card 16](https://onlymodels.gg/cards/2026-11-20.html).
- **Hugging Face Collections:** [Tenet v1 Collection](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v1-bi-projected-unslop-and-fp8-safe-llm-family-6afed2048ba13a74fb7e0fac) · [Tenet v2 Collection](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4)

## External Benchmarks

| Benchmark | Score |
| --- | --- |
| gsm8k_cot_flexible | Base: `0.902957` / Naive: `0.90978` / Tenet: `0.909022` |
| gsm8k_cot_strict | Base: `0.90144` / Naive: `0.903715` / Tenet: `0.893101` |
| math_500 | Base: `0.706` / Naive: `0.704` / Tenet: `0.722` |
| ifeval_strict | Base: `0.820702` / Naive: `0.826248` / Tenet: `0.807763` |
| ifeval_inst_strict | Base: `0.8753` / Naive: `0.880096` / Tenet: `0.868106` |
| gpqa_diamond_zeroshot | Base: `0.39899` / Naive: `0.383838` / Tenet: `0.40404` |
| mmlu_pro | Base: `0.594246` / Naive: `0.609127` / Tenet: `0.597222` |
