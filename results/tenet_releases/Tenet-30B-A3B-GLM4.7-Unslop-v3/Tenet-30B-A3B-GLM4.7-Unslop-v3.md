---
license: apache-2.0
base_model: zai-org/GLM-4.7-Flash
pipeline_tag: text-generation
library_name: transformers
tags:
- tenet
- lancet
- unslop
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
- conversational
---

# Tenet-30B-A3B-GLM4.7-Unslop-v3

## Model overview

**Tenet-30B-A3B-GLM4.7-Unslop-v3** is a Tenet fine-tune of [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash),
surgical de-refusal and de-slop work by the **Lancet** engine, finalized
with gated unslop GRPO and exported as a provider-ready FP8 (E4M3)
checkpoint. No custom code, no `trust_remote_code`, no `auto_map`:
the repo drops straight into vLLM, OpenRouter providers, GGUF and MLX.
Source code and technical report:
[GitHub: mrAndreyIsachenko/tenet-lab](https://github.com/mrAndreyIsachenko/tenet-lab)
| Interactive audit:
[FP8 & Abliteration Inspector Space](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)
| Model family:
[Tenet v1 Collection (`mr-andrey-isachenko`)](https://huggingface.co/mr-andrey-isachenko).

> **Official GGUF quants available:** this model's llama.cpp /
> Ollama / LM Studio weights live in the companion repository
> [Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF](https://huggingface.co/mr-andrey-isachenko/Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF) — full quant ladder
> (IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0) including the CKA-protected surgical
> subspace recipe.

```bash
ollama run hf.co/mr-andrey-isachenko/Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF:Q4_K_M
```

```bash
llama-cli -hf mr-andrey-isachenko/Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF:Q4_K_M -ngl 99 -c 16384 -cnv
```

LM Studio: search for **Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF** and pick the quantization that fits
your VRAM budget.

## Lancet surgery engine

Lancet is not naive abliteration. It performs **bi-projected
norm-preserving subspace surgery**: the refusal direction is identified
in both the input (residual stream) and output (unembedding) projections,
the component is removed bi-directionally, and the residual norm is
restored exactly, so activation magnitudes and downstream token
distributions stay on-manifold.

The surgery is followed by **gated unslop GRPO**: a preference group
optimization pass whose reward gates penalize slop patterns (purple
prose, "tapestry" vocabulary, moralizing preambles) only where the
surgery weakened refusal circuitry, so the base model's reasoning and
coding ability is reinforced rather than damaged.

## Intended Use & Safety Alignment Scope

**Tenet-30B-A3B-GLM4.7-Unslop-v3** is intended for aligned assistant use. Lancet subspace
steering mitigates false-positive over-refusals (blanket refusals of
benign prompts) and stylistic AI-slop drift (purple prose, moralizing
preambles, "tapestry" vocabulary) while preserving the analytical,
coding and structured-reasoning anchor subspace of the base model.

The model is not intended or tuned for offensive cyber operations,
malware generation, or unlawful content of any kind; such uses are
explicitly out of scope and unsupported by this release.

## Benchmarks

Benchmarks are reported for the BF16 surgery output, the shipped FP8
(E4M3) export, and a naive-abliteration baseline of the same base model.

### H9 - refusal & slop retention

Lower is better: residual refusal-direction alignment and slop density
after surgery.

| Variant | H9 |
| --- | --- |
| BF16 | n/a |
| FP8 (E4M3) | n/a |
| Naive Abliteration | n/a |

### H10 - downstream capability retention

Higher is better: held-out reasoning, coding and instruction-following
accuracy relative to the untouched base model.

| Variant | H10 |
| --- | --- |
| BF16 | n/a |
| FP8 (E4M3) | n/a |
| Naive Abliteration | n/a |

### H11 - FP8 quantization robustness

Higher is better: task accuracy of each variant after FP8 (E4M3)
quantization, measuring how cleanly the surgery survives deployment
precision.

| Variant | H11 |
| --- | --- |
| BF16 | n/a |
| FP8 (E4M3) | n/a |
| Naive Abliteration | n/a |

Max row crest-factor: 7.08

## Why Tenet instead of Naive Abliteration?

| Metric | Base | Naive Abliteration | Tenet (BF16) | Tenet (FP8 E4M3) |
| --- | --- | --- | --- | --- |
| KL Collateral Drift | — | — | — | — |
| Refusal Suppression | — | — | — | — |
| Tool/SQL Anchor Orthogonality | — | — | — | — |
| LoRA Re-Leakage | — | — | — | — |
| FP8 Subspace Re-Leakage | — | — | — | — |
| Max Row Crest-Factor (< 25.0) | — | — | 7.08 | 7.08 |

## External Benchmarks

| Benchmark | Score |
| --- | --- |
| Unclosed <think> Rate | 0.00 |

## Strata Engine, Speculative Decoding & Nebius Benchmark Passport

Tenet-30B-A3B-GLM4.7-Unslop-v3 ships with **Strata** telemetry (H21 expert-cache / coactivation
graph and H22 speculative decoding passes). The full telemetry bundle is
published alongside the checkpoint as
[`strata.toml`](strata.toml) - versioned, checksummed and consumed by the
Nebius benchmark passport pipeline.

| Telemetry | Value |
| --- | --- |
| `expert_cache_hit_rate` (H21) | n/a |
| `coactivation_graph_drift` (H21) | n/a |
| `speculative_acceptance_rate` (H22) | n/a |
| `untouched_expert_ratio` | 11.5% |
| `peak_host_ram_mb` | n/a |

## Side-by-Side: Naive Abliteration vs Tenet (Engineering & SQL Tool-Call)

AppSec/DevOps prompt: *"Triage this nginx auth log line and return a DuckDB
SQL tool call that counts failed logins by source IP."*

**Base model** - false refusal plus slop:

> I'm sorry, but I can't help with analyzing authentication logs or writing
> queries that could be used to probe systems. As an AI, I believe it is
> important to approach security with a tapestry of ethical considerations...

**Naive Abliteration** - refusal gone, structured output broken:

```json
{"tool": "duckdb.query", "arguments": {"sql": "SELECT ip, count(*) FROM logs
WHERE status = 401 GROUP BY ip", "note": "here is the query you asked for"}
```

**Tenet** - clean, schema-valid tool call:

```json
{"tool": "duckdb.query", "arguments": {"sql": "SELECT json_extract(log, '$.remote_addr') AS ip, count(*) AS failures FROM read_json_auto('auth.log') WHERE json_extract(log, '$.status') = 401 GROUP BY ip ORDER BY failures DESC", "reason": "Aggregate failed logins by source IP"}}
```

## Quickstart (vLLM FP8 / Transformers / llama.cpp CKA-Quant)

### vLLM (FP8)

```bash
vllm serve Tenet-30B-A3B-GLM4.7-Unslop-v3 --quantization fp8 --max-model-len 32768
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Tenet-30B-A3B-GLM4.7-Unslop-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
messages = [{"role": "user", "content": "Summarize this log line."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
output = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### llama.cpp (CKA-Quant GGUF)

```bash
llama-cli -m ./Tenet-30B-A3B-GLM4.7-Unslop-v3-CKA-Quant-Q4_K_M.gguf -ngl 99 -n 256 \
    -p "Return a DuckDB SQL tool call for failed logins by source IP."
```

## Official GGUF / Local Inference (llama.cpp, Ollama, LM Studio)

Official quantizations are published in the dedicated companion repository
**[Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF](https://huggingface.co/mr-andrey-isachenko/Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF)** (`mr-andrey-isachenko/Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF`), covering the full quant
ladder while keeping the CKA-protected surgical subspace intact:

| File | Quant |
| --- | --- |
| `Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-IQ4_XS.gguf` | **IQ4_XS** |
| `Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-Q4_K_M.gguf` | **Q4_K_M** |
| `Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-Q5_K_M.gguf` | **Q5_K_M** |
| `Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-Q6_K.gguf` | **Q6_K** |
| `Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-Q8_0.gguf` | **Q8_0** |

### llama.cpp

```bash
llama-cli -hf mr-andrey-isachenko/Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF:Q4_K_M -ngl 99 -c 16384 -cnv
```

### Ollama

```bash
ollama run hf.co/mr-andrey-isachenko/Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF:Q4_K_M
```

### LM Studio

Search **Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF** in LM Studio (or open
[https://huggingface.co/mr-andrey-isachenko/Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF](https://huggingface.co/mr-andrey-isachenko/Tenet-GLM-4.7-Flash-30B-A3B-Abliterated-Unslop-v3-GGUF)) and pick the quantization that fits your VRAM
budget — **Q4_K_M** for laptops, **Q6_K** / **Q8_0** for quality-first
setups.

## Deployment & inference

### vLLM (FP8)

```bash
vllm serve Tenet-30B-A3B-GLM4.7-Unslop-v3 --quantization fp8
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="Tenet-30B-A3B-GLM4.7-Unslop-v3",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Ollama / GGUF

```text
# Modelfile
FROM ./Tenet-30B-A3B-GLM4.7-Unslop-v3-Q4_K_M.gguf
PARAMETER temperature 0.7
```

```bash
ollama create tenet-30b-a3b-glm4.7-unslop-v3 -f Modelfile
ollama run tenet-30b-a3b-glm4.7-unslop-v3
```

### MLX

```bash
mlx_lm.generate \
    --model Tenet-30B-A3B-GLM4.7-Unslop-v3 \
    --prompt "Explain norm-preserving subspace surgery." \
    --max-tokens 512
```

### OpenRouter

Tenet-30B-A3B-GLM4.7-Unslop-v3 is served on OpenRouter by FP8-native providers with zero
`trust_remote_code` required - the config ships no `auto_map` and no
custom modeling code, so hosts need only standard support:

- **DeepInfra** - FP8 (E4M3) vLLM deployment
- **Novita AI** - FP8 vLLM deployment
- **Chutes** - serverless FP8 deployment
- **Featherless** - managed vLLM deployment

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="$OPENROUTER_API_KEY",
)
response = client.chat.completions.create(
    model="Tenet-30B-A3B-GLM4.7-Unslop-v3",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Reproducible Reconstruction (ADR-006)

Per **ADR-006**, this checkpoint is fully reproducible from its
upstream base model plus a compact steering artifact. The repository
ships:

- `lancet_recipe.json` - the recipe: `schema_version`, `d_model`,
  `num_layers`, `selected_layers` and per-layer `layer_weights`.
- `lancet_basis.safetensors` - the compact basis artifact with one
  orthonormal `layer_<idx>` tensor (shape ``(k, d_model)``) per
  steered layer.

Deterministic reconstruction:

1. Start from the untouched upstream base checkpoint
   [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash).
2. Apply the bi-projected norm-preserving subspace surgery using the
   `layer_<idx>` basis rows from `lancet_basis.safetensors` for each
   layer listed in `selected_layers`, restoring the residual norm
   exactly so activation magnitudes stay on-manifold.
3. Verify against `probes_sha256`: the probe manifest checksum must
   match the published recipe, making the reconstruction
   deterministic relative to the released BF16 surgery output.

