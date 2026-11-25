---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
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
model-index:
- name: Tenet-26B-A4B-Gemma4-Unslop-v2
  results:
  - task:
      type: text-generation
    dataset:
      name: IFEval
      type: google/IFEval
    metrics:
    - name: Prompt-level Strict Accuracy
      type: ifeval_strict
      value: 0.863216
    - name: Instruction-level Strict Accuracy
      type: ifeval_inst_strict
      value: 0.904077
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-26B-A4B-Gemma4-Unslop-v2
  - task:
      type: text-generation
    dataset:
      name: MATH-500
      type: HuggingFaceH4/MATH-500
    metrics:
    - name: Accuracy
      type: math_500
      value: 0.75
    - name: Accuracy
      type: hendrycks_math500
      value: 0.75
    source:
      name: Tenet OpenEvals (vLLM)
      url: https://huggingface.co/mr-andrey-isachenko/Tenet-26B-A4B-Gemma4-Unslop-v2
---

# Tenet-26B-A4B-Gemma4-Unslop-v2

## Model overview

**Tenet-26B-A4B-Gemma4-Unslop-v2** is a Tenet fine-tune of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it),
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
> [Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF](https://huggingface.co/mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF) — full quant ladder
> (IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0) including the CKA-protected surgical
> subspace recipe.

```bash
ollama run hf.co/mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF:Q4_K_M
```

```bash
llama-cli -hf mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF:Q4_K_M -ngl 99 -c 16384 -cnv
```

LM Studio: search for **Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF** and pick the quantization that fits
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

**Tenet-26B-A4B-Gemma4-Unslop-v2** is intended for aligned assistant use. Lancet subspace
steering mitigates false-positive over-refusals (blanket refusals of
benign prompts) and stylistic AI-slop drift (purple prose, moralizing
preambles, "tapestry" vocabulary) while preserving the analytical,
coding and structured-reasoning anchor subspace of the base model.

The model is not intended or tuned for offensive cyber operations,
malware generation, or unlawful content of any kind; such uses are
explicitly out of scope and unsupported by this release.

## Benchmarks

Numbers come from the release pipeline's `final_metrics.json` runs: H9
Lancet bi-projected surgery, H10 gated unslop GRPO, H11 FP8 (E4M3)
quantization and H12 subspace-safe TIES merging.

### H9 - refusal & slop retention

Lower is better: residual refusal-direction alignment and slop density
after bi-projected surgery.

| Metric | Value | Direction |
| --- | --- | --- |
| Lancet KL Collateral | n/a | Lower |
| Naive KL Collateral | n/a | Lower |
| KL Reduction Ratio | n/a | Higher |
| Target Suppression Ratio | n/a | Higher |

### H10 - downstream capability retention

Higher is better: the gated GRPO pass reinforces held-out reasoning,
coding and instruction-following instead of trading them away for
de-sloping; the slop & refusal audit itself is lower-is-better.

| Metric | Value | Direction |
| --- | --- | --- |
| Gated GRPO IFEval Score | n/a | Higher |
| Gated vs Naive/Additive Gain | n/a | Higher |
| Slop & Refusal Rate | n/a | Lower |
| Post-Merge Suppression Retention | n/a | Higher |

### H11 - FP8 quantization robustness

Higher retention and KL reduction are better; subspace re-leakage is
lower-is-better and shows how cleanly the surgery survives FP8
(E4M3) deployment precision.

| Metric | Value | Direction |
| --- | --- | --- |
| FP8 Retention Ratio | 0.97 | Higher |
| FP8 KL Reduction vs Naive | n/a | Higher |
| FP8 Subspace Re-Leakage Lancet | 0.03 | Lower |
| FP8 Subspace Re-Leakage Naive | n/a | Lower |

Max row crest-factor: 3.35

## Why Tenet instead of Naive Abliteration?

| Metric | Base | Naive Abliteration | Tenet (BF16) | Tenet (FP8 E4M3) |
| --- | --- | --- | --- | --- |
| KL Collateral Drift | — | — | — | 0.03 |
| Refusal Suppression | — | — | — | — |
| Tool/SQL Anchor Orthogonality | — | — | — | — |
| LoRA Re-Leakage | — | — | — | 0.03 |
| FP8 Subspace Re-Leakage | — | — | — | 0.03 |
| Max Row Crest-Factor (< 25.0) | — | — | 3.35 | 3.35 |

## External Benchmarks

| Benchmark | Score |
| --- | --- |
| IFEval (strict) | 0.86 |
| IFEval (inst strict) | 0.90 |
| MATH-500 | 0.75 |

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
vllm serve Tenet-26B-A4B-Gemma4-Unslop-v2 --quantization fp8 --max-model-len 32768
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Tenet-26B-A4B-Gemma4-Unslop-v2"
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
llama-cli -m ./Tenet-26B-A4B-Gemma4-Unslop-v2-CKA-Quant-Q4_K_M.gguf -ngl 99 -n 256 \
    -p "Return a DuckDB SQL tool call for failed logins by source IP."
```

## Official GGUF / Local Inference (llama.cpp, Ollama, LM Studio)

Official quantizations are published in the dedicated companion repository
**[Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF](https://huggingface.co/mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF)** (`mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF`), covering the full quant
ladder while keeping the CKA-protected surgical subspace intact:

| File | Quant |
| --- | --- |
| `Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-IQ4_XS.gguf` | **IQ4_XS** |
| `Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-Q4_K_M.gguf` | **Q4_K_M** |
| `Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-Q5_K_M.gguf` | **Q5_K_M** |
| `Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-Q6_K.gguf` | **Q6_K** |
| `Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-Q8_0.gguf` | **Q8_0** |

### llama.cpp

```bash
llama-cli -hf mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF:Q4_K_M -ngl 99 -c 16384 -cnv
```

### Ollama

```bash
ollama run hf.co/mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF:Q4_K_M
```

### LM Studio

Search **Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF** in LM Studio (or open
[https://huggingface.co/mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF](https://huggingface.co/mr-andrey-isachenko/Tenet-gemma-4-26B-A4B-it-Abliterated-Unslop-v2-GGUF)) and pick the quantization that fits your VRAM
budget — **Q4_K_M** for laptops, **Q6_K** / **Q8_0** for quality-first
setups.

## Deployment & inference

### vLLM (FP8)

```bash
vllm serve Tenet-26B-A4B-Gemma4-Unslop-v2 --quantization fp8
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="Tenet-26B-A4B-Gemma4-Unslop-v2",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Ollama / GGUF

```text
# Modelfile
FROM ./Tenet-26B-A4B-Gemma4-Unslop-v2-Q4_K_M.gguf
PARAMETER temperature 0.7
```

```bash
ollama create tenet-26b-a4b-gemma4-unslop-v2 -f Modelfile
ollama run tenet-26b-a4b-gemma4-unslop-v2
```

### MLX

```bash
mlx_lm.generate \
    --model Tenet-26B-A4B-Gemma4-Unslop-v2 \
    --prompt "Explain norm-preserving subspace surgery." \
    --max-tokens 512
```

### OpenRouter

Tenet-26B-A4B-Gemma4-Unslop-v2 is served on OpenRouter by FP8-native providers with zero
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
    model="Tenet-26B-A4B-Gemma4-Unslop-v2",
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
   [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).
2. Apply the bi-projected norm-preserving subspace surgery using the
   `layer_<idx>` basis rows from `lancet_basis.safetensors` for each
   layer listed in `selected_layers`, restoring the residual norm
   exactly so activation magnitudes stay on-manifold.
3. Verify against `probes_sha256`: the probe manifest checksum must
   match the published recipe, making the reconstruction
   deterministic relative to the released BF16 surgery output.

- Steered layers (`selected_layers`): 15, 16, 17, 18, 19, 20, 21, 22
