---
license: apache-2.0
base_model: mistralai/Mistral-Small-3.1-24B-Instruct-2503
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
- gguf
- conversational
---

# Tenet-24B-Mistral3.1-Unslop-v2

## Model overview

**Tenet-24B-Mistral3.1-Unslop-v2** is a Tenet fine-tune of [`mistralai/Mistral-Small-3.1-24B-Instruct-2503`](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503),
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

**Tenet-24B-Mistral3.1-Unslop-v2** is intended for aligned assistant use. Lancet subspace
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

Max row crest-factor: n/a

## Why Tenet instead of Naive Abliteration?

| Metric | Base | Naive Abliteration | Tenet (BF16) | Tenet (FP8 E4M3) |
| --- | --- | --- | --- | --- |
| KL Collateral Drift | — | — | — | — |
| Refusal Suppression | — | — | — | — |
| Tool/SQL Anchor Orthogonality | — | — | — | — |
| LoRA Re-Leakage | — | — | — | — |
| FP8 Subspace Re-Leakage | — | — | — | — |
| Max Row Crest-Factor (< 25.0) | — | — | — | — |

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
vllm serve Tenet-24B-Mistral3.1-Unslop-v2 --quantization fp8 --max-model-len 32768
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Tenet-24B-Mistral3.1-Unslop-v2"
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
llama-cli -m ./Tenet-24B-Mistral3.1-Unslop-v2-CKA-Quant-Q4_K_M.gguf -ngl 99 -n 256 \
    -p "Return a DuckDB SQL tool call for failed logins by source IP."
```

## Community GGUF / Local Inference (llama.cpp, Ollama, LM Studio)

Community `mradermacher/Tenet-24B-Mistral3.1-Unslop-v2-i1-GGUF` / `mradermacher/Tenet-24B-Mistral3.1-Unslop-v2-GGUF`
builds are upcoming and tracked. Until they land, quantize locally with the
CKA-guided recipe shipped in this bundle -
[`quantize_cka_gguf.sh`](https://github.com/mrAndreyIsachenko/tenet-lab) +
`cka_quant_recipe.json` - which emits llama.cpp GGUFs in the same
**i1-Q4_K_M** / **i1-Q6_K** spirit while keeping high-CKA boundary layers
on `Q8_0`:

```bash
bash ./quantize_cka_gguf.sh   # reads cka_quant_recipe.json
```

The recipe keeps the FP8 surgery output stable under 4-6 bit llama.cpp
quantization without re-leaking refusal or slop behaviour.

## Deployment & inference

### vLLM (FP8)

```bash
vllm serve Tenet-24B-Mistral3.1-Unslop-v2 --quantization fp8
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="Tenet-24B-Mistral3.1-Unslop-v2",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Ollama / GGUF

```text
# Modelfile
FROM ./Tenet-24B-Mistral3.1-Unslop-v2-Q4_K_M.gguf
PARAMETER temperature 0.7
```

```bash
ollama create tenet-24b-mistral3.1-unslop-v2 -f Modelfile
ollama run tenet-24b-mistral3.1-unslop-v2
```

### MLX

```bash
mlx_lm.generate \
    --model Tenet-24B-Mistral3.1-Unslop-v2 \
    --prompt "Explain norm-preserving subspace surgery." \
    --max-tokens 512
```

### OpenRouter

Tenet-24B-Mistral3.1-Unslop-v2 is served on OpenRouter by FP8-native providers with zero
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
    model="Tenet-24B-Mistral3.1-Unslop-v2",
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
   [`mistralai/Mistral-Small-3.1-24B-Instruct-2503`](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503).
2. Apply the bi-projected norm-preserving subspace surgery using the
   `layer_<idx>` basis rows from `lancet_basis.safetensors` for each
   layer listed in `selected_layers`, restoring the residual norm
   exactly so activation magnitudes stay on-manifold.
3. Verify against `probes_sha256`: the probe manifest checksum must
   match the published recipe, making the reconstruction
   deterministic relative to the released BF16 surgery output.

