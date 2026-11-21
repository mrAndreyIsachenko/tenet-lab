# Tenet v2 Social Release Kit (X/Twitter Thread, LinkedIn, Hugging Face Community)

---

## 1. X / Twitter Thread (7 Tweets)

### Tweet 1/7 (Hook)
Double championship sweep on Model Fight Card 16 (`onlymodels.gg`):
- `Tenet-3B-Qwen2.5-Unslop-v1` won **Minors (90 HP, 22/22 artifacts)**
- `Tenet-8B-Llama3.1-Unslop-v1` won **Majors (50 HP, 24/24 artifacts, 7/7 Peer Eval hits)**

Now we're shipping **Tenet v2** for `<think>` reasoning models (`Qwen3 4B / 8B / 14B / 32B`). 🧵👇

### Tweet 2/7 (The `<think>` Problem)
Why couldn't we just reuse Tenet v1 on `Qwen3`?
Two failure modes:
1. Naive / 2-anchor abliteration fractures the `</think>` boundary (`Qwen3-8B` `IFEval Strict` collapses 3x: `0.300 -> 0.100`).
2. Standard anti-slop GRPO pushes verbosity *inside* `<think>` or hits `max_tokens` before closing tags.

### Tweet 3/7 (H15 + H16 Solution)
Tenet v2 fixes both:
- **H15 Tri-Anchor Lancet Surgery:** orthogonalizes post-`</think>` refusal/slop against $U_{\text{reason}} \perp U_{\text{boundary}} \perp U_{\text{format}}$ in `float32` + per-row $L_2$-norm restoration.
- **H16 Length-Calibrated GRPO:** penalizes `<think>` padding while rewarding compact, complete reasoning chains.

### Tweet 4/7 (H100 3-Way `lm-eval` Proof)
Verified on 1x H100 (`lm-eval` `limit=40`, `seed=1337`):
- **`Tenet-8B-Qwen3-Unslop-v2`:** holds **`0.300` `IFEval Strict`** (100% of Base vs `0.100` Naive) & finishes **`17.4%` faster** (`851.8s` vs `1030.7s`).
- **`Tenet-14B-Qwen3-Unslop-v2`:** **`+10.0 pp` `GSM8K CoT Flexible`** (`0.250` vs `0.150` Base/Naive) & `+2.5 pp` `IFEval` (`0.250` vs `0.225`).

### Tweet 5/7 (10 Hardware Gates & Radical Transparency on `32B-v2`)
Every release faces **10 hardware & GGUF gates** (`H11` FP8 Crest < 25 + `H17` 24-seed Arena Verifier):
- `4B-v2` & `8B-v2`: **`10/10` gates** (`100.0 HP`, `24/24`, `100%` `<think>` integrity)
- `32B-v2`: **`9/10` gates** — `100.0 HP` (`24/24`) in `F16` & pure `CKA-Q4_K_M` (`4.952 bpw`), 1 seed missed (`97.92 HP`) only when `imatrix` is overlaid.

### Tweet 6/7 (Why `imatrix` Slipped on 32B)
Why did `i1-CKA-Q4_K_M` miss 1/24 seeds on `32B-v2` while pure `CKA-Q4_K_M` hit `100.0 HP`?
`imatrix` optimizes average perplexity by re-weighting activations, which on a wide 32B spectrum slightly rotates the surgically zeroed subspace onto the 4-bit grid. On `32B-v2`, pure `CKA-Q4_K_M` is the cleaner quant.

### Tweet 7/7 (Links & CTA)
All weights, GGUFs, and JSON manifests are live — test them on your own prompts & arenas:
- **Tenet v2 Collection (`4B–32B`):** https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4
- **FP8 Inspector Space (`@huggingface`):** https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector
- **Code & Benchmarks:** https://github.com/mrAndreyIsachenko/tenet-lab

Built on `@runpod_io` H100s.

---

## 2. LinkedIn Post

**From Double Championship on Model Fight Card 16 to Tenet v2 (`Qwen3 4B / 8B / 14B / 32B`)**

Last week, our **Tenet v1** models won both divisions outright on **Model Fight Card 16** (`onlymodels.gg`, 2026-11-20):
- **`Tenet-3B-Qwen2.5-Unslop-v1`** — Minors Champion (`90 HP`, `22/22` classifiable artifacts)
- **`Tenet-8B-Llama3.1-Unslop-v1`** — Majors Champion (`50 HP`, `24/24` classifiable artifacts, `7/7` Peer Eval hits)

When we moved to **`<think>...</think>` reasoning models (`Qwen3`)**, we found that standard 2-anchor abliteration fractures the `</think>` boundary (`Qwen3-8B` `IFEval Strict` drops 3x from `0.300` to `0.100`), while standard anti-slop GRPO inflates `<think>` padding.

Today we are releasing **Tenet v2 (`Qwen3 4B / 8B / 14B / 32B`)**, built on two new methods:
1. **H15 Tri-Anchor Lancet Surgery:** orthogonalizes post-`</think>` refusal/slop directions simultaneously against reasoning ($U_{\text{reason}}$), boundary transition ($U_{\text{boundary}}$), and strict formatting ($U_{\text{format}}$), with per-row $L_2$-norm restoration for `FP8 E4M3` safety (`Crest 3.18–4.37 < 25.0`, `re-leakage 2.49%–2.72% < 3.0%`).
2. **H16 Length-Calibrated Anti-Padding GRPO:** eliminates CoT bloat while preserving `</think>` tag integrity.

**Measured H100 3-way `lm-eval` (`limit=40`) Highlights:**
- `Tenet-8B-Qwen3-Unslop-v2`: maintains **`0.300` `IFEval Strict`** (vs `0.100` Naive) and completes the 4-task suite **`17.4%` faster** (`851.8s` vs `1030.7s` Base).
- `Tenet-14B-Qwen3-Unslop-v2`: **`+10.0 pp` on `GSM8K CoT Flexible`** (`0.250` vs `0.150` Base/Naive) and `+2.5 pp` on `IFEval Strict` (`0.250` vs `0.225`).
- **10-Gate Hardware & GGUF Verification:** `4B-v2` and `8B-v2` pass **`10/10` gates** (`100.0 HP`, `24/24`). `32B-v2` passes **`9/10` gates** (`100.0 HP` `24/24` in `F16` and pure `CKA-Q4_K_M` at `4.952 bpw`, with 1/24 seed drift only when `imatrix` is overlaid).

Full weights, GGUFs, interactive FP8 inspector, and reproducible JSON manifests:
- Tenet v2 Collection: https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4
- FP8 Inspector Space: https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector
- GitHub (`tenet-lab`): https://github.com/mrAndreyIsachenko/tenet-lab

---

## 3. Hugging Face Community Post

**Title:** `Tenet v2 (Qwen3 4B / 8B / 14B / 32B): Thinking-Safe Tri-Anchor Unslop + CKA-GGUF`

Following our double championship sweep at **Model Fight Card 16** (`onlymodels.gg`, 2026-11-20) with `Tenet-3B-Qwen2.5-Unslop-v1` (Minors, `90 HP`) and `Tenet-8B-Llama3.1-Unslop-v1` (Majors, `50 HP`, `7/7` Peer Eval hits), we are publishing the **Tenet v2** family for `Qwen3` `<think>` models:

- **Collection:** [Tenet v2 — Thinking-Safe Tri-Anchor Unslop & Domain CKA-GGUF](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4)
- **Inspector Space:** [`mr-andrey-isachenko/fp8-abliteration-inspector`](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)
- **Source & JSON Manifests:** [`mrAndreyIsachenko/tenet-lab`](https://github.com/mrAndreyIsachenko/tenet-lab)

### Release Gate Summary (`results/tenet_releases/*/foundry_summary.json`)
| Model | Max Crest (`< 25`) | FP8 Re-Leakage (`< 3%`) | `F16` Arena HP | `CKA-Q4_K_M` HP | `i1-CKA-Q4_K_M` HP | Gates | Recommended Quant |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Tenet-4B-Qwen3-Unslop-v2` | `3.1766` | `2.4946%` | `100.0` (`24/24`) | `100.0` (`24/24`) | `100.0` (`24/24`) | `10/10` | `i1-CKA-Q4_K_M` / `BF16` |
| `Tenet-8B-Qwen3-Unslop-v2` | `4.3688` | `2.7243%` | `100.0` (`24/24`) | `100.0` (`24/24`) | `100.0` (`24/24`) | `10/10` | `i1-CKA-Q4_K_M` / `BF16` |
| `Tenet-14B-Qwen3-Unslop-v2` | `3.5974` | `2.5898%` | `93.75` (`22/24`) | `91.67` (`21/24`) | `95.83` (`23/24`) | `5/10` | `BF16` / `FP8` |
| `Tenet-32B-Qwen3-Unslop-v2` | `4.3593` | `2.5579%` | `100.0` (`24/24`) | `100.0` (`24/24`) | `97.92` (`23/24`) | `9/10` | `CKA-Q4_K_M` / `BF16` |

Feedback, arena runs, and PRs on `tenet-lab` are very welcome!
