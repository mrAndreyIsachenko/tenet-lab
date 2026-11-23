# Double-Championed Model Fight Card 16 (3B at 90 HP & 8B at 50 HP), Then Built Tenet v2 for Qwen3 `<think>` Models — Why 1 Gate Failed on 32B-v2 (and Why We Didn't Hide It)

**TL;DR:**
1. Last week on **Model Fight Card 16** ([`onlymodels.gg`](https://onlymodels.gg/cards/2026-11-20.html), 2026-11-20), our **Tenet v1** models swept both divisions: `Tenet-3B-Qwen2.5-Unslop-v1` won the **Minors Championship at `90 HP`** (`22/22` classifiable artifacts) and `Tenet-8B-Llama3.1-Unslop-v1` won the **Majors Championship at `50 HP`** (`24/24` classifiable artifacts, plus the night's only valid Peer Eval with **`7/7` clean hits**).
2. Moving from standard instruct models to **`<think>...</think>` reasoning models (`Qwen3 4B / 8B / 14B / 32B`)** exposed two new failure modes: naive 2-anchor abliteration fractures the `</think>` boundary (`IFEval Strict` on `Qwen3-8B` collapses to a third of base), while standard anti-slop GRPO bloats `<think>` padding or truncates closing tags at `max_tokens`.
3. **Tenet v2** fixes this with **H15 Tri-Anchor Lancet Surgery** ($U_{\text{reason}} \perp U_{\text{boundary}} \perp U_{\text{format}}$) + **H16 Length-Calibrated Anti-Padding GRPO**, verified with **Full vLLM OpenEvals** (`n=3,566` questions per arm, `think_end_token=</think>`, **`0.00%` unclosed `<think>`**):
   - **`Tenet-8B-Qwen3-Unslop-v2`:** `GSM8K Flex` **`90.90%`** (+0.61 pp vs Base `90.30%`), `MATH-500` **`72.20%`** (+1.60 pp vs Base, +1.80 pp vs Naive), `GPQA Diamond` **`40.40%`** (+2.02 pp vs Naive), `MMLU-Pro` **`59.72%`**, `IFEval` **`80.78%`** (`86.81%` inst-strict).
   - **`Tenet-32B-Qwen3-Unslop-v2`:** `GSM8K Flex` **`92.19%`** vs `84.76%` Naive (**`+7.43 pp`**), `MATH-500` **`74.80%`** (+3.60 pp vs Naive), `MMLU-Pro` **`65.48%`** (+2.58 pp vs Naive), `IFEval` **`84.47%`** (+2.21 pp vs Naive).
4. **Full transparency:** `4B-v2` and `8B-v2` pass **`10/10` hardware & GGUF gates** (`100.0 HP`, `24/24` artifacts, `100%` `<think>` integrity). `32B-v2` passes **`9/10` gates** — scoring `100.0 HP` (`24/24`) in `F16` and pure `CKA-Q4_K_M` (`4.952 bpw`), while failing **1 seed out of 24** (`97.92 HP`, `23/24`) only when `imatrix` is overlaid (`i1-CKA-Q4_K_M`). Below is the exact physics of why that happened and why we recommend pure `CKA-Q4_K_M` on 32B.

- **Tenet v2 HF Collection (`4B / 8B / 14B / 32B`):** [Tenet v2 — Thinking-Safe Tri-Anchor Unslop & Domain CKA-GGUF](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4)
- **Tenet v1 HF Collection (`3B / 8B / 12B / 32B`):** [Tenet v1 — Bi-Projected Unslop & FP8-Safe LLM Family](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v1-bi-projected-unslop-and-fp8-safe-llm-family-6afed2048ba13a74fb7e0fac)
- **Interactive FP8 & Abliteration Inspector Space:** [`mr-andrey-isachenko/fp8-abliteration-inspector`](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)
- **Open-Source Code & JSON Manifests:** [`github.com/mrAndreyIsachenko/tenet-lab`](https://github.com/mrAndreyIsachenko/tenet-lab)

---

## 1. What Model Fight Card 16 & `Qwen3` Exposed

In **Tenet v1**, dual-anchor orthogonal projection (`H9`) + per-row $L_2$-norm restoration (`H11`, `Row Crest-Factor < 25`, `FP8 E4M3 re-leakage < 3%`) proved out on independent arena testing ([`onlymodels.gg/cards/2026-11-20.html`](https://onlymodels.gg/cards/2026-11-20.html)):
- **`Tenet-3B-Qwen2.5-Unslop-v1`** → **Minors Champion (`90 HP`, `22/22` classifiable artifacts, `21/22` House escapes, `4/4` APEX seeds)**
- **`Tenet-8B-Llama3.1-Unslop-v1`** → **Majors Champion (`50 HP`, `24/24` classifiable artifacts, `20/24` House escapes, `4/4` APEX seeds, `7/7` Peer Eval hits)**

When we ported that exact v1 recipe to **`Qwen3` (`<think>...</think>`)**, two things broke immediately:
1. **Boundary Collapse:** Naive or 2-anchor refusal/slop subtraction intersects the latent direction governing the `</think>` transition and structured output formatting. On `Qwen3-8B`, naive abliteration crashed `IFEval Strict` to roughly a third of base and dragged `unslop_pass_rate` down with it.
2. **CoT Padding Under GRPO:** Penalizing post-answer slop without calibrating `<think>` length causes the policy to hide verbosity *inside* `<think>` or hit `max_tokens` before emitting `</think>`.

---

## 2. Tenet v2: Tri-Anchor Lancet (`H15`) + Length-Calibrated GRPO (`H16`)

1. **H15 Tri-Anchor Lancet Surgery (`PhaseSegmenter` + 2-Pass MGS):**
   Instead of orthogonalizing only against a single reasoning/tool basis, `H15` builds a `float32` Modified Gram-Schmidt orthonormal basis over **three** protected subspaces simultaneously:
   - $U_{\text{reason}}$: multi-step logic inside `<think>...</think>`
   - $U_{\text{boundary}}$: structural `</think>` transition tokens
   - $U_{\text{format}}$: strict JSON / read-only SQL / code compliance

   Target refusal/slop directions are extracted strictly from **post-`</think>`** tokens, ranked by CoT-weighted Signal-to-Collateral Ratio (SCR), and applied with exact per-row $L_2$-norm restoration.

2. **H16 Length-Calibrated Anti-Padding GRPO:**
   Applies an exponential length-excess penalty on bloated reasoning traces alongside a reward bonus for compact, complete `<think>...</think>` chains + clean post-tag answers.

### Full vLLM OpenEvals (`n=3,566` questions per arm, `think_end_token=</think>`, `0.00%` unclosed `<think>`)

| Model | IFEval | GSM8K | MATH-500 | GPQA Diamond | MMLU-Pro |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Tenet-4B-Qwen3-Unslop-v2`** (`10/10` gates, `100.0 HP`) | **`80.59%`** Strict (`n=541`, `+2.03 pp` vs Base `78.56%`) | **`88.93%`** Strict (`n=1319`, `+2.96 pp` vs Naive `85.97%`) | `70.20%` (`n=500`) | `34.34%` (`= Base`, `n=198`) | `56.35%` (`n=1008`, `+0.60 pp` vs Naive) |
| **`Tenet-8B-Qwen3-Unslop-v2`** (`10/10` gates, `100.0 HP`) | `80.78%` (`86.81%` inst-strict) | **`90.90%`** Flex (`+0.61 pp` vs Base `90.30%`) | **`72.20%`** (`+1.60 pp` vs Base `70.60%`, `+1.80 pp` vs Naive `70.40%`) | **`40.40%`** (`+0.50 pp` vs Base `39.90%`, `+2.02 pp` vs Naive `38.38%`) | `59.72%` (`+0.30 pp` vs Base) |
| **`Tenet-32B-Qwen3-Unslop-v2`** (`9/10` gates) | **`84.47%`** (`+2.21 pp` vs Naive `82.26%`) | **`92.19%`** Flex vs `84.76%` Naive (`+7.43 pp` vs Naive, `+0.45 pp` vs Base `91.74%`) | **`74.80%`** (`+3.60 pp` vs Naive `71.20%`) | **`40.40%`** (`+1.51 pp` vs Base `38.89%`, `+2.52 pp` vs Naive `37.88%`) | **`65.48%`** (`+2.58 pp` vs Naive `62.90%`) |

---

## 3. Why 1 Gate Failed on `32B-v2` (and Why We Didn't Hide It)

Every Tenet v2 release is run through **10 automated hardware & GGUF release gates** (`H11` FP8 Crest `< 25.0` & Re-Leakage `< 3.0%`, plus `H14+` / `H17` 24-seed Arena House Verifier across `F16`, pure `CKA-Q4_K_M`, and `i1-CKA-Q4_K_M`, plus `6/6` read-only SQL checks):

| Release | Max Row Crest (`< 25`) | FP8 Re-Leakage (`< 3%`) | `F16` Arena HP | `CKA-Q4_K_M` HP | `i1-CKA-Q4_K_M` HP | Avg BPW | Gates Passed | Recommended Format |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`4B-Qwen3-Unslop-v2`** | `3.1766` | `2.4946%` | **`100.0`** (`24/24`) | **`100.0`** (`24/24`) | **`100.0`** (`24/24`) | — | **`10/10`** | `i1-CKA-Q4_K_M` / `BF16` |
| **`8B-Qwen3-Unslop-v2`** | `4.3688` | `2.7243%` | **`100.0`** (`24/24`) | **`100.0`** (`24/24`) | **`100.0`** (`24/24`) | `5.074` | **`10/10`** | `i1-CKA-Q4_K_M` / `BF16` |
| **`14B-Qwen3-Unslop-v2`** | `3.5974` | `2.5898%` | `93.75` (`22/24`) | `91.67` (`21/24`, `6/6` SQL) | `95.83` (`23/24`, `0/6` SQL) | `5.014` | `5/10` | `BF16` / `FP8` |
| **`32B-Qwen3-Unslop-v2`** | `4.3593` | `2.5579%` | **`100.0`** (`24/24`) | **`100.0`** (`24/24`, `6/6` SQL) | `97.92` (`23/24`) | `4.952` | **`9/10`** | **`CKA-Q4_K_M`** / `BF16` |

### What happened on `32B-v2` `i1-CKA-Q4_K_M`?
In `BF16`, `FP8` (`Crest=4.3593`, `re-leakage=2.5579%`), `F16 GGUF`, and **pure `CKA-Q4_K_M` (`4.952 bpw`)**, `Tenet-32B-Qwen3-Unslop-v2` scores **`100.0 HP` (`24/24` artifacts, `100%` `<think>` integrity, `100%` unslop, `6/6` SQL)**.

When we overlaid `tenet_domain.imatrix` (`i1-CKA-Q4_K_M`), **1 out of 24 seeds** (`house-edge-leap-years`) slipped (`97.92 HP`, `23/24`), failing the strict `house_artifact_classification_rate == 1.0` gate.

**The physics:** `imatrix` minimizes average calibration perplexity by scaling quantization error according to $\mathbb{E}[x_j^2]$ across the calibration corpus. On `4B` and `8B`, that helps (`8B` `i1-CKA-Q4_K_M` cuts Mean KLD vs `F16` by `66.1%` to `0.01314`). On `32B`, however, the model already has wide, well-conditioned singular spectra, and `imatrix` re-weighting slightly rotates the surgically zeroed subspace onto the discrete 4-bit `Q4_K_M` grid on edge-case boundary tokens. Meanwhile, on `14B`, 4-bit `imatrix` quantization perturbs the `sql_readonly` boundary (`SHOW TABLES;` vs `SELECT`), dropping `sql_readonly_rate` from `6/6` (`CKA-Q4_K_M`) to `0/6` (`i1-CKA-Q4_K_M`).

So instead of cherry-picking or dropping the failing gate, we published the exact `foundry_summary.json` manifests and ship **`CKA-Q4_K_M`** (without `imatrix`) as the recommended 4-bit deployment artifact for `32B-v2`.

**Coming next (`H18` NVFP4 Quant-Snap preview):** the same domain-CKA pipeline rebuilt on NVFP4 Quant-Snap cuts FP8 re-leakage by **`-26.2%`** in early snapshots — we'll publish the full matrix once the 24-seed arena verifier completes.

---

## Try It & Inspect the Weights

- **Run `8B-v2` or `32B-v2` locally** with `llama.cpp` / Ollama using the GGUFs in the [Tenet v2 Collection](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4).
- **Inspect any HF model's FP8 Row Crest-Factor & CKA layers** in our Space: [`mr-andrey-isachenko/fp8-abliteration-inspector`](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector).
- **Reproduce every number** from [`mrAndreyIsachenko/tenet-lab`](https://github.com/mrAndreyIsachenko/tenet-lab).
