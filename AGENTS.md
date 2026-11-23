# Small Model Lab — Agent Operating Rules & Cloud GPU Pod Protocol

These rules are **mandatory** for every agent session operating in `/Users/mr_a_i/Developer/personal/Small Model Lab`. They exist to prevent lost GPU compute hours, lost artifacts on pod termination, unverified multi-hour blind runs, and LLM token burn from polling loops.

---

## 1. Unified Entry Point — agent-harness

All delegation and pod operations go through `agent-harness` (`/Users/mr_a_i/Developer/personal/Harness`):

- **`harness run` — delegating code edits:** `openrouter/z-ai/glm-5.3-flash`, max 1–3 files per call, exact absolute file paths and symbol names, no `find`/`glob`, only the listed files. Always review `git diff` and run `pytest` in the main session after the worker exits.
- **`harness pod` — all cloud GPU pod operations** (training, Tenet Foundry, OpenEvals, H18/H19 benchmarks), exposed via Makefile targets:
  - `make pod-preflight POD_HOST=<host>` → `harness pod preflight --host ... --workdir ... --min-free-vram-mib 6144`
  - `make pod-sync-daemon POD_HOST=<host>` → `harness pod sync ... --daemon --interval 60`
  - `make pod-sync-stop POD_HOST=<host>` → `harness pod sync ... --stop`
  - `make pod-launch POD_HOST=<host> POD_CMD=<command>` → `harness pod launch --stagger-vram-mib 6144 [--on-checkpoint ...] -- <command>`
  - `make pod-wait POD_HOST=<host>` → `harness pod wait --max-wait-sec 180`

## 2. The 5 Iron Laws of Cloud GPU Pods (Training, Weight Surgery, OpenEvals, H18/H19)

### Law 1 — Pre-Flight & VRAM / ETA Budget Before Any Run
Before launching any GPU workload on a remote pod, you **MUST** run `make pod-preflight POD_HOST=<host> POD_PORT=<port>` and present a concise budget table:
1. **Hardware Passport:** GPU model, count, total VRAM per card, compute capability (`sm_90`, `sm_100`, `sm_103`), and free disk space on `/workspace`.
2. **VRAM Allocation & KV-Cache Budget:**
   - Every `vLLM` worker requires large pure KV-cache headroom (`>= 40 GiB` for `4B/8B`, `>= 100 GiB` for `32B`) for high-throughput continuous batching (`15–50 it/s`):
     $$\text{VRAM}_{\text{worker}} = \text{Weights}_{\text{bf16}} + \text{CUDA Graphs / Activations} + \text{KV Cache}$$
   - On a single `B300 275 GB`, **never** run `4B + 8B + 32B` simultaneously (`32B` at `0.45` leaves only `3.6 GiB` KV cache and degrades 55x). Instead:
     1. Run `4B (0.42)` + `8B (0.48)` in parallel (`~100–110 GiB` pure KV cache each → ~10 min total).
     2. Then run `32B (0.90)` **exclusively** (`~130 GiB` pure KV cache → ~15 min for all 3 arms).
   - **NEVER** initialize multiple `vLLM EngineCore` instances at the exact same second on the same GPU. Use **staggered startup** (`harness pod launch --stagger-vram-mib 6144`).
3. **ETA Table & Fail-Fast Pipeline Order:**
   - Always run fast steps (`h18_audit`, `h18_h19_bench`, `export_model_index.py`, `summarize_h100_v3_suite.py`) **first** before long `vLLM` runs (`STEPS="h18_audit,h18_h19_bench,eval_v2,tenet_v3"`).
   - Remember: in `lm-eval`, `mmlu_pro` is a group of **14 subtasks**, so `--limit N` applies **per subtask** (`14 × N` total questions). Always pass `--mmlu-pro-limit 72` (`14 × 72 = 1008`).

### Law 2 — Mandatory 60-Second Smoke Gate (`--limit 2`), Chat-0Shot Split & Sanity Gate
After any environment setup, package install, or script modification on a pod:
- **NEVER** launch a `--suite full` run before passing a 60-second smoke check (`--limit 2 --mmlu-pro-limit 2`) and verifying:
  1. `UV_NO_SYNC=1 SKIP_UV_SYNC=1` is exported so `uv run` does not uninstall `vllm` / `evalplus`;
  2. `.env` (`HF_TOKEN`) is loaded and gated datasets (`gpqa_diamond_zeroshot`) authenticate cleanly;
  3. `0-shot` instruction tasks (`ifeval`, `hendrycks_math500`) run with `--split-chat-0shot` (`--apply_chat_template`, `enable_thinking=False`, `max_gen_toks=1280`) and `--sanity-gate` so `vLLM` stops immediately on task stop sequences / `<|im_end|>` instead of looping to 8192 tokens, and aborts immediately if `math_500 <= 5%` or `ifeval_strict <= 10%`.

### Law 3 — Continuous Pull, Immediate Publish & Auto-Stop Pod (`runpodctl stop pod`)
- **At the exact moment a pod job starts**, start the local sync daemon on the Mac (`make pod-sync-daemon POD_HOST=<host>` → `harness pod sync --daemon --interval 60`).
- The daemon runs purely in local bash (**0 LLM tokens**) and pulls `results/external_evals/`, `results/tenet_releases/`, `results/h18_nvfp4_quant_snap/`, and `results/h19_loop_unrolled_lancet/` every 60 seconds (`POD_PATHS=results`).
- **Never wait for the end of a multi-model batch to save or publish results:** as soon as any single model (`4B`, `8B`, `32B`) finishes, immediately run `export_model_index.py`, sync the canonical JSON to Mac, and push updated artifacts/model cards to Hugging Face (`hf upload`).
- **Auto-Stop Pod When Finished:** As soon as all artifacts are synced to Mac and pushed to Hugging Face, stop the sync daemon (`make pod-sync-stop POD_HOST=<host>`) and immediately pause/stop the pod (`--auto-stop-pod` / `runpodctl stop pod "${RUNPOD_POD_ID}"`) so zero idle GPU billing occurs.

### Law 4 — Granular `per-arm` & `per-subtask` Resume + Automatic Retry
- Always run `scripts/eval/run_vllm_open_evals.py` with `--resume` (and `--max-retries 2`).
- Completed arms (`raw/<arm>/arm_summary.json` with `status == "ok"` and non-empty `scores`, JSON-validated) and completed subtasks (`raw/<arm>` main suite JSON or `raw/<arm>/mmlu_pro` JSON) must **never** be recomputed from scratch on restart.
- Retry workers start with `start_new_session=True` and are terminated via `os.killpg(..., signal.SIGKILL)` so no orphaned GPU processes linger.
- Always stream live `lm-eval` progress to the worker log file so `tail -n 20 /workspace/b300_*.log` shows real-time `tqdm` prompt counts and `tok/s`.

### Law 5 — Zero-Token Supervision (`No schedule / manage_task Loops`)
- **STRICTLY FORBIDDEN:** Using `schedule` (`*/15 * * * *`) or repeated `manage_task` turns in the main LLM session to poll `nvidia-smi` or `tail` logs. Every wakeup re-bills 150K–250K context tokens.
- Instead, wait only via `make pod-wait POD_HOST=<host>` (`harness pod wait`) inside a single blocking call at **0 LLM token cost**, max 180 seconds (`POD_MAX_WAIT_SEC`); it returns only when a checkpoint completes, a worker errors, or the timeout elapses.
- If `pod-wait` reports `STATUS: RUNNING`, **end your turn and yield control back to the user** — do not chain a second wait in the same turn.

---

## 3. Coding, Delegation & Context Hygiene Rules

1. **Delegate mechanical multi-line code edits to `harness run` (`openrouter/z-ai/glm-5.3-flash`):**
   - Max 1–3 files per call; pass exact absolute paths, function names, and `/Users/mr_a_i/Developer/personal/Small Model Lab/.venv/bin/pytest`; instruct the worker to read/edit only the listed files (no `find`/`glob`).
   - Always review `git diff` and run `uv run ruff check`, `uv run ruff format --check`, and `uv run pytest -q` in the main session after edits.
2. **Isolate multi-file codebase exploration in `research` subagents (`Model: "flash"`):**
   - Never perform >3 exploratory `read_file` calls in the main session.
   - Always pass bounded `StartLine` and `EndLine` windows on files over 150 lines.
