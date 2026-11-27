---
trigger: always_on
description: Mandatory Cloud GPU Pod Protocol for Small Model Lab via agent-harness (VRAM/ETA pre-flight, 60s smoke gate, 60s sync daemon with immediate publish, per-arm/per-subtask resume, and zero LLM-token supervision).
---

# Small Model Lab — Cloud GPU Pod Protocol

Always follow `/Users/mr_a_i/Developer/personal/Small Model Lab/AGENTS.md`. All pod operations go through the unified `agent-harness` entry point (`/Users/mr_a_i/Developer/personal/Harness`), exposed via Makefile targets:

- `make pod-preflight POD_HOST=<host>` → `harness pod preflight`
- `make pod-sync-daemon POD_HOST=<host>` → `harness pod sync --daemon`
- `make pod-sync-stop POD_HOST=<host>` → `harness pod sync --stop`
- `make pod-launch POD_HOST=<host> POD_CMD=<command>` → `harness pod launch`
- `make pod-wait POD_HOST=<host>` → `harness pod wait`

## 5 Iron Laws

1. **Pre-Flight, KV-Cache Headroom & Fail-Fast Order:** Run `make pod-preflight` before any pod run and present a budget table. Never start multiple `vLLM EngineCore` workers simultaneously — `harness pod launch --stagger-vram-mib` staggers startups. On a 275 GB B300, run `4B (0.42)` + `8B (0.48)` in parallel (`~100+ GiB` pure KV cache each), and then `32B (0.90)` **exclusively** (`~130 GiB` pure KV cache) — never run `32B` at `0.45` alongside `4B`+`8B`. Run fast steps (`h18_audit`, `h18_h19_bench`, exporters) **first** before long `vLLM` runs. Always pass `--mmlu-pro-limit 72` (1008 questions across 14 subtasks), never `1000`.
2. **60-Second Smoke Gate, Chat-0Shot Split & Sanity Gate:** After any environment/code change, verify with `--limit 2 --mmlu-pro-limit 2` and `UV_NO_SYNC=1 SKIP_UV_SYNC=1`. Always run `0-shot` instruction tasks (`ifeval`, `hendrycks_math500`) with `--split-chat-0shot` (`--apply_chat_template`, `enable_thinking=False`, `max_gen_toks=1280`) and `--sanity-gate` so any `0.0%` metric or runaway token-wall loop aborts immediately on the first subtask.
3. **Continuous Pull, Immediate Publish & Auto-Stop Pod:** Start `make pod-sync-daemon` (60s local rsync daemon, 0 LLM tokens) at pod launch. After each individual model completes, immediately run `export_model_index.py`, sync to Mac, and `hf upload` — never wait for the whole batch. As soon as all models are synced and published, stop the sync daemon and pause/stop the pod (`--auto-stop-pod` / `runpodctl stop pod "${RUNPOD_POD_ID}"`) to halt GPU billing immediately.
4. **Granular Resume & Auto-Retry:** Always run `run_vllm_open_evals.py` with `--resume` (per-arm `arm_summary.json` and per-subtask `raw/<arm>` JSON reuse with validation) and `--max-retries 2` (process groups started with `start_new_session=True` and killed via `os.killpg(..., signal.SIGKILL)`). Live `lm-eval` progress streams to log files.
5. **Zero-Token Autonomous Supervision (No Polling, No "Continue?" Prompts & No Killing SSH Inside `{worker}`):** Strictly no `schedule`, no repeated `manage_task` polling, no 180s timeouts, and **never** call `nebius_ctl.sh stop` / `shutdown` inside `{worker}`'s `EXIT` trap (which kills SSH before `DONE` is written and synced). Instead, `{worker}` runs `nebius_ctl.sh s3-sync-up` and exits cleanly (`tenet-idle-watchdog.service` remains on the VM as a 10-minute idle fallback), while the Mac runs **`make nebius-wait-and-stop POD_MAX_WAIT_SEC=36000`** (`WaitMsBeforeAsync: 5000`, **omit `NotificationTimeoutSeconds`**) — which waits for `DONE`/`ERROR`, runs the final `rsync`, stops `pod-sync-daemon`, calls `nebius_ctl.sh stop`, and wakes the LLM session immediately.

## Context Hygiene

Isolate multi-file codebase exploration in a `research` subagent (`Model: "flash"`). Use bounded `read_file` with `StartLine`/`EndLine` on files over 150 lines; never perform more than 3 exploratory reads in the main session.
