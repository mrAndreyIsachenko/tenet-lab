#!/usr/bin/env python3
"""2026-standard 3-way OpenEvals runner: Base vs Naive Abliteration vs Tenet.

Evaluates three model references with ``lm-eval --model vllm`` (or
``local-chat-completions`` for GGUF served behind llama-server / ``vllm serve``):

1. Base model
2. Naive Abliteration control (pre-built abliterated reference)
3. Tenet release

Suite profiles (``--suite full | quick``; ``full`` has NO default limit):
- ``full``:  ifeval, hendrycks_math_500, gsm8k_cot, gpqa_diamond_zeroshot,
  mmlu_pro + evalplus (humaneval_plus, mbpp_plus) + eqbench
  (eqbench_creative_writing, slop_score)
- ``quick``: ifeval, gsm8k_cot at limit=40

Reasoning-model handling (Qwen3 etc.): responses are parsed for an unclosed
``<think>`` block and the per-arm ``unclosed_think_rate`` is reported.

Outputs:
- raw lm-eval logs: ``results/external_evals/<release_id>/raw/<arm>/``
- canonical JSON:   ``results/external_evals/<release_id>.json``
- copy:             ``results/tenet_releases/<release_id>/external_benchmarks.json``
- markdown report:  ``results/external_evals/<release_id>/report.md``

This script does NOT perform surgery or load weights itself; it only drives
lm-eval (and optional evalplus/eqbench entrypoints when installed). Use
``--dry-run`` to print the commands without executing anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_THINK_START_TOKEN = "<think>"
DEFAULT_THINK_END_TOKEN = "</think>"
DEFAULT_MAX_GEN_TOKS = 8192
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_API_BASE = "http://localhost:8080"

ARMS = ("base", "naive", "tenet")

SUITE_PROFILES: dict[str, dict[str, Any]] = {
    "full": {
        "limit": None,
        "tasks": (
            "ifeval",
            "hendrycks_math_500",
            "gsm8k_cot",
            "gpqa_diamond_zeroshot",
            "mmlu_pro",
        ),
        "evalplus_tasks": ("humaneval_plus", "mbpp_plus"),
        "eqbench_tasks": ("eqbench_creative_writing", "slop_score"),
    },
    "quick": {
        "limit": 40,
        "tasks": ("ifeval", "gsm8k_cot"),
        "evalplus_tasks": (),
        "eqbench_tasks": (),
    },
}


# ---------------------------------------------------------------------------
# Pure helpers: reasoning-response parsing
# ---------------------------------------------------------------------------


def parse_thinking_response(
    text: str,
    *,
    think_start_token: str = DEFAULT_THINK_START_TOKEN,
    think_end_token: str = DEFAULT_THINK_END_TOKEN,
) -> tuple[str, bool]:
    """Split a reasoning-model response into (answer, is_unclosed).

    - If ``think_end_token`` is present: split on the *last* occurrence and
      return ``(after.strip(), False)``.
    - Else if ``think_start_token`` is present: the think block never closed,
      return ``("", True)``.
    - Otherwise return ``(text.strip(), False)``.
    """
    if think_end_token and think_end_token in text:
        answer = text.rsplit(think_end_token, 1)[1]
        return answer.strip(), False
    if think_start_token and think_start_token in text:
        return "", True
    return text.strip(), False


def compute_unclosed_think_rate(
    responses: Sequence[str],
    *,
    think_start_token: str = DEFAULT_THINK_START_TOKEN,
    think_end_token: str = DEFAULT_THINK_END_TOKEN,
) -> float:
    """Fraction of responses whose ``<think>`` block never closed (0.0 if empty)."""
    items = list(responses)
    if not items:
        return 0.0
    unclosed = sum(
        1
        for r in items
        if parse_thinking_response(
            r, think_start_token=think_start_token, think_end_token=think_end_token
        )[1]
    )
    return unclosed / len(items)


def _iter_sample_texts(sample: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("resps", "filtered_resps"):
        value = sample.get(key)
        if value is None:
            continue
        for entry in value:
            if isinstance(entry, str):
                texts.append(entry)
            elif isinstance(entry, (list, tuple)):
                texts.extend(e for e in entry if isinstance(e, str))
    return texts


def extract_lm_eval_samples_unclosed_think_rate(
    raw_samples: dict[str, Any] | list[dict[str, Any]],
    *,
    think_start_token: str = DEFAULT_THINK_START_TOKEN,
    think_end_token: str = DEFAULT_THINK_END_TOKEN,
) -> float:
    """Unclosed-think rate over lm-eval ``log_samples`` payloads.

    Accepts a list of sample dicts, a single sample dict, or a results dict
    with a ``samples`` key; reads text from ``resps`` / ``filtered_resps``.
    """
    samples: list[dict[str, Any]] = []
    if isinstance(raw_samples, dict):
        if "resps" in raw_samples or "filtered_resps" in raw_samples:
            samples.append(raw_samples)
        elif isinstance(raw_samples.get("samples"), list):
            samples = [s for s in raw_samples["samples"] if isinstance(s, dict)]
    elif isinstance(raw_samples, list):
        samples = [s for s in raw_samples if isinstance(s, dict)]
    texts: list[str] = []
    for sample in samples:
        texts.extend(_iter_sample_texts(sample))
    return compute_unclosed_think_rate(
        texts, think_start_token=think_start_token, think_end_token=think_end_token
    )


# ---------------------------------------------------------------------------
# Pure helpers: command building
# ---------------------------------------------------------------------------


def _shell_safe(value: Any) -> str:
    """Render a model_args value as an lm-eval-safe string."""
    text = str(value)
    for ch in (",", "=", "\n"):
        text = text.replace(ch, " ")
    return text.strip()


def build_lm_eval_command(
    model_ref: str,
    *,
    tasks: Sequence[str],
    output_path: Path,
    backend: str = "vllm",
    precision: str = "bf16",
    limit: int | None = None,
    batch_size: int | str = "auto",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    think_end_token: str = DEFAULT_THINK_END_TOKEN,
    api_base: str | None = None,
) -> list[str]:
    """Build the ``lm_eval`` argv for one arm.

    ``backend='vllm'`` runs the in-process vLLM backend (precision bf16/fp8);
    ``backend='openai-compatible'`` runs ``local-chat-completions`` against an
    OpenAI-compatible server (llama-server / ``vllm serve``), which is how GGUF
    precision is evaluated. ``precision='gguf'`` callers should pass
    ``backend='openai-compatible'``.
    """
    cmd = ["lm_eval", "--tasks", ",".join(tasks)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    cmd += ["--batch_size", str(batch_size)]
    cmd += ["--output_path", str(output_path)]
    cmd += ["--log_samples", "--gen_kwargs", f"max_gen_toks={int(max_gen_toks)}"]

    if backend == "vllm":
        dtype = "bfloat16" if precision == "bf16" else precision
        model_args = [
            f"pretrained={_shell_safe(model_ref)}",
            f"dtype={dtype}",
            f"tensor_parallel_size={int(tensor_parallel_size)}",
            f"gpu_memory_utilization={float(gpu_memory_utilization)}",
            f"max_model_len={int(max_model_len)}",
        ]
        if think_end_token:
            model_args.append(f"think_end_token={think_end_token}")
        cmd += ["--model", "vllm", "--model_args", ",".join(model_args)]
        return cmd

    if backend == "openai-compatible":
        base = (api_base or DEFAULT_API_BASE).rstrip("/")
        model_args = [
            f"model={_shell_safe(model_ref)}",
            f"base_url={base}/v1",
            "tokenized_requests=False",
        ]
        if think_end_token:
            model_args.append("enable_thinking=True")
        cmd += ["--model", "local-chat-completions", "--model_args", ",".join(model_args)]
        cmd += ["--apply_chat_template"]
        return cmd

    raise ValueError(f"Unsupported backend: {backend!r} (use 'vllm' or 'openai-compatible')")


def build_evalplus_command(
    model_ref: str,
    *,
    tasks: Sequence[str],
    backend: str = "vllm",
    api_base: str | None = None,
) -> list[str]:
    """Build the ``evalplus.evaluate`` argv (best-effort; entrypoint-dependent)."""
    datasets = [t.removesuffix("_plus") for t in tasks]
    cmd = ["evalplus.evaluate", "--model", model_ref, "--datasets", *datasets, "--greedy"]
    if backend == "openai-compatible":
        cmd += ["--backend", "openai-compat", "--base-url", (api_base or DEFAULT_API_BASE)]
    else:
        cmd += ["--backend", "vllm"]
    return cmd


def build_eqbench_command(
    task: str,
    model_ref: str,
    *,
    api_base: str | None = None,
) -> list[str]:
    """Build the EqBench 3 argv for one component (best-effort)."""
    component = task.removeprefix("eqbench_") or task
    return [
        "python",
        "run_eqbench.py",
        "--model",
        model_ref,
        "--components",
        component,
        "--url",
        (api_base or DEFAULT_API_BASE),
    ]


# ---------------------------------------------------------------------------
# Pure helper: reporting
# ---------------------------------------------------------------------------


def _fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_delta(value: Any) -> str:
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    return f"{value:+.4f}"


def format_three_way_markdown_table(payload: dict[str, Any]) -> str:
    """Render Base vs Naive Abliteration vs Tenet metrics as markdown."""
    arms: dict[str, dict[str, Any]] = payload.get("arms", {}) or {}
    base_arm = arms.get("base", {}) or {}
    naive_arm = arms.get("naive", {}) or {}
    tenet_arm = arms.get("tenet", {}) or {}

    keys: list[str] = []
    for key in ("unclosed_think_rate",):
        if any(key in a for a in (base_arm, naive_arm, tenet_arm)) and key not in keys:
            keys.append(key)
    for arm in (base_arm, naive_arm, tenet_arm):
        for key in arm:
            if key in ("status", "elapsed_seconds", "error") or key.startswith("_"):
                continue
            if key not in keys:
                keys.append(key)

    deltas_base = payload.get("deltas_vs_base")
    deltas_naive = payload.get("deltas_vs_naive")
    if deltas_base is None:
        deltas_base = {
            k: round(float(tenet_arm.get(k, 0.0)) - float(base_arm.get(k, 0.0)), 6)
            for k in keys
            if isinstance(tenet_arm.get(k), (int, float))
            and isinstance(base_arm.get(k), (int, float))
        }
    if deltas_naive is None:
        deltas_naive = {
            k: round(float(tenet_arm.get(k, 0.0)) - float(naive_arm.get(k, 0.0)), 6)
            for k in keys
            if isinstance(tenet_arm.get(k), (int, float))
            and isinstance(naive_arm.get(k), (int, float))
        }

    lines = [
        f"# Three-way external eval: {payload.get('model_id', payload.get('release_id', 'unknown'))}",
        "",
        f"- Suite: `{payload.get('suite', 'full')}` | Backend: `{payload.get('backend', 'vllm')}`"
        f" | Precision: `{payload.get('precision', 'bf16')}` | Limit: `{payload.get('limit')}`",
        f"- Base: `{payload.get('base_model', '-')}`",
        f"- Naive Abliteration: `{payload.get('naive_model', '-')}`",
        f"- Tenet: `{payload.get('tenet_model', '-')}`",
        f"- Think-end token: `{payload.get('think_end_token', DEFAULT_THINK_END_TOKEN)}`"
        f" | Max gen toks: `{payload.get('max_gen_toks', DEFAULT_MAX_GEN_TOKS)}`",
        "",
        "| Metric | Base | Naive Abliteration | Tenet | Δ Tenet−Base | Δ Tenet−Naive |",
        "|---|---|---|---|---|---|",
    ]
    for key in keys:
        lines.append(
            f"| {key} | {_fmt_value(base_arm.get(key))} | {_fmt_value(naive_arm.get(key))} | "
            f"{_fmt_value(tenet_arm.get(key))} | {_fmt_delta(deltas_base.get(key))} | "
            f"{_fmt_delta(deltas_naive.get(key))} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _resolve_limit(cli_limit: int | None, suite: str) -> int | None:
    if cli_limit is None:
        return SUITE_PROFILES[suite]["limit"]
    return cli_limit if cli_limit > 0 else None


def _resolve_mmlu_pro_limit(cli_limit: int | None) -> int | None:
    """Resolve the mmlu_pro-only limit override (0/negative/absent = full split)."""
    if cli_limit is not None and cli_limit > 0:
        return cli_limit
    return None


def _flatten_lm_eval_scores(results: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task, metrics in (results.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if key in ("alias",):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            clean = key.split(",")[0]
            scores[f"{task}_{clean}"] = round(float(value), 6)
    return scores


def _load_lm_eval_payload(raw_dir: Path) -> dict[str, Any] | None:
    """Load the main lm-eval results JSON (the one holding a ``results`` dict)."""
    candidates = sorted(raw_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            return data
    return None


def _load_sample_dicts(raw_dir: Path, results: dict[str, Any] | None) -> list[dict[str, Any]]:
    embedded = (results or {}).get("samples")
    if isinstance(embedded, list) and embedded:
        return [s for s in embedded if isinstance(s, dict)]
    samples: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("samples_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                samples.append(record)
    return samples


def _run_optional_tool(
    cmd: list[str], *, cwd: Path | None, timeout_s: int = 3600
) -> dict[str, Any]:
    """Run an optional external entrypoint, degrading to a skip record."""
    if shutil.which(cmd[0]) is None:
        return {"status": "skipped", "reason": f"entrypoint not found: {cmd[0]}", "command": cmd}
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "reason": str(exc), "command": cmd}
    if proc.returncode != 0:
        return {
            "status": "error",
            "reason": (proc.stderr or proc.stdout or "")[-2000:],
            "command": cmd,
        }
    return {"status": "ok", "command": cmd, "stdout_tail": (proc.stdout or "")[-2000:]}


def _eval_arm(
    arm: str,
    model_ref: str,
    *,
    tasks: Sequence[str],
    evalplus_tasks: Sequence[str],
    eqbench_tasks: Sequence[str],
    raw_dir: Path,
    limit: int | None,
    mmlu_pro_limit: int | None = None,
    backend: str,
    precision: str,
    batch_size: int | str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_gen_toks: int,
    think_start_token: str,
    think_end_token: str,
    api_base: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _build_cmd(run_tasks: Sequence[str], output_path: Path, run_limit: int | None) -> list[str]:
        return build_lm_eval_command(
            model_ref,
            tasks=run_tasks,
            output_path=output_path,
            backend=backend,
            precision=precision,
            limit=run_limit,
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_gen_toks=max_gen_toks,
            think_end_token=think_end_token,
            api_base=api_base,
        )

    if mmlu_pro_limit is not None and mmlu_pro_limit > 0 and "mmlu_pro" in tasks:
        # Split the run: everything except mmlu_pro at the suite limit, and
        # mmlu_pro alone into its own subdirectory with the override limit.
        other_tasks = tuple(t for t in tasks if t != "mmlu_pro")
        mmlu_dir = raw_dir / "mmlu_pro"
        cmds: list[list[str]] = []
        sub_dirs: list[Path] = []
        if other_tasks:
            cmds.append(_build_cmd(other_tasks, raw_dir, limit))
            sub_dirs.append(raw_dir)
        cmds.append(_build_cmd(("mmlu_pro",), mmlu_dir, mmlu_pro_limit))
        sub_dirs.append(mmlu_dir)
    else:
        cmds = [_build_cmd(tasks, raw_dir, limit)]
        sub_dirs = [raw_dir]

    record: dict[str, Any] = {"arm": arm, "model": model_ref, "command": cmds[0]}
    if len(cmds) > 1:
        record["commands"] = cmds
    if dry_run:
        record["status"] = "dry-run"
        return record

    elapsed = 0.0
    scores: dict[str, float] = {}
    samples: list[dict[str, Any]] = []
    lm_eval_version: str | None = None
    payload_found = False
    for cmd, sub_dir in zip(cmds, sub_dirs):
        print(f"[{arm}] Running: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed += time.time() - t0
        if proc.returncode != 0:
            record["elapsed_seconds"] = round(elapsed, 1)
            record["status"] = "error"
            record["error"] = (proc.stderr or proc.stdout or "")[-4000:]
            print(f"[{arm}] lm-eval FAILED in {round(elapsed, 1)}s", flush=True)
            return record
        results = _load_lm_eval_payload(sub_dir)
        if results is None:
            print(f"[{arm}] WARNING: no lm-eval results JSON found under {sub_dir}", flush=True)
            continue
        payload_found = True
        scores.update(_flatten_lm_eval_scores(results))
        if lm_eval_version is None:
            lm_eval_version = results.get("git_hash") or results.get("config", {}).get(
                "lm_eval_version"
            )
        samples.extend(_load_sample_dicts(sub_dir, results))

    record["elapsed_seconds"] = round(elapsed, 1)
    record["status"] = "ok"
    if payload_found:
        record["scores"] = scores
        record["lm_eval_version"] = lm_eval_version
        record["num_sampled_responses"] = len(samples)
        record["unclosed_think_rate"] = round(
            extract_lm_eval_samples_unclosed_think_rate(
                samples,
                think_start_token=think_start_token,
                think_end_token=think_end_token,
            ),
            6,
        )
    else:
        record["scores"] = {}
        record["unclosed_think_rate"] = None

    if evalplus_tasks:
        record["evalplus"] = {
            task: _run_optional_tool(
                build_evalplus_command(model_ref, tasks=evalplus_tasks, backend=backend, api_base=api_base),
                cwd=REPO_ROOT,
            )
            for task in evalplus_tasks
        }
    if eqbench_tasks:
        record["eqbench"] = {
            task: _run_optional_tool(
                build_eqbench_command(task, model_ref, api_base=api_base), cwd=REPO_ROOT
            )
            for task in eqbench_tasks
        }
    print(f"[{arm}] Done in {elapsed}s: unclosed_think_rate={record.get('unclosed_think_rate')}", flush=True)
    return record


def _deltas(tenet: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        k: round(float(tenet[k]) - float(reference[k]), 6)
        for k in tenet
        if k in reference
        and isinstance(tenet[k], (int, float))
        and isinstance(reference[k], (int, float))
        and not isinstance(tenet[k], bool)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="3-way OpenEvals runner (vLLM / OpenAI-compatible).")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--naive-model", required=True, help="Naive Abliteration control model ref")
    parser.add_argument("--tenet-repo", required=True)
    parser.add_argument("--suite", choices=["full", "quick"], default="full")
    parser.add_argument("--limit", type=int, default=None, help="Override suite limit (0/negative = None)")
    parser.add_argument(
        "--mmlu-pro-limit",
        type=int,
        default=None,
        help="Override limit only for mmlu_pro (0/negative = full split)",
    )
    parser.add_argument("--backend", choices=["vllm", "openai-compatible", "hf"], default="vllm")
    parser.add_argument("--precision", choices=["bf16", "fp8", "gguf"], default="bf16")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-gen-toks", type=int, default=DEFAULT_MAX_GEN_TOKS)
    parser.add_argument("--think-end-token", default=DEFAULT_THINK_END_TOKEN)
    parser.add_argument("--dry-run", action="store_true", help="Print commands and exit without running")
    args = parser.parse_args()

    if args.backend == "hf":
        parser.error("--backend hf is not supported here; use run_external_benchmark_3way.py")
    backend = "openai-compatible" if args.precision == "gguf" else args.backend
    if args.precision == "gguf" and args.backend != "openai-compatible":
        print(f"precision=gguf -> forcing backend 'openai-compatible' against {args.api_base}", flush=True)

    profile = SUITE_PROFILES[args.suite]
    limit = _resolve_limit(args.limit, args.suite)
    mmlu_pro_limit = _resolve_mmlu_pro_limit(args.mmlu_pro_limit)
    tasks = profile["tasks"]
    evalplus_tasks = profile["evalplus_tasks"]
    eqbench_tasks = profile["eqbench_tasks"]

    base_raw = REPO_ROOT / "results" / "external_evals" / args.release_id / "raw"
    arms: dict[str, dict[str, Any]] = {}
    model_refs = {"base": args.base_model, "naive": args.naive_model, "tenet": args.tenet_repo}
    for arm in ARMS:
        arms[arm] = _eval_arm(
            arm,
            model_refs[arm],
            tasks=tasks,
            evalplus_tasks=evalplus_tasks,
            eqbench_tasks=eqbench_tasks,
            raw_dir=base_raw / arm,
            limit=limit,
            mmlu_pro_limit=mmlu_pro_limit,
            backend=backend,
            precision=args.precision,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_gen_toks=args.max_gen_toks,
            think_start_token=DEFAULT_THINK_START_TOKEN,
            think_end_token=args.think_end_token,
            api_base=args.api_base,
            dry_run=args.dry_run,
        )
        (base_raw / arm / "arm_summary.json").write_text(
            json.dumps(arms[arm], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if args.dry_run:
        print("Dry run complete; no lm-eval/vllm processes were launched.", flush=True)
        return 0

    failed = [arm for arm in ARMS if arms[arm].get("status") == "error"]
    if failed:
        print(f"ERROR: arms failed: {', '.join(failed)}", flush=True)
        return 1

    payload: dict[str, Any] = {
        "model_id": args.release_id,
        "release_id": args.release_id,
        "base_model": args.base_model,
        "naive_model": args.naive_model,
        "tenet_model": args.tenet_repo,
        "suite": args.suite,
        "backend": backend,
        "precision": args.precision,
        "limit": limit,
        "mmlu_pro_limit": mmlu_pro_limit,
        "max_gen_toks": args.max_gen_toks,
        "max_model_len": args.max_model_len,
        "think_end_token": args.think_end_token,
        "api_base": args.api_base if backend == "openai-compatible" else None,
        "tasks": list(tasks),
        "evalplus_tasks": list(evalplus_tasks),
        "eqbench_tasks": list(eqbench_tasks),
        "arms": arms,
        "before": arms["base"],
        "after": arms["tenet"],
        "external_benchmarks": arms["tenet"],
        "deltas": _deltas(arms["tenet"], arms["base"]),
        "deltas_vs_base": _deltas(arms["tenet"], arms["base"]),
        "deltas_vs_naive": _deltas(arms["tenet"], arms["naive"]),
    }
    payload["markdown_report"] = format_three_way_markdown_table(payload)

    canonical = REPO_ROOT / "results" / "external_evals" / f"{args.release_id}.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    release_dir = REPO_ROOT / "results" / "tenet_releases" / args.release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "external_benchmarks.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_path = REPO_ROOT / "results" / "external_evals" / args.release_id / "report.md"
    report_path.write_text(format_three_way_markdown_table(payload), encoding="utf-8")
    print(f"Saved canonical evals to {canonical}", flush=True)
    print(f"Saved markdown report to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
