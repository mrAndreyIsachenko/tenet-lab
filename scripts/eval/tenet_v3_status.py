#!/usr/bin/env python3
"""Live status dashboard for Tenet v3 runs (stdlib only)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

PROGRESS_PREFIXES = (
    "Processed prompts:",
    "[supervisor]",
    "[foundry]",
    "[strata]",
    "[base] Running:",
    "[tenet] Running:",
    "ERROR:",
)

ANSI_RE = None


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)


def extract_live_progress(log_text: str) -> str:
    """Return the most recent meaningful progress line from worker.log text."""
    for raw in reversed(log_text.splitlines()):
        line = _strip_ansi(raw).replace("\r", " ").strip()
        # tqdm overwrites lines with \r; keep the last segment
        if "\r" in raw:
            line = _strip_ansi(raw.rsplit("\r", 1)[-1]).strip()
        if not line:
            continue
        if any(line.startswith(p) for p in PROGRESS_PREFIXES):
            return line
    return "(no active log)"


def _pct(v) -> str:
    try:
        f = float(v)
        return f"{f * 100:.1f}%" if f <= 1.0 else f"{f:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _load(path: Path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _model_display(name: str) -> str:
    return name


def collect_model_rows(results_dir: Path) -> list[dict[str, str]]:
    """Discover models and gather per-model status/eval metrics."""
    foundry_dir = results_dir / "tenet_v3_foundry"
    ext_dir = results_dir / "external_evals"

    def legacy(n):
        return n in ("tenet-v3-foundry-v1",) or n.endswith("-v1") or n.endswith("-v2")

    models: list[str] = []
    if foundry_dir.is_dir():
        models += [p.name for p in foundry_dir.iterdir() if p.is_dir() and not legacy(p.name)]
    if ext_dir.is_dir():
        models += [p.stem for p in ext_dir.glob("*.json") if not legacy(p.stem)]
    models = sorted(set(models))

    rows = []
    for m in models:
        row = {
            "model": m,
            "status": "-",
            "h21_22": "-",
            "gate": "-",
            "strata": "-",
            "mmlu_pro": "-",
            "math_500": "-",
            "gsm8k": "-",
            "gpqa": "-",
            "ifeval": "-",
        }

        fs = _load(foundry_dir / m / "foundry_summary.json")
        if fs:
            row["status"] = fs.get("status", "-")
            h21 = fs.get("h21_verified")
            if h21 is None and isinstance(fs.get("h21"), dict):
                h21 = fs["h21"].get("h21_verified")
            h22 = fs.get("h22_verified")
            if h22 is None and isinstance(fs.get("h22"), dict):
                h22 = fs["h22"].get("h22_verified")
            if h21 is not None or h22 is not None:
                row["h21_22"] = f"{'✓' if h21 else '✗'}/{'✓' if h22 else '✗'}"

        gate = _load(foundry_dir / m / "fight_card_gate.json")
        if gate:
            gp = gate.get("gate_passed")
            if gp is not None:
                row["gate"] = "pass" if gp else "FAIL"
            elif (
                gate.get("cot_tag_integrity_rate") is not None
                or gate.get("length_collapse_rate") is not None
            ):
                row["gate"] = _pct(gate.get("cot_tag_integrity_rate"))

        strata = _load(foundry_dir / m / "strata_bench.json")
        if strata:
            if isinstance(strata.get("passed"), bool):
                row["strata"] = "pass" if strata["passed"] else "FAIL"
            elif strata.get("status"):
                row["strata"] = str(strata["status"])
            elif isinstance(strata.get("metrics"), dict) and not strata.get("dry_run", False):
                row["strata"] = "ok"

        ev = _load(ext_dir / f"{m}.json")
        if ev:
            after = ev.get("after", {})
            scores = after.get("scores", after)
            if scores.get("mmlu_pro") is not None:
                row["mmlu_pro"] = _pct(scores["mmlu_pro"])
            for k in ("math_500", "hendrycks_math500"):
                if scores.get(k) is not None:
                    row["math_500"] = _pct(scores[k])
                    break
            for k in ("gsm8k_math_acc", "gsm8k_cot_exact_match", "gsm8k"):
                if scores.get(k) is not None:
                    row["gsm8k"] = _pct(scores[k])
                    break
            for k in ("gpqa_diamond", "gpqa_diamond_zeroshot"):
                if scores.get(k) is not None:
                    row["gpqa"] = _pct(scores[k])
                    break
            for k in ("ifeval_strict", "ifeval_prompt_level_strict_acc"):
                if scores.get(k) is not None:
                    row["ifeval"] = _pct(scores[k])
                    break
        rows.append(row)
    return rows


def _remote_ssh_host() -> str | None:
    for key in ("NEBIUS_SSH_HOST", "POD_HOST"):
        v = os.environ.get(key)
        if v:
            return v
    env_file = Path(".env.nebius")
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            for key in ("NEBIUS_SSH_HOST", "POD_HOST"):
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _ssh_ok(host: str) -> bool:
    if shutil.which("ssh") is None:
        return False
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes", host, "true"],
            capture_output=True,
            timeout=10,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _ssh_run(host: str, cmd: str, timeout: int = 15) -> str:
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes", host, cmd],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def fetch_remote_info() -> dict[str, str]:
    host = _remote_ssh_host()
    info: dict[str, str] = {}
    if not host or not _ssh_ok(host):
        info["pod"] = "remote unreachable"
        return info
    gpu = _ssh_run(
        host,
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total"
        " --format=csv,noheader | head -n 1",
    )
    if gpu:
        info["gpu"] = gpu
    pod = _ssh_run(host, "cat results/.pod_status.json 2>/dev/null")
    info["pod_raw"] = pod
    info["log_tail"] = _ssh_run(host, "tail -n 30 results/worker.log 2>/dev/null")
    try:
        st = json.loads(pod).get("status", "?")
    except Exception:
        st = "?"
    info["pod"] = st or "?"
    return info


def render_dashboard(results_dir: Path, *, remote_info: dict[str, str] | None = None) -> str:
    ri = remote_info or {}
    lines: list[str] = []

    # Header: pod status
    pod = ri.get("pod")
    if not pod:
        pod_local = _load(results_dir / ".pod_status.json")
        pod = (pod_local or {}).get("status", "unknown")
    lines.append(f"Pod status: {pod}")

    gpu = ri.get("gpu", "")
    if gpu:
        lines.append(f"GPU (util,mem used/total): {gpu}")

    log_text = ri.get("log_tail", "")
    if not log_text:
        log_path = results_dir / "worker.log"
        if log_path.is_file():
            try:
                log_text = log_path.read_text(errors="replace")
            except OSError:
                log_text = ""
    lines.append(f"Live: {extract_live_progress(log_text)}")
    lines.append("")

    rows = collect_model_rows(results_dir)
    headers = [
        "Model",
        "Status",
        "H21/22",
        "Gate",
        "Strata",
        "MMLU-Pro",
        "MATH-500",
        "GSM8K",
        "GPQA",
        "IFEval",
    ]
    keys = [
        "model",
        "status",
        "h21_22",
        "gate",
        "strata",
        "mmlu_pro",
        "math_500",
        "gsm8k",
        "gpqa",
        "ifeval",
    ]
    widths = [
        max(len(h), *(len(str(r[k])) for r in rows)) if rows else len(h)
        for h, k in zip(headers, keys, strict=False)
    ]

    def fmt(r) -> str:
        return "  ".join(str(r[k]).ljust(w) for k, w in zip(keys, widths, strict=False))

    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False)))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt(r))
    lines.append("")

    out = "\n".join(lines)
    print(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tenet v3 live status dashboard")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    ap.add_argument("--remote", action="store_true", help="fetch GPU/pod/log from VM over ssh")
    args = ap.parse_args()

    if args.remote:
        try:
            remote_info = fetch_remote_info()
        except Exception as e:
            remote_info = {"pod": f"remote error: {e}"}
    else:
        remote_info = None

    try:
        while True:
            if args.watch > 0:
                print("\x1b[2J\x1b[H", end="")
            render_dashboard(Path(args.results_dir), remote_info=remote_info)
            if args.watch <= 0:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
