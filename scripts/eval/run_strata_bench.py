#!/usr/bin/env python3
"""Run Strata Engine benchmark and log metrics to MLflow.

Collects the 5 Strata Engine metrics:
  - ttft_ms
  - decode_tok_per_s
  - expert_cache_hit_rate
  - coactivation_graph_drift
  - speculative_acceptance_rate

Backends:
  - strata: real benchmark via the ``strata-bench`` CLI (JSON metrics output).
  - sim: deterministic simulation metrics (no external tooling required).

Writes a JSON report and logs metrics (plus the strata TOML artifact when
present) via ``sml.tracking.nebius_mlflow.log_strata_bench_run``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sml.tracking.nebius_mlflow import log_strata_bench_run  # noqa: E402

DEFAULT_MODEL_NAME = "Tenet-26B-A4B-Gemma4-Unslop-v2"
DEFAULT_BASE_MODEL = "google/gemma-4-26B-A4B-it"

METRIC_KEYS = (
    "ttft_ms",
    "decode_tok_per_s",
    "expert_cache_hit_rate",
    "coactivation_graph_drift",
    "speculative_acceptance_rate",
)


def simulate_metrics(model_name: str, base_model: str) -> dict[str, float]:
    """Deterministic simulation metrics for the sim backend."""
    del model_name, base_model  # currently model-independent
    return {
        "ttft_ms": 128.0,
        "decode_tok_per_s": 74.5,
        "expert_cache_hit_rate": 0.94,
        "coactivation_graph_drift": 0.0,
        "speculative_acceptance_rate": 0.71,
    }


def dry_run_metrics() -> dict[str, float]:
    """Placeholder metrics emitted for a dry run of the strata backend."""
    return {
        "ttft_ms": 0.0,
        "decode_tok_per_s": 0.0,
        "expert_cache_hit_rate": 0.0,
        "coactivation_graph_drift": 0.0,
        "speculative_acceptance_rate": 0.0,
    }


def run_strata_backend(
    model_name: str,
    base_model: str,
    strata_toml: str | None,
    dry_run: bool,
) -> dict[str, float]:
    """Run (or simulate) the real strata-bench tool and parse its metrics."""
    if dry_run:
        return dry_run_metrics()

    bench = shutil.which("strata-bench")
    if bench is None:
        raise SystemExit(
            "error: --backend strata requires the 'strata-bench' executable on PATH. "
            "Install Strata Engine tooling or use --backend sim."
        )

    cmd: list[str] = [bench, "--model-name", model_name, "--base-model", base_model, "--json"]
    if strata_toml:
        cmd += ["--strata-toml", strata_toml]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            f"error: strata-bench exited with code {proc.returncode}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: failed to parse strata-bench JSON output: {exc}") from exc

    raw = payload.get("metrics", payload) if isinstance(payload, dict) else {}
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = raw.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        raise SystemExit(f"error: strata-bench output missing metrics: {', '.join(missing)}")
    return metrics


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Strata Engine benchmark and log metrics to MLflow.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Model to benchmark (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model identifier (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--release-id",
        default=None,
        help="Release identifier used for run naming and report path (default: model name)",
    )
    parser.add_argument(
        "--strata-toml",
        default=None,
        help="Optional path to strata.toml (attached as an MLflow artifact when present)",
    )
    parser.add_argument(
        "--backend",
        choices=["strata", "sim"],
        default="sim",
        help="Benchmark backend (default: sim)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit placeholder metrics without running the benchmark",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON report path (default: results/strata_bench/<release_id>.json)",
    )
    parser.add_argument(
        "--mirror-dir",
        default=None,
        help="Optional mirror directory forwarded to log_strata_bench_run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    release_id = args.release_id or args.model_name
    output_path = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "results" / "strata_bench" / f"{release_id}.json"
    )

    if args.backend == "strata":
        metrics = run_strata_backend(
            model_name=args.model_name,
            base_model=args.base_model,
            strata_toml=args.strata_toml,
            dry_run=args.dry_run,
        )
    else:
        metrics = simulate_metrics(args.model_name, args.base_model)

    report: dict[str, Any] = {
        "release_id": release_id,
        "model_name": args.model_name,
        "base_model": args.base_model,
        "backend": args.backend,
        "dry_run": args.dry_run,
        "strata_toml": args.strata_toml,
        "metrics": metrics,
    }
    write_report(output_path, report)
    print(f"report written: {output_path}")

    result = log_strata_bench_run(
        release_id=release_id,
        strata_metrics=metrics,
        strata_toml_path=args.strata_toml,
        mirror_dir=args.mirror_dir,
    )
    print(f"mlflow run: {result.get('run_name', 'logged')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
