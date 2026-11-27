#!/usr/bin/env python3
"""Single, idempotent wave runner for remote VM jobs.

Instead of passing fragile multi-line bash strings (POD_CMD) over SSH, this
script is copied to the VM and invoked with a small argv surface:

    python3 scripts/remote/run_wave.py --wave 1
    python3 scripts/remote/run_wave.py --wave 2 --skip-gguf --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path

# ModelId -> ReleaseId is 1:1 for this batch (model id equals release id).
WAVE_SPECS: dict[int, dict[str, list[str]]] = {
    1: {
        "models": [
            "Tenet-30B-A3B-GLM4.7-Unslop-v3",
        ],
        "gguf_models": [
            "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
            "Tenet-30B-A3B-GLM4.7-Unslop-v3",
        ],
    },
    2: {
        "models": [
            "Tenet-106B-A12B-GLM4.5-Unslop-v3",
            "Tenet-125B-A6B-Qwen3.8-Unslop-v3",
            "Tenet-32B-A3B-GLM4.5-Unslop-v3",
        ],
        "gguf_models": [],
    },
}


def load_env_file(
    path: Path,
    environ: MutableMapping[str, str],
) -> None:
    """Parse a dotenv-style file into ``environ``.

    - Blank lines and ``#`` comments are ignored.
    - Lines are ``KEY=VALUE``; whitespace around KEY/VALUE is stripped.
    - Optional surrounding single or double quotes on VALUE are removed.
    - Always pins UV offline sync flags so the VM never re-resolves deps.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            environ[key] = value

    # Hard-pin: the wave runner must never trigger a uv re-sync on the VM.
    environ["UV_NO_SYNC"] = "1"
    environ["SKIP_UV_SYNC"] = "1"


def build_wave_commands(
    wave: int,
    *,
    models: list[str] | None = None,
    mmlu_pro_limit: int = 72,
    eval_suite: str = "full",
    skip_gguf: bool = False,
) -> list[list[str]]:
    """Build the ordered commands for one wave."""
    if wave not in WAVE_SPECS:
        raise ValueError(f"unknown wave: {wave}")

    model_list = models if models is not None else list(WAVE_SPECS[wave]["models"])

    commands: list[list[str]] = []

    for model_id in model_list:
        commands.append(
            [
                sys.executable,
                "scripts/run_tenet_v3_foundry.py",
                "--model-id",
                model_id,
                "--release-id",
                model_id,
                "--download",
                "--run-fight-card-gate",
                "--run-strata-bench",
                "--run-open-evals",
                "--eval-suite",
                eval_suite,
                "--mmlu-pro-limit",
                str(mmlu_pro_limit),
                "--upload-hf",
                "--cleanup-cache",
                "--keep-going-on-gate-fail",
            ]
        )

    gguf_models = list(WAVE_SPECS[wave]["gguf_models"])
    if not skip_gguf and gguf_models:
        gguf_cmd: list[str] = [
            sys.executable,
            "scripts/hub/build_gguf_ladder.py",
        ]
        for gguf_model in gguf_models:
            gguf_cmd += ["--model-id", gguf_model]
        gguf_cmd += [
            "--llama-cpp-dir",
            "/mnt/nvme/llama.cpp",
            "--work-dir",
            "/mnt/nvme/gguf_work",
            "--threads",
            "16",
            "--upload",
        ]
        commands.append(gguf_cmd)

    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, required=True, choices=sorted(WAVE_SPECS))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--mmlu-pro-limit", type=int, default=72)
    parser.add_argument("--eval-suite", choices=["smoke", "full"], default="full")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-s3-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)

    load_env_file(Path(args.env_file), os.environ)
    commands = build_wave_commands(
        args.wave,
        models=args.models,
        mmlu_pro_limit=args.mmlu_pro_limit,
        eval_suite=args.eval_suite,
        skip_gguf=args.skip_gguf,
    )

    if args.dry_run:
        print(json.dumps(commands))
        return 0

    try:
        for cmd in commands:
            subprocess.run(cmd, check=True)
    finally:
        if not args.skip_s3_sync:
            # Best effort: never let the sync failure wedge the worker.
            with contextlib.suppress(OSError):
                subprocess.run(
                    ["bash", "scripts/remote/nebius_ctl.sh", "s3-sync-up"],
                    check=False,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
