"""Tests for scripts/remote/run_wave.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "remote"))

import run_wave  # noqa: E402

PY = sys.executable


def test_load_env_file_parses_and_pins_flags(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "HF_TOKEN=abc123\n"
        'QUOTED="double quoted"\n'
        "SINGLE='single quoted'\n"
        "SPACED = spaced value \n"
        "NO_EQUALS_LINE\n"
    )
    env: dict[str, str] = {}
    run_wave.load_env_file(env_file, env)

    assert env["HF_TOKEN"] == "abc123"
    assert env["QUOTED"] == "double quoted"
    assert env["SINGLE"] == "single quoted"
    assert env["SPACED"] == "spaced value"
    assert "NO_EQUALS_LINE" not in env
    assert env["UV_NO_SYNC"] == "1"
    assert env["SKIP_UV_SYNC"] == "1"


def test_load_env_file_pins_flags_even_without_file(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    run_wave.load_env_file(tmp_path / "missing.env", env)
    assert env["UV_NO_SYNC"] == "1"
    assert env["SKIP_UV_SYNC"] == "1"


def test_build_wave_commands_wave1() -> None:
    cmds = run_wave.build_wave_commands(1)

    assert len(cmds) == 2  # 1 foundry model + 1 gguf ladder batch
    foundry = cmds[0]
    assert foundry[0] == PY
    assert foundry[1] == "scripts/run_tenet_v3_foundry.py"
    assert foundry[foundry.index("--model-id") + 1] == "Tenet-30B-A3B-GLM4.7-Unslop-v3"
    assert foundry[foundry.index("--release-id") + 1] == "Tenet-30B-A3B-GLM4.7-Unslop-v3"
    for flag in (
        "--download",
        "--run-fight-card-gate",
        "--run-strata-bench",
        "--run-open-evals",
        "--upload-hf",
        "--cleanup-cache",
        "--keep-going-on-gate-fail",
    ):
        assert flag in foundry
    assert foundry[foundry.index("--eval-suite") + 1] == "full"
    assert foundry[foundry.index("--mmlu-pro-limit") + 1] == "72"

    gguf = cmds[1]
    assert gguf[0] == PY
    assert gguf[1] == "scripts/hub/build_gguf_ladder.py"
    ids = [gguf[i + 1] for i, tok in enumerate(gguf) if tok == "--model-id"]
    assert ids == [
        "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
        "Tenet-30B-A3B-GLM4.7-Unslop-v3",
    ]
    assert gguf[gguf.index("--llama-cpp-dir") + 1] == "/mnt/nvme/llama.cpp"
    assert gguf[gguf.index("--work-dir") + 1] == "/mnt/nvme/gguf_work"
    assert gguf[gguf.index("--threads") + 1] == "16"
    assert "--upload" in gguf


def test_build_wave_commands_wave2_no_gguf() -> None:
    cmds = run_wave.build_wave_commands(2)

    assert len(cmds) == 4  # four foundry models, no gguf batch
    ids = [c[c.index("--model-id") + 1] for c in cmds]
    assert ids == [
        "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
        "Tenet-106B-A12B-GLM4.5-Unslop-v3",
        "Tenet-125B-A6B-Qwen3.8-Unslop-v3",
        "Tenet-32B-A3B-GLM4.5-Unslop-v3",
    ]
    assert all(c[1] == "scripts/run_tenet_v3_foundry.py" for c in cmds)
    assert not any("build_gguf_ladder.py" in c for c in cmds)


def test_build_wave_commands_overrides() -> None:
    cmds = run_wave.build_wave_commands(
        1,
        models=["Custom-Model-v1"],
        mmlu_pro_limit=10,
        eval_suite="smoke",
        skip_gguf=True,
    )
    assert len(cmds) == 1
    c = cmds[0]
    assert c[c.index("--model-id") + 1] == "Custom-Model-v1"
    assert c[c.index("--mmlu-pro-limit") + 1] == "10"
    assert c[c.index("--eval-suite") + 1] == "smoke"


def test_main_dry_run_wave1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    env_file = tmp_path / "env"
    env_file.write_text("HF_TOKEN=x\n")

    rc = run_wave.main(["--wave", "1", "--dry-run", "--env-file", str(env_file)])

    assert rc == 0
    out = capsys.readouterr().out
    cmds = json.loads(out)
    assert len(cmds) == 2
    assert cmds[0][1] == "scripts/run_tenet_v3_foundry.py"
    assert cmds[1][1] == "scripts/hub/build_gguf_ladder.py"


def test_main_unknown_wave_rejected() -> None:
    with pytest.raises(SystemExit):
        run_wave.main(["--wave", "3", "--dry-run"])
