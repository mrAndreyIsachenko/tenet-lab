"""Unit tests for Nebius/Strata infra: MLflow mirror fallback, strata bundle,
strata bench script, and the nebius_ctl.sh dry-run surface."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# ---------------------------------------------------------------- module loading


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nebius_mlflow = _load_module(
    "sml_tracking_nebius_mlflow_under_test", SRC / "sml" / "tracking" / "nebius_mlflow.py"
)
strata_bundle = _load_module(
    "sml_hub_strata_bundle_under_test", SRC / "sml" / "hub" / "strata_bundle.py"
)
run_strata_bench = _load_module(
    "run_strata_bench_under_test", REPO_ROOT / "scripts" / "eval" / "run_strata_bench.py"
)
backfill = _load_module(
    "backfill_mlflow_history_under_test",
    REPO_ROOT / "scripts" / "hub" / "backfill_mlflow_history.py",
)


@pytest.fixture(autouse=True)
def _offline_mlflow(monkeypatch, tmp_path):
    """Guarantee the local-mirror fallback: no tracking URI, no repo .env."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.chdir(tmp_path)  # avoid picking up a repo-root .env


# ------------------------------------------------------- nebius_mlflow fallbacks


def test_log_foundry_surgery_run_mirror_fallback(tmp_path):
    mirror = tmp_path / "mirror"
    result = nebius_mlflow.log_foundry_surgery_run(
        release_id="Tenet-Test-v1",
        params={"base_model": "google/gemma-4-26B-A4B-it", "generation": "v1", "scale": 0.3},
        surgery_metrics={"router_weight_drift": 0.01, "wall_time_sec": 12.5},
        gate_metrics={"perplexity": 7.4, "note": "non-numeric-ignored"},
        artifacts_paths=None,
        mirror_dir=mirror,
    )
    assert result["remote_logged"] is False
    assert result["metrics"]["router_weight_drift"] == pytest.approx(0.01)
    assert result["metrics"]["perplexity"] == pytest.approx(7.4)
    assert "note" not in result["metrics"]
    mirror_files = list(mirror.glob("*.json"))
    assert len(mirror_files) == 1
    payload = json.loads(mirror_files[0].read_text(encoding="utf-8"))
    assert payload["run_name"] == "foundry_surgery_Tenet-Test-v1"
    assert payload["remote_logged"] is False
    assert payload["params"]["base_model"] == "google/gemma-4-26B-A4B-it"


def test_log_open_evals_subtask_mirror_fallback(tmp_path):
    mirror = tmp_path / "mirror"
    result = nebius_mlflow.log_open_evals_subtask(
        release_id="Tenet-Test-v2",
        arm="after",
        subtask_name="mmlu",
        scores_dict={"acc": 0.61, "f1": 0.55, "skip_me": "text"},
        mirror_dir=mirror,
        tags={"generation": "v2"},
    )
    assert result["remote_logged"] is False
    assert result["tags"]["arm"] == "after"
    assert result["metrics"] == {"acc": pytest.approx(0.61), "f1": pytest.approx(0.55)}
    mirror_files = list(mirror.glob("*.json"))
    assert len(mirror_files) == 1
    payload = json.loads(mirror_files[0].read_text(encoding="utf-8"))
    assert payload["run_name"] == "open_evals_Tenet-Test-v2_after_mmlu"
    assert payload["remote_logged"] is False


def test_log_strata_bench_run_mirror_fallback(tmp_path):
    mirror = tmp_path / "mirror"
    metrics = {
        "ttft_ms": 120.0,
        "decode_tok_per_s": 70.0,
        "expert_cache_hit_rate": 0.9,
        "coactivation_graph_drift": 0.0,
        "speculative_acceptance_rate": 0.7,
    }
    result = nebius_mlflow.log_strata_bench_run(
        release_id="Tenet-Test-v2",
        strata_metrics=metrics,
        strata_toml_path=None,
        mirror_dir=mirror,
    )
    assert result["remote_logged"] is False
    assert set(result["metrics"]) == set(metrics)
    mirror_files = list(mirror.glob("*.json"))
    assert len(mirror_files) == 1
    payload = json.loads(mirror_files[0].read_text(encoding="utf-8"))
    assert payload["run_name"] == "strata_bench_Tenet-Test-v2"
    assert payload["remote_logged"] is False
    assert payload["metrics"]["ttft_ms"] == pytest.approx(120.0)


# ------------------------------------------------- backfill_mlflow_history dry-run


def _make_results_tree(results_dir: Path) -> None:
    release = results_dir / "tenet_releases" / "Tenet-4B-Qwen3-Unslop-v2"
    release.mkdir(parents=True)
    (release / "foundry_summary.json").write_text(
        json.dumps(
            {
                "surgery_metrics": {"router_weight_drift": 0.02, "wall_time_sec": 33.0},
                "gate_metrics": {"perplexity": 6.9},
            }
        ),
        encoding="utf-8",
    )
    (release / "release_manifest.json").write_text(
        json.dumps({"base_model": "Qwen/Qwen3-4B", "generation": "v2"}),
        encoding="utf-8",
    )

    evals = results_dir / "external_evals"
    evals.mkdir(parents=True)
    (evals / "Tenet-4B-Qwen3-Unslop-v2.json").write_text(
        json.dumps(
            {
                "arms": {
                    "base": {"tasks": {"mmlu": {"acc": 0.55}}},
                    "after": {"tasks": {"mmlu": {"acc": 0.62}, "gsm8k": {"acc": 0.71}}},
                }
            }
        ),
        encoding="utf-8",
    )

    h18 = results_dir / "h18_nvfp4_quant_snap"
    h18.mkdir(parents=True)
    (h18 / "h18_summary.json").write_text(
        json.dumps({"summary_metrics": {"expert_cache_hit_rate": 0.93}}),
        encoding="utf-8",
    )
    h19 = results_dir / "h19_loop_unrolled_lancet"
    h19.mkdir(parents=True)
    (h19 / "h19_summary.json").write_text(
        json.dumps({"metrics": {"speculative_acceptance_rate": 0.72}}),
        encoding="utf-8",
    )


def test_backfill_mlflow_history_dry_run(tmp_path):
    results_dir = tmp_path / "results"
    _make_results_tree(results_dir)

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "hub" / "backfill_mlflow_history.py"),
            "--results-dir",
            str(results_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "MLFLOW_TRACKING_URI": ""},
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    # Dry run without --mirror-dir writes no mirror files.
    assert not (tmp_path / "results" / "mlflow_mirror").exists()

    summary_start = proc.stdout.index("{")
    summary = json.loads(proc.stdout[summary_start:])
    assert summary["dry_run"] is True

    assert len(summary["releases"]) == 1
    release = summary["releases"][0]
    assert release["release"] == "Tenet-4B-Qwen3-Unslop-v2"
    assert release["generation"] == "v2"
    assert "router_weight_drift" in release["surgery_metrics_keys"]
    assert "perplexity" in release["gate_metrics_keys"]

    evals = {(entry["arm"], entry["subtask"]): entry for entry in summary["external_evals"]}
    assert {arm for arm, _ in evals} == {"base", "after"}
    assert all(entry["generation"] == "v2" for entry in evals.values())
    assert evals[("after", "mmlu")]["scores"]["acc"] == pytest.approx(0.62)
    assert evals[("after", "gsm8k")]["scores"]["acc"] == pytest.approx(0.71)

    hypotheses = {entry["hypothesis"] for entry in summary["hypotheses"]}
    assert hypotheses == {"h18", "h19"}
    assert len(summary["hypotheses"]) == 2


# ------------------------------------------------------------ strata_bundle tests


def test_build_strata_tier_profile_tiers():
    profiles = {}
    cka = {}
    for i in range(10):
        expert_id = f"expert_{i:02d}"
        profiles[expert_id] = {"activity": i / 10.0}
        cka[expert_id] = 0.0
    profiles["shared_expert"] = {"activity": 0.99}
    profiles["attention"] = {"activity": 0.99}
    profiles["drafter"] = {"activity": 0.99}

    result = strata_bundle.build_strata_tier_profile(profiles, cka, hot_vram_ratio=0.10)
    assert result["total_experts"] == 13
    hot_ids = {entry["expert_id"] for entry in result["vram_hot_tier"]}
    cold_ids = {entry["expert_id"] for entry in result["nvme_cold_tier"]}
    assert hot_ids | cold_ids == set(profiles)
    assert not hot_ids & cold_ids

    # Always-resident components must be hot.
    assert {"shared_expert", "attention", "drafter"} <= hot_ids
    # ceil(13 * 0.10) = 2 ranked slots on the 10 rankable experts; top activity wins.
    assert {"expert_09", "expert_08"} <= hot_ids
    # The remaining ~80% of rankable experts stay cold.
    assert cold_ids == {f"expert_{i:02d}" for i in range(8)}
    assert result["hot_expert_count"] == len(result["vram_hot_tier"])
    assert result["cold_expert_count"] == len(result["nvme_cold_tier"])


def test_build_strata_tier_profile_quant_always_hot():
    profiles = {
        "expert_a": {"activity": 0.1, "quant": "Q8_0"},
        "expert_b": {"activity": 0.9, "quant": "Q4_K_M"},
    }
    result = strata_bundle.build_strata_tier_profile(profiles, [0.0, 0.0], hot_vram_ratio=0.5)
    hot_ids = {entry["expert_id"] for entry in result["vram_hot_tier"]}
    # Q8_0 is always hot even with low activity; Q4_K_M ranked expert is hot too.
    assert hot_ids == {"expert_a", "expert_b"}


def test_render_strata_toml_parses():
    profile = strata_bundle.build_strata_tier_profile(
        {"expert_0": {"activity": 1.0}, "expert_1": {"activity": 0.0}},
        [0.5, 0.1],
        hot_vram_ratio=0.5,
    )
    text = strata_bundle.render_strata_toml(
        model_name="Tenet-26B-A4B-Gemma4-Unslop-v2",
        base_model="google/gemma-4-26B-A4B-it",
        tier_profile=profile,
        gguf_filename="tenet-26b-Q4_K_M.gguf",
    )
    data = tomllib.loads(text)
    assert data["model"]["name"] == "Tenet-26B-A4B-Gemma4-Unslop-v2"
    assert data["model"]["base"] == "google/gemma-4-26B-A4B-it"
    assert data["model"]["gguf"] == "tenet-26b-Q4_K_M.gguf"
    assert data["model"]["total_experts"] == 2
    assert data["tiers"]["vram_hot"]["quant"] == strata_bundle.HOT_TIER_ALWAYS_QUANT
    assert data["tiers"]["nvme_cold"]["quant"] == strata_bundle.COLD_TIER_LABEL
    hot_ids = set(data["tiers"]["vram_hot"]["experts"])
    cold_ids = set(data["tiers"]["nvme_cold"]["experts"])
    assert hot_ids | cold_ids == {"expert_0", "expert_1"}
    assert data["runtime"]["offload_backend"] == "nvme"


def test_compute_coactivation_graph_drift():
    import numpy as np

    rng = np.random.default_rng(42)
    base = rng.dirichlet(np.ones(8), size=64)

    # Identical routing -> exactly zero drift.
    assert strata_bundle.compute_coactivation_graph_drift(base, base) == 0.0

    # Negligible perturbation stays below 1e-6.
    tiny = base + 1e-10 * rng.random(base.shape)
    assert strata_bundle.compute_coactivation_graph_drift(base, tiny) < 1e-6

    # Divergent routing (expert permutation) is clearly detectable.
    divergent = base[:, np.random.permutation(base.shape[1])]
    assert strata_bundle.compute_coactivation_graph_drift(base, divergent) > 1e-3

    with pytest.raises(ValueError):
        strata_bundle.compute_coactivation_graph_drift(base, base[:, :4])


# ------------------------------------------------------------ run_strata_bench sim


def test_run_strata_bench_sim_writes_report_and_mirror(tmp_path):
    output = tmp_path / "strata.json"
    mirror = tmp_path / "mirror"
    exit_code = run_strata_bench.main(
        [
            "--backend",
            "sim",
            "--dry-run",
            "--release-id",
            "Tenet-Sim-Test",
            "--output",
            str(output),
            "--mirror-dir",
            str(mirror),
        ]
    )
    assert exit_code == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["backend"] == "sim"
    assert report["dry_run"] is True
    for key in (
        "ttft_ms",
        "decode_tok_per_s",
        "expert_cache_hit_rate",
        "coactivation_graph_drift",
        "speculative_acceptance_rate",
    ):
        assert key in report["metrics"]
        assert isinstance(report["metrics"][key], float)

    mirror_files = list(mirror.glob("strata_bench_*.json"))
    assert len(mirror_files) == 1
    mirror_payload = json.loads(mirror_files[0].read_text(encoding="utf-8"))
    assert mirror_payload["remote_logged"] is False
    for key in (
        "ttft_ms",
        "decode_tok_per_s",
        "expert_cache_hit_rate",
        "coactivation_graph_drift",
        "speculative_acceptance_rate",
    ):
        assert key in mirror_payload["metrics"]


# ------------------------------------------------------------- nebius_ctl.sh dry-run


def _run_ctl(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "NEBIUS_INSTANCE_ID": "nebius-test-instance",
        "NEBIUS_SSH_HOST": "127.0.0.1",
        "NEBIUS_S3_BUCKET": "tenet-v3-artifacts",
        "MLFLOW_TRACKING_URI": "",
        "DRY_RUN": "0",
    }
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "remote" / "nebius_ctl.sh"), "--dry-run", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )


@pytest.mark.parametrize("subcmd", ["preflight", "start", "stop", "s3-sync-up", "s3-sync-down"])
def test_nebius_ctl_dry_run_exit_zero(tmp_path, subcmd):
    proc = _run_ctl([subcmd], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "[nebius_ctl]" in proc.stdout
    assert "DRY-RUN" in proc.stdout


def test_nebius_ctl_dry_run_telemetry_snapshot(tmp_path):
    output = tmp_path / "telemetry.json"
    proc = _run_ctl(
        ["telemetry-snapshot", "--run-id", "test_run", "--output", str(output)], tmp_path
    )
    assert proc.returncode == 0, proc.stderr
    assert output.is_file()
    passport = json.loads(output.read_text(encoding="utf-8"))
    assert passport["run_id"] == "test_run"
    for key in (
        "gpu_model",
        "nvme_read_mb_s",
        "s3_upload_gb_s",
        "instance_uptime_min",
        "estimated_cost_usd",
    ):
        assert key in passport
    assert passport["dry_run"] is True
    assert passport["provider"] == "nebius"
