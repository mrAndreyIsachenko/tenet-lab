"""Unit tests for the H100 v3 suite driver and its building blocks.

Everything here runs on mocks and synthetic tensors: no network access, no
GPU, and no Hugging Face downloads.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SUITE_SCRIPT = SCRIPTS_DIR / "remote" / "run_h100_v3_suite.sh"

for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_module(path: Path, name: str):
    """Import a repo script by file path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_quant_stub_module(torch: Any) -> types.ModuleType:
    """Build a minimal ``sml.quantization`` stand-in for the H18 audit script.

    ``audit_h18_qwen3_4b.py`` resolves ``quantize_fp8_e4m3`` and
    ``compute_microblock_crest_factor`` at import time from
    ``sml.surgery.subspace``/``sml.surgery``/``sml.quant``/``sml.quantization``.
    ``sml.surgery.subspace`` provides the NVFP4 quantizer and the subspace edit
    but not these two helpers, and ``sml.quant``/``sml.quantization`` do not
    exist yet, so the synthetic test supplies faithful CPU implementations.
    """
    module = types.ModuleType("sml.quantization", "CPU test stub for H18 quant helpers.")

    def quantize_fp8_e4m3(weight: Any, per_channel: bool = False) -> Any:
        """Fake E4M3 quantization: optional per-channel rescale + fp8 cast."""
        w = weight.detach().float()
        scale = None
        if per_channel:
            scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
            w = w / scale
        try:
            w_q = w.to(torch.float8_e4m3fn).to(torch.float32)
        except (TypeError, RuntimeError, NotImplementedError):
            w_q = w  # fp8 casts unavailable: identity keeps metrics finite
        if scale is not None:
            w_q = w_q * scale
        return w_q.to(weight.dtype)

    def compute_microblock_crest_factor(weight: Any, block_size: int = 16) -> Any:
        """Per-micro-block peak-to-mean ratio along the last dimension."""
        w = weight.detach().float()
        pad = (-w.shape[-1]) % block_size
        if pad:
            w = torch.nn.functional.pad(w, (0, pad))
        blocks = w.reshape(-1, block_size)
        mean_abs = blocks.abs().mean(dim=-1).clamp_min(1e-12)
        return blocks.abs().amax(dim=-1) / mean_abs

    module.quantize_fp8_e4m3 = quantize_fp8_e4m3
    module.compute_microblock_crest_factor = compute_microblock_crest_factor
    return module


# ---------------------------------------------------------------------------
# 1. `sml run --dry-run` CLI
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("config", "hypothesis_id"),
    [
        ("experiments/h18_nvfp4_quant_snap.yaml", "H18"),
        ("experiments/h19_loop_unrolled_lancet.yaml", "H19"),
        ("experiments/h20_moe_router_aware_lancet.yaml", "H20"),
    ],
)
def test_sml_run_cli_dry_run(config: str, hypothesis_id: str, monkeypatch, capsys) -> None:
    import sml.cli as sml_cli

    monkeypatch.chdir(REPO_ROOT)
    returncode = sml_cli.main(["run", "--config", config, "--dry-run"])
    assert returncode == 0

    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert payload["config"] == config
    assert payload["hypothesis_id"].upper() == hypothesis_id
    assert payload["experiment_id"]


# ---------------------------------------------------------------------------
# 2. export_model_index.py
# ---------------------------------------------------------------------------
def test_export_model_index_mock(tmp_path: Path) -> None:
    eval_payload = {
        "release_id": "Tenet-4B-Qwen3-Unslop-v2",
        "tenet_model": "Tenet-4B-Qwen3-Unslop-v2",
        "ifeval_prompt_strict": 0.748,
        "gsm8k_cot_strict": 0.912,
        "math_500": 0.654,
    }
    eval_json = tmp_path / "Tenet-4B-Qwen3-Unslop-v2.json"
    eval_json.write_text(json.dumps(eval_payload), encoding="utf-8")
    out_yaml = tmp_path / "model_index.yaml"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "eval" / "export_model_index.py"),
            "--eval-json",
            str(eval_json),
            "--output",
            str(out_yaml),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out_yaml.exists()

    text = out_yaml.read_text(encoding="utf-8")
    assert "model-index:" in text
    assert "gsm8k_cot_strict" in text
    assert "math_500" in text


# ---------------------------------------------------------------------------
# 3. audit_h18_qwen3_4b.run_h18_audit with a mocked shard loader
# ---------------------------------------------------------------------------
def test_audit_h18_qwen3_4b_synthetic(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    audit = _load_module(
        SCRIPTS_DIR / "eval" / "audit_h18_qwen3_4b.py",
        "audit_h18_qwen3_4b_synthetic",
    )

    dim = 64
    torch.manual_seed(0)
    basis_path = tmp_path / "lancet_basis.safetensors"
    save_file(
        {"layer_9": torch.randn(dim), "layer_13": torch.randn(dim)},
        str(basis_path),
    )

    index_filename = audit.INDEX_FILENAME
    weight_map = {
        "model.layers.9.mlp.down_proj.weight": "shard-00001-of-00002.safetensors",
        "model.layers.9.self_attn.o_proj.weight": "shard-00001-of-00002.safetensors",
        "model.layers.13.mlp.down_proj.weight": "shard-00002-of-00002.safetensors",
        "model.layers.13.self_attn.o_proj.weight": "shard-00002-of-00002.safetensors",
    }
    requested: list[str] = []

    def fake_shard_loader(filename: str) -> dict[str, Any]:
        requested.append(filename)
        if filename == index_filename:
            return {"metadata": {"total_size": 1}, "weight_map": dict(weight_map)}
        return {key: torch.randn(64, 64) for key, shard in weight_map.items() if shard == filename}

    output_path = tmp_path / "h18_nvfp4_quant_snap" / "qwen3_4b_audit.json"
    result = audit.run_h18_audit(
        basis_path=basis_path,
        model_id="Qwen/Qwen3-4B",
        output_path=output_path,
        shard_loader=fake_shard_loader,
        cleanup_cache=False,
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert index_filename in requested
    assert sorted(result["layers"]) == [9, 13]
    assert result["num_layers_audited"] == 2
    assert result["config"]["shard_loader_injected"] is True

    variant_names = ("naive_ablation", "standard_lancet_h15", "h18_quant_snap_lancet")
    quant_names = ("nvfp4_b16", "int4_b32", "fp8_per_channel")
    for layer_results in result["per_layer"].values():
        for proj_results in layer_results.values():
            variants = proj_results["variants"]
            for variant_name in variant_names:
                assert variant_name in variants
                for quant_name in quant_names:
                    assert quant_name in variants[variant_name]
                    for value in variants[variant_name][quant_name].values():
                        assert math.isfinite(float(value))
            ratio = proj_results["summary_ratios"]["nvfp4_releakage_reduction_ratio"]
            assert math.isfinite(float(ratio))

    mean_metrics = result["mean_metrics"]
    for variant_name in variant_names:
        for quant_name in quant_names:
            assert math.isfinite(float(mean_metrics[f"{variant_name}.{quant_name}.releak_ratio"]))
            assert math.isfinite(float(mean_metrics[f"{variant_name}.{quant_name}.rel_l2_drift"]))
    assert math.isfinite(float(result["summary_ratios"]["nvfp4_releakage_reduction_ratio"]))


# ---------------------------------------------------------------------------
# 4. run_tenet_v3_pilot.py --dry-run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["moe", "dense"])
def test_run_tenet_v3_pilot_dry_run(target: str) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_tenet_v3_pilot.py"),
            "--target",
            target,
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    plan = json.loads(proc.stdout)
    assert plan["target"] == target
    assert plan["dry_run"] is True
    assert plan["model_id"] and plan["release_id"]
    assert "streaming_surgery_" + target in plan["steps"]
    assert plan["steps"][-1] == "write_foundry_summary"


# ---------------------------------------------------------------------------
# 5. summarize_h100_v3_suite.py over a mocked results tree
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_summarize_h100_v3_suite_mock(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_json(
        results / "external_evals" / "Tenet-4B-Qwen3-Unslop-v2.json",
        {
            "release_id": "Tenet-4B-Qwen3-Unslop-v2",
            "after": {"gsm8k_cot_strict": 0.912, "math_500": 0.654},
        },
    )
    _write_json(
        results / "h18_nvfp4_quant_snap" / "qwen3_4b_audit.json",
        {
            "model_id": "Qwen/Qwen3-4B",
            "mean_metrics": {
                "nvfp4_b16.releak_ratio": 0.12,
                "int4_b32.releak_ratio": 0.20,
                "fp8_per_channel.releak_ratio": 0.30,
            },
        },
    )
    _write_json(
        results / "h18_nvfp4_quant_snap" / "h18_summary.json",
        {"h18_nvfp4_kl_reduction_ratio": 0.31, "max_block16_crest_factor": 3.1},
    )
    _write_json(
        results / "h19_loop_unrolled_lancet" / "h19_summary.json",
        {"h19_total_collateral_kl": 0.021, "h19_effective_rank_per_layer": 2.0},
    )
    _write_json(
        results / "tenet_releases" / "Tenet-26B-Gemma4-MoE-Unslop-v3" / "foundry_summary.json",
        {
            "release_id": "Tenet-26B-Gemma4-MoE-Unslop-v3",
            "target": "moe",
            "model_id": "google/gemma-4-26b-a4b-it",
            "status": "ok",
            "quant_snap": True,
            "beta": 1.0,
            "eval_suite": "full",
        },
    )

    out_md = tmp_path / "h100_v3_suite_summary.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "eval" / "summarize_h100_v3_suite.py"),
            "--results-dir",
            str(results),
            "--output",
            str(out_md),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    text = out_md.read_text(encoding="utf-8")
    assert "## 1. Tenet v2 Full vLLM OpenEvals" in text
    assert "## 2. H18 Quant-Snap Audit" in text
    assert "## 3. H18 & H19 Standard Benchmarks" in text
    assert "## 4. Tenet v3" in text
    assert "Tenet-26B-Gemma4-MoE-Unslop-v3" in text


# ---------------------------------------------------------------------------
# 6. run_h100_v3_suite.sh syntax + dry run
# ---------------------------------------------------------------------------
def test_run_h100_v3_suite_shell_syntax_and_dry_run() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(SUITE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert syntax.returncode == 0, syntax.stderr

    proc = subprocess.run(
        [
            "bash",
            str(SUITE_SCRIPT),
            "--dry-run",
            "--steps",
            "eval_v2,h18_audit,h18_h19_bench,tenet_v3",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = proc.stdout

    dry_lines = [line for line in out.splitlines() if line.startswith("[dry-run]")]
    assert len(dry_lines) >= 10

    expected_fragments = (
        "run_vllm_open_evals.py --release-id Tenet-4B-Qwen3-Unslop-v2",
        "run_vllm_open_evals.py --release-id Tenet-8B-Qwen3-Unslop-v2",
        "run_vllm_open_evals.py --release-id Tenet-32B-Qwen3-Unslop-v2",
        "export_model_index.py --eval-json results/external_evals/Tenet-4B-Qwen3-Unslop-v2.json",
        "cleanup_hf_model_cache",
        "audit_h18_qwen3_4b.py",
        "sml run --config experiments/h18_nvfp4_quant_snap.yaml"
        " --output-dir results --run-id h18_nvfp4_quant_snap",
        "sml run --config experiments/h19_loop_unrolled_lancet.yaml"
        " --output-dir results --run-id h19_loop_unrolled_lancet --quant-snap",
        "run_tenet_v3_pilot.py --target moe --model-id google/gemma-4-26b-a4b-it",
        "run_tenet_v3_pilot.py --target dense"
        " --model-id mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "summarize_h100_v3_suite.py --results-dir results",
    )
    for fragment in expected_fragments:
        matching = [line for line in dry_lines if fragment in line]
        assert matching, f"missing dry-run line containing: {fragment}"

    assert (
        '[cleanup_hf_model_cache(m) for m in ["Qwen/Qwen3-4B", '
        '"huihui-ai/Qwen3-4B-abliterated", "mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2"]]'
    ) in out
    assert "Tenet-26B-A4B-Gemma4-Unslop-v2" in out  # moe release default
    assert "Tenet-24B-Mistral3.1-Unslop-v2" in out  # dense release default
    assert "Tenet-26B-Gemma4-MoE-Unslop-v3" not in out  # old default removed
    assert "Tenet-24B-Mistral-Small-3.1-Unslop-v3" not in out  # old default removed

    # moe keeps the thinking token; dense (Mistral non-thinking) gets ''.
    moe_lines = [line for line in dry_lines if "--target moe" in line]
    assert moe_lines, "missing moe pilot dry-run line"
    assert all("--think-end-token </think>" in line for line in moe_lines)
    dense_lines = [line for line in dry_lines if "--target dense" in line]
    assert dense_lines, "missing dense pilot dry-run line"
    assert all(
        "--think-end-token  --max-gen-toks" in line  # empty token renders as ''
        for line in dense_lines
    )
    assert "</think>" in out  # THINK_END_TOKEN default threaded through
    assert "uv sync --frozen" not in out  # preflight skipped in dry-run
    assert "idle_watchdog" not in out  # watchdog not started in dry-run


def test_run_h100_v3_suite_two_gpus_and_mmlu_pro_limit_dry_run() -> None:
    env = dict(os.environ)
    env["NUM_GPUS"] = "2"
    env["MMLU_PRO_LIMIT"] = "1000"
    proc = subprocess.run(
        ["bash", str(SUITE_SCRIPT), "--dry-run", "--steps", "eval_v2,tenet_v3"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = proc.stdout.splitlines()

    # eval_v2: three entries pinned to GPUs 0/1/0, each with the mmlu_pro limit.
    eval_lines = [line for line in lines if "run_vllm_open_evals.py" in line]
    assert len(eval_lines) == 3
    assert [line.split("CUDA_VISIBLE_DEVICES=")[1][0] for line in eval_lines] == [
        "0",
        "1",
        "0",
    ]
    for line in eval_lines:
        assert line.index("CUDA_VISIBLE_DEVICES=") < line.index("run_vllm_open_evals.py")
        assert "--mmlu-pro-limit 1000" in line

    # tenet_v3: moe pinned to GPU 0, dense pinned to GPU 1 (dry-run order).
    pilot_lines = [line for line in lines if "run_tenet_v3_pilot.py" in line]
    assert len(pilot_lines) == 2
    assert [line.split("CUDA_VISIBLE_DEVICES=")[1][0] for line in pilot_lines] == [
        "0",
        "1",
    ]
    assert "--target moe" in pilot_lines[0]
    assert "--target dense" in pilot_lines[1]
    for line in pilot_lines:
        assert line.index("CUDA_VISIBLE_DEVICES=") < line.index("run_tenet_v3_pilot.py")


def test_run_h100_v3_suite_auto_stop_pod_and_fail_fast_default_dry_run() -> None:
    env = dict(os.environ)
    env["RUNPOD_POD_ID"] = "testpod123"
    proc = subprocess.run(
        ["bash", str(SUITE_SCRIPT), "--dry-run", "--auto-stop-pod"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = proc.stdout.splitlines()

    # Default (no --steps) now runs the fail-fast order: h18_audit and
    # h18_h19_bench must both appear before eval_v2 (and tenet_v3 last).
    first_positions = {}
    for name, marker in (
        ("h18_audit", "audit_h18_qwen3_4b.py"),
        ("h18_h19_bench", "sml run --config"),
        ("eval_v2", "run_vllm_open_evals.py"),
        ("tenet_v3", "run_tenet_v3_pilot.py"),
    ):
        matches = [i for i, line in enumerate(lines) if marker in line]
        assert matches, f"missing {name} dry-run lines"
        first_positions[name] = matches[0]
    assert first_positions["h18_audit"] < first_positions["h18_h19_bench"]
    assert first_positions["h18_h19_bench"] < first_positions["eval_v2"]
    assert first_positions["eval_v2"] < first_positions["tenet_v3"]

    # --auto-stop-pod emits the (dry-run) pod stop for the given pod id.
    assert "[dry-run] runpodctl stop pod testpod123" in lines


def test_run_h100_v3_suite_rejects_unknown_step() -> None:
    proc = subprocess.run(
        ["bash", str(SUITE_SCRIPT), "--dry-run", "--steps", "eval_v2,bogus_step"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "unknown step 'bogus_step'" in proc.stderr


def test_run_h100_v3_suite_help() -> None:
    proc = subprocess.run(
        ["bash", str(SUITE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "Usage:" in proc.stdout
    assert "--steps" in proc.stdout
