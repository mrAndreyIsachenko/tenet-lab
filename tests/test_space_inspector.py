"""Tests for the FP8 Abliteration Inspector space.

Covers ``spaces/fp8_abliteration_inspector/app.py``:

* the real Tenet-3B release bundle passes the full FP8 audit,
* a synthetic checkpoint whose rows collapse onto the injected refusal
  direction (row crest factor ~32) fails it,
* the demo preset / SVG / GGUF command generator helpers stay usable,
* the Space README frontmatter pins the Gradio SDK and every Tenet model,
* ``gpu_decorator`` and the Tenet v1 + v2 ``TENET_PROFILES`` registry exist.
"""

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "spaces" / "fp8_abliteration_inspector" / "app.py"
TENET_RELEASE_DIR = REPO_ROOT / "results" / "tenet_releases" / "Tenet-3B-Qwen2.5-Unslop-v1"
README_PATH = REPO_ROOT / "spaces" / "fp8_abliteration_inspector" / "README.md"

# Import the Space module straight from its file so the tests work cleanly
# under pytest without requiring ``spaces.*`` to be an importable package.
_spec = importlib.util.spec_from_file_location("fp8_abliteration_inspector_app", APP_PATH)
assert _spec is not None and _spec.loader is not None, f"cannot load {APP_PATH}"
app = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = app
_spec.loader.exec_module(app)

# The 7 verify_provider_readiness checks mirrored from src/sml/hub/validator.py.
SEVEN_READINESS_CHECKS = (
    "standard_architecture",
    "no_trust_remote_code",
    "valid_chat_template",
    "no_nan_inf",
    "no_unmerged_lora",
    "weight_outliers_ok",
    "acceptable_row_crest_factor",
)


def test_inspect_tenet_release_bundle() -> None:
    """The real Tenet-3B release bundle must pass the full FP8 audit."""
    result = app.inspect_directory_or_bundle(TENET_RELEASE_DIR)

    assert result["ready"] is True
    assert result["fp8_verdict"] == "PASS"

    # All 7 readiness checks plus recipe provenance must be green.
    for name in SEVEN_READINESS_CHECKS:
        assert result["checks"][name] is True, f"readiness check failed: {name}"
    assert result["checks"]["recipe_provenance"] is True
    provenance = result["recipe_provenance"]
    assert provenance is not None and provenance["ok"] is True

    # Row crest factor stays inside the FP8 budget (25.0) with sane percentiles.
    assert result["max_row_crest_factor"] is not None
    assert result["max_row_crest_factor"] < 25.0
    assert (
        result["row_crest_factor_p50"]
        <= result["row_crest_factor_p99"]
        <= result["max_row_crest_factor"]
    )

    assert len(result["top_3_outlier_layers"]) == 3

    # The inline chart is a well-formed SVG document.
    root = ET.fromstring(result["svg_chart"])
    assert root.tag.endswith("svg")


def test_inspect_synthetic_fail_crest_factor(tmp_path: Path) -> None:
    """A row dominated by one injected direction (crest ~32) must fail FP8."""
    from safetensors.numpy import save_file

    w = np.full((4, 1024), 0.001, dtype=np.float32)
    w[0, 0] = 40.0  # row 0 collapses onto the refusal direction → crest ≈ 32.0
    save_file({"model.layers.3.mlp.down_proj.weight": w}, str(tmp_path / "model.safetensors"))

    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "trust_remote_code": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tokenizer_config = {
        "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")

    result = app.inspect_directory_or_bundle(tmp_path)

    assert result["ready"] is False
    assert result["fp8_verdict"] == "FAIL"
    assert result["checks"]["acceptable_row_crest_factor"] is False
    assert result["max_row_crest_factor"] is not None
    assert result["max_row_crest_factor"] >= 25.0
    assert result["max_row_crest_factor"] == pytest.approx(32.0, rel=0.02)
    assert len(result["top_3_outlier_layers"]) >= 1


def test_demo_preset_and_command_generator() -> None:
    """DEMO_PRESET carries the naive-vs-lancet data; SVG and CLI helpers work."""
    preset = app.DEMO_PRESET
    naive, tenet = preset["naive"], preset["tenet"]
    assert naive["fp8_verdict"] == "FAIL"
    assert tenet["fp8_verdict"] == "PASS"
    assert naive["max_row_crest_factor"] >= 25.0
    assert 0.0 < tenet["max_row_crest_factor"] < 25.0
    assert naive["fp8_retention"] is None
    assert tenet["fp8_retention"] is not None
    assert naive["label"] != tenet["label"]

    root = ET.fromstring(app.render_demo_svg())
    assert root.tag.endswith("svg")

    script = app.generate_gguf_commands(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        bundle_dir=str(TENET_RELEASE_DIR),
        method="lancet",
        gguf_quant="Q4_K_M",
    )
    assert script.strip()
    assert script.startswith("#!/usr/bin/env bash")
    assert "llama-quantize" in script
    assert len([line for line in script.splitlines() if line.strip()]) > 5


# Tenet model families advertised by the Space README frontmatter.
TENET_V1_MODEL_IDS = (
    "mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1",
    "mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1",
    "mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1",
    "mr-andrey-isachenko/Tenet-32B-Qwen2.5-Unslop-v1",
    "mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1",
)

TENET_V2_MODEL_IDS = (
    "mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2",
    "mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2",
    "mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2",
    "mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2",
)

TENET_V2_PROFILE_KEYS = ("4b-v2", "8b-v2", "14b-v2", "32b-v2")


def test_readme_frontmatter_declares_gradio_sdk_and_tenet_models() -> None:
    """The Space README pins the Gradio SDK and lists every Tenet v1/v2 model."""
    text = README_PATH.read_text(encoding="utf-8")
    assert text.startswith("---"), "README must keep its YAML frontmatter block"
    frontmatter = text.split("---", 2)[1]

    assert "sdk: gradio" in frontmatter
    assert "sdk_version: 5.9.1" in frontmatter
    assert "app_file: app.py" in frontmatter

    for model_id in TENET_V1_MODEL_IDS + TENET_V2_MODEL_IDS:
        assert f"- {model_id}" in frontmatter, f"missing model in frontmatter: {model_id}"


def test_app_defines_gpu_decorator_and_tenet_v2_profiles() -> None:
    """``gpu_decorator`` stays callable and TENET_PROFILES covers Tenet v1 + v2."""
    gpu_decorator = getattr(app, "gpu_decorator", None)
    assert callable(gpu_decorator), "app.py must define a callable gpu_decorator"

    profiles = app.TENET_PROFILES
    for key in TENET_V2_PROFILE_KEYS:
        assert key in profiles, f"missing Tenet v2 profile: {key}"
    # The Tenet v1 profiles must still be registered alongside the v2 ones.
    assert len(set(profiles) - set(TENET_V2_PROFILE_KEYS)) >= 3
