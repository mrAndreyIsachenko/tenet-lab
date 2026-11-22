"""H18: NVFP4 quant-snap survival tests.

Covers the NVFP4 e2m1 quantization helpers, microblock crest-factor
regularization, quant-snap subspace projection, Lancet biprojected /
streaming surgery with quant snap, the H11 provider validator, and the
`verify-h18` CLI entry point.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sml.cli import build_parser
from sml.hub.validator import _inspect_weights, verify_provider_readiness
from sml.model.transformer import ModelConfig, Transformer
from sml.surgery.quantization import (
    _NVFP4_E2M1_LEVELS,
    compute_microblock_crest_factor,
    quantize_nvfp4_e2m1,
    regularize_microblock_crest_factor,
)
from sml.surgery.quantization import (
    evaluate_h18_nvfp4_quant_snap_survival,
)
from sml.surgery.subspace import (
    apply_lancet_biprojected_surgery,
    apply_lancet_recipe_streaming,
    LancetRecipe,
    project_and_snap_subspace_norm_preserving,
    project_out_subspace_norm_preserving,
)


def _finite_leaves(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _finite_leaves(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _finite_leaves(value)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield float(obj)


# ---------------------------------------------------------------------------
# 1. NVFP4 e2m1 quantization
# ---------------------------------------------------------------------------


def test_quantize_nvfp4_e2m1_levels_and_validation():
    levels = {float(v) for v in _NVFP4_E2M1_LEVELS}
    assert len(levels) >= 8
    assert 0.0 in levels

    torch.manual_seed(0)
    weight = torch.randn(2, 16, dtype=torch.float32)
    quantized = quantize_nvfp4_e2m1(weight, block_size=16)

    # NVFP4 dequantizes e2m1 codes against a per-block scale, so values are
    # lattice points times a positive scale factor rather than raw codes.
    assert quantized.shape == weight.shape
    assert quantized.dtype == weight.dtype
    assert torch.isfinite(quantized).all()

    flat_w = weight.flatten()
    flat_q = quantized.flatten()
    wc = flat_w - flat_w.mean()
    qc = flat_q - flat_q.mean()
    corr = float((wc * qc).sum() / (wc.norm() * qc.norm()).clamp_min(1e-12))
    assert corr > 0.95
    assert float((flat_q - flat_w).abs().max()) <= 0.5 * float(flat_w.abs().max())

    # Quantization must actually map arbitrary values onto the code lattice.
    off_lattice = torch.full((1, 16), 0.7, dtype=torch.float32)
    snapped = quantize_nvfp4_e2m1(off_lattice, block_size=16)
    assert float(snapped.flatten()[0]) != 0.7

    with pytest.raises(ValueError):
        quantize_nvfp4_e2m1(weight, block_size=0)
    with pytest.raises(ValueError):
        quantize_nvfp4_e2m1(weight, block_size=-4)


# ---------------------------------------------------------------------------
# 2. Microblock crest-factor statistics
# ---------------------------------------------------------------------------


def test_compute_microblock_crest_factor_statistics():
    torch.manual_seed(1)
    weight = torch.randn(32, 64, dtype=torch.float32)

    stats = compute_microblock_crest_factor(weight, block_size=16)
    assert isinstance(stats, dict)
    for key in (
        "max_block_crest_factor",
        "mean_block_crest_factor",
        "p50_block_crest_factor",
        "p99_block_crest_factor",
    ):
        assert key in stats
        assert math.isfinite(float(stats[key]))

    max_cf = float(stats["max_block_crest_factor"])
    p99_cf = float(stats["p99_block_crest_factor"])
    p50_cf = float(stats["p50_block_crest_factor"])
    mean_cf = float(stats["mean_block_crest_factor"])

    # Crest factor is peak/rms, which is >= 1 for any non-degenerate block.
    assert max_cf >= 1.0
    assert max_cf >= p99_cf - 1e-9
    assert p99_cf >= p50_cf - 1e-9
    assert mean_cf <= max_cf + 1e-9

    # An extreme in-block outlier inflates the worst crest factor.
    spiked = weight.clone()
    spiked[0, 0] *= 64.0
    spiked_stats = compute_microblock_crest_factor(spiked, block_size=16)
    assert float(spiked_stats["max_block_crest_factor"]) > max_cf


# ---------------------------------------------------------------------------
# 3. Microblock crest-factor regularization
# ---------------------------------------------------------------------------


def test_regularize_microblock_crest_factor_bounds_cf():
    torch.manual_seed(2)
    weight = torch.randn(16, 64, dtype=torch.float32)
    weight[:, ::16] *= 32.0  # sparse in-block spikes

    before = compute_microblock_crest_factor(weight, block_size=16)
    assert float(before["max_block_crest_factor"]) > 3.2

    regularized = regularize_microblock_crest_factor(
        weight, block_size=16, max_block_cf=3.2
    )
    assert isinstance(regularized, torch.Tensor)
    assert regularized.shape == weight.shape
    assert regularized.dtype == weight.dtype
    assert torch.isfinite(regularized).all()

    after = compute_microblock_crest_factor(regularized, block_size=16)
    after_max = float(after["max_block_crest_factor"])
    assert after_max <= float(before["max_block_crest_factor"]) + 1e-6
    # The regularizer approximately enforces the crest-factor budget.
    assert after_max <= 3.2 * 1.25


# ---------------------------------------------------------------------------
# 4. H18 quant-snap survival evaluation
# ---------------------------------------------------------------------------


def test_evaluate_h18_nvfp4_quant_snap_survival_summary():
    summary = evaluate_h18_nvfp4_quant_snap_survival(
        seed=2026,
        n_prompts=2,
        seq_len=8,
        num_iters=2,
        step_size=0.75,
        block_size=16,
        max_block_cf=3.2,
    )
    assert isinstance(summary, dict)
    assert len(summary) > 0

    # The CLI dumps this structure verbatim, so it must be JSON-serializable.
    json.dumps(summary)
    values = list(_finite_leaves(summary))
    assert values, "expected numeric metrics in the H18 summary"
    assert all(math.isfinite(value) for value in values)


# ---------------------------------------------------------------------------
# 5. Quant-snap subspace projection, norm preserving
# ---------------------------------------------------------------------------


def test_project_and_snap_subspace_norm_preserving_rows():
    torch.manual_seed(3)
    weight = torch.randn(64, 64, dtype=torch.float32)
    directions = torch.randn(4, 64, dtype=torch.float32)

    projected = project_out_subspace_norm_preserving(
        weight, directions, scale=1.0, preserve_row_norm=True
    )
    assert projected.shape == weight.shape
    assert projected.dtype == weight.dtype
    assert torch.isfinite(projected).all()

    rel_drift = (
        (projected.norm(dim=-1) - weight.norm(dim=-1)).abs()
        / weight.norm(dim=-1).clamp_min(1e-12)
    )
    assert float(rel_drift.max()) <= 1e-4

    snapped = project_and_snap_subspace_norm_preserving(
        weight,
        directions,
        scale=1.0,
        snap_mode="hybrid_nvfp4_int4",
        num_iters=4,
        step_size=0.75,
        block_size=16,
        max_block_cf=3.2,
        preserve_bf16_nullspace=False,
        preserve_row_norm=True,
        max_rel_drift=0.02,
    )
    assert snapped.shape == weight.shape
    assert snapped.dtype == weight.dtype
    assert torch.isfinite(snapped).all()

    snap_drift = (
        (snapped.norm(dim=-1) - weight.norm(dim=-1)).abs()
        / weight.norm(dim=-1).clamp_min(1e-12)
    )
    assert float(snap_drift.max()) <= 0.02 * 1.25 + 1e-9

    # The snap stage perturbs the plain projection (quantization lattice).
    assert not torch.allclose(snapped, projected)


# ---------------------------------------------------------------------------
# 6. Lancet biprojected surgery + streaming recipe with quant snap
# ---------------------------------------------------------------------------


def _build_tiny_model() -> Transformer:
    config = ModelConfig(
        name="tiny",
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=192,
        max_seq_len=32,
        rope_theta=10000.0,
    )
    torch.manual_seed(4)
    return Transformer(config)


def _o_proj_weight(model: Transformer) -> torch.Tensor:
    return model.layers[0].self_attn.o_proj.weight


def test_lancet_surgery_streaming_quant_snap_and_provider_readiness(tmp_path):
    model = _build_tiny_model()
    before = _o_proj_weight(model).detach().clone()

    torch.manual_seed(5)
    target_dirs = {0: torch.randn(2, 64, dtype=torch.float32)}
    anchor_dirs = {0: torch.randn(2, 64, dtype=torch.float32)}
    surgically = apply_lancet_biprojected_surgery(
        model,
        target_dirs,
        anchor_dirs,
        layer_indices=[0],
        quant_snap=True,
        quant_snap_iters=2,
        max_block16_crest_factor=3.2,
    )

    after = _o_proj_weight(surgically)
    assert after.shape == before.shape
    assert after.dtype == before.dtype
    for parameter in surgically.parameters():
        assert torch.isfinite(parameter).all()
    assert not torch.allclose(after, before)
    # Only the selected layer is touched; layer 1 stays bit-identical.
    assert torch.equal(
        surgically.layers[1].self_attn.o_proj.weight.detach(),
        model.layers[1].self_attn.o_proj.weight.detach(),
    )

    # Streaming recipe over a minimal Qwen2-style checkpoint.
    ckpt_dir = tmp_path / "ckpt"
    out_dir = tmp_path / "out"
    ckpt_dir.mkdir()
    out_dir.mkdir()
    save_file(
        {"layers.0.self_attn.o_proj.weight": torch.randn(64, 64)},
        str(ckpt_dir / "model.safetensors"),
    )
    (ckpt_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"})
    )
    (ckpt_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"}
        )
    )

    recipe = LancetRecipe(
        model_id="tiny",
        base_model="tiny",
        extraction_mode="random",
        d_model=64,
        num_layers=2,
        scale=1.0,
        top_k_layers=1,
        selected_layers=[0],
        layer_weights={0: 1.0},
        basis_per_layer={0: torch.eye(64, dtype=torch.float32)[:1]},
        probes_sha256="0" * 64,
    )
    result = apply_lancet_recipe_streaming(
        ckpt_dir,
        out_dir,
        recipe,
        quant_snap=True,
        quant_snap_iters=4,
        max_block16_crest_factor=3.2,
    )
    assert isinstance(result, dict)
    for key in (
        "quant_snap",
        "quant_snap_iters",
        "max_block16_crest_factor",
        "provider_ready",
        "tensors_modified",
    ):
        assert key in result
    assert result["quant_snap"] is True
    assert result["quant_snap_iters"] == 4
    assert float(result["max_block16_crest_factor"]) == pytest.approx(3.2)
    assert result["tensors_modified"] >= 1
    assert result["provider_ready"] is True

    # H11 provider validator on a minimal HF bundle.
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    save_file(
        {"model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32)},
        str(bundle_dir / "model.safetensors"),
    )
    (bundle_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"})
    )
    (bundle_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"}
        )
    )

    report = verify_provider_readiness(bundle_dir)
    assert isinstance(report, dict)
    assert "ready" in report
    assert isinstance(report["ready"], bool)
    assert "max_row_crest_factor" in report
    assert len(report["checks"]) == 7

    weight_files = sorted(bundle_dir.glob("*.safetensors"))
    errors = []
    inspected = _inspect_weights(weight_files, report["checks"], errors)
    assert "max_block16_crest_factor" in inspected
    assert math.isfinite(float(inspected["max_block16_crest_factor"]))


# ---------------------------------------------------------------------------
# 7. CLI: verify-h18
# ---------------------------------------------------------------------------


def test_cli_verify_h18_writes_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    parser = build_parser()
    args = parser.parse_args(
        [
            "verify-h18",
            "--output-dir",
            str(tmp_path),
            "--n-prompts",
            "2",
            "--seq-len",
            "8",
        ]
    )
    assert args.func(args) == 0

    summary_path = tmp_path / "h18_nvfp4_quant_snap" / "h18_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert isinstance(summary, dict)
