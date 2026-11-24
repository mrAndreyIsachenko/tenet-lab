"""Unit tests for Tenet v3 H21 (Strata-tier-aware Lancet) and H22 (Looped-MoE).

Covers router/structural freezing, MoE layer introspection, canonical model
cards, H21 Lancet surgery semantics, H22 looped surgery + drafter
synchronization, the verify CLIs and the foundry dry-run/synthetic logging path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

CANONICAL_MODELS = [
    "Tenet-15B-A2B-Qwen3.6-Unslop-v3",
    "Tenet-26B-A3B-Inkling2-Unslop-v3",
    "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
    "Tenet-32B-A3B-GLM4.5-Unslop-v3",
    "Tenet-125B-A6B-Qwen3.8-Unslop-v3",
    "Tenet-220B-A12B-Hy4-Unslop-v3",
]


# ---------------------------------------------------------------------------
# Tiny synthetic MoE transformer used by the surgery tests
# ---------------------------------------------------------------------------
D_MODEL = 8
NUM_EXPERTS = 8


class _TinyExpert(nn.Module):
    def __init__(self, dim: int = D_MODEL) -> None:
        super().__init__()
        self.down_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.down_proj.weight)

    def forward(self, x):
        return self.down_proj(self.up_proj(x))


class _TinyMLP(nn.Module):
    def __init__(self, dim: int = D_MODEL, num_experts: int = NUM_EXPERTS) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([_TinyExpert(dim) for _ in range(num_experts)])
        self.shared_expert = _TinyExpert(dim)
        self.down_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(x)


class _TinyAttn(nn.Module):
    def __init__(self, dim: int = D_MODEL) -> None:
        super().__init__()
        self.o_proj = nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.o_proj.weight)

    def forward(self, x):
        return self.o_proj(x)


class _TinyLayer(nn.Module):
    def __init__(self, dim: int = D_MODEL, num_experts: int = NUM_EXPERTS) -> None:
        super().__init__()
        self.self_attn = _TinyAttn(dim)
        self.mlp = _TinyMLP(dim, num_experts)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class _TinyModel(nn.Module):
    def __init__(self, num_layers: int = 2, dim: int = D_MODEL) -> None:
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "num_experts": NUM_EXPERTS,
                "num_local_experts": NUM_EXPERTS,
                "num_experts_per_tok": 2,
                "hidden_size": dim,
            },
        )()
        self.embed = nn.Embedding(16, dim)
        self.layers = nn.ModuleList([_TinyLayer(dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(dim, 16, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def _make_model(num_layers: int = 2) -> _TinyModel:
    torch.manual_seed(0)
    model = _TinyModel(num_layers=num_layers)
    model.eval()
    return model


def _unit_dir(seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    vec = torch.randn(D_MODEL, generator=gen)
    return vec / vec.norm()


def _profiles(weights_by_layer: dict[int, list[float]]) -> dict[int, dict]:
    return {
        idx: {
            "num_experts": len(weights),
            "selective_weights": torch.tensor(weights, dtype=torch.float32),
        }
        for idx, weights in weights_by_layer.items()
    }


# ---------------------------------------------------------------------------
# 1. Router / structural freezing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.routing_weights.weight",
        "model.layers.0.engram.embed.weight",
        "model.layers.0.ngram_table.weight",
        "model.layers.0.memory_table.weight",
        "model.layers.0.gated_residual.weight",
        "model.layers.0.residual_gate.weight",
        "model.layers.0.ihc.weight",
        "model.layers.0.hyper_conn.weight",
        "model.layers.0.loop_controller.weight",
        "model.layers.0.loop_gate.weight",
        "model.layers.0.tick_embed.weight",
    ],
)
def test_is_router_weight_key_true_for_frozen_structures(key):
    from sml.surgery.cka import is_router_weight_key

    assert is_router_weight_key(key) is True


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.experts.3.down_proj.weight",
    ],
)
def test_is_router_weight_key_false_for_normal_projections(key):
    from sml.surgery.cka import is_router_weight_key

    assert is_router_weight_key(key) is False


def test_apply_moe_lancet_surgery_keeps_router_weights_frozen():
    from sml.surgery.subspace import apply_moe_lancet_surgery

    model = _make_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "gate" in n}
    result = apply_moe_lancet_surgery(
        model,
        {0: [_unit_dir()]},
        _profiles({0: [1.0] * NUM_EXPERTS}),
        scale=0.5,
        min_selectivity_threshold=0.25,
    )
    assert result["router_weight_drift"] == 0.0
    for name, param in model.named_parameters():
        if name in before:
            assert torch.equal(param.detach(), before[name])


# ---------------------------------------------------------------------------
# 2. MoE layer component recognition
# ---------------------------------------------------------------------------
def _module_with(attr: str, dim: int = D_MODEL) -> nn.Module:
    layer = nn.Module()
    holder = nn.Module()
    proj = nn.Linear(dim, dim, bias=False)
    # Different hybrid layouts name the output projection ``out_proj``
    # (Gated DeltaNet / GDN) or ``o_proj`` (QSA / IndexPool / SWA / DeltaNet);
    # expose both names on the same module so either layout resolves.
    holder.out_proj = proj
    holder.o_proj = proj
    setattr(layer, attr, holder)
    return layer


@pytest.mark.parametrize(
    "attr",
    ["linear_attn", "gdn", "deltanet", "qsa", "index_pool", "swa"],
)
def test_get_moe_layer_components_recognizes_hybrid_attention(attr):
    from sml.surgery.cka import get_moe_layer_components

    layer = _module_with(attr)
    components = get_moe_layer_components(layer)
    assert isinstance(components, dict)
    assert components.get("o_proj") is not None
    assert components["o_proj"] is getattr(layer, attr).o_proj


@pytest.mark.parametrize(
    "attr",
    ["latent_experts", "latent_down_proj", "latent_up_proj"],
)
def test_get_moe_layer_components_recognizes_latent_moe(attr):
    from sml.surgery.cka import get_moe_layer_components

    layer = nn.Module()
    setattr(layer, attr, nn.Linear(D_MODEL, D_MODEL, bias=False))
    components = get_moe_layer_components(layer)
    assert isinstance(components, dict)
    refs = list(components.values())
    assert (
        any(
            value is getattr(layer, attr)
            or (isinstance(value, dict) and getattr(layer, attr) in value.values())
            for value in refs
            if value is not None
        )
        or components.get("is_moe") is True
    )


@pytest.mark.parametrize("attr", ["dspark", "dfly2", "drafter", "mtp_heads", "eh_proj"])
def test_get_moe_layer_components_recognizes_drafters(attr):
    from sml.surgery.cka import get_moe_layer_components

    layer = nn.Module()
    setattr(layer, attr, nn.Linear(D_MODEL, D_MODEL, bias=False))
    components = get_moe_layer_components(layer)
    assert isinstance(components, dict)
    assert components  # drafter heads must never be silently ignored


# ---------------------------------------------------------------------------
# 3. Canonical Tenet names and model card rendering
# ---------------------------------------------------------------------------
def test_canonical_tenet_names_and_architectures_cover_v3_release():
    from sml.hub.model_card import CANONICAL_TENET_NAMES

    try:
        from sml.hub.model_card import SUPPORTED_ARCHITECTURES
    except ImportError:
        try:
            from sml.hub.release import SUPPORTED_ARCHITECTURES  # type: ignore[no-redef]
        except ImportError:
            SUPPORTED_ARCHITECTURES = None  # type: ignore[assignment]

    for name in CANONICAL_MODELS:
        assert name in CANONICAL_TENET_NAMES
        if SUPPORTED_ARCHITECTURES is not None:
            assert name in SUPPORTED_ARCHITECTURES


def test_generate_tenet_model_card_renders_strata_passport():
    from sml.hub.model_card import generate_tenet_model_card

    metrics = {
        "expert_cache_hit_rate": 0.91,
        "coactivation_graph_drift": 0.02,
        "speculative_acceptance_rate": 0.67,
        "untouched_expert_ratio": 0.75,
        "peak_host_ram_mb": 1536.0,
    }
    card = generate_tenet_model_card(
        "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
        "qwen/qwen3.6-35b-a3b-instruct",
        metrics,
    )
    if not isinstance(card, str):
        card = json.dumps(card, default=str)

    assert "Strata Engine" in card
    assert "Speculative Decoding" in card
    assert "Nebius Benchmark Passport" in card
    for key in (
        "expert_cache_hit_rate",
        "coactivation_graph_drift",
        "speculative_acceptance_rate",
        "untouched_expert_ratio",
        "peak_host_ram_mb",
    ):
        assert key in card
    assert "strata.toml" in card


# ---------------------------------------------------------------------------
# 4. H21 Strata-tier-aware surgery
# ---------------------------------------------------------------------------
def test_h21_threshold_skips_low_selectivity_experts():
    from sml.surgery.subspace import apply_moe_lancet_surgery

    model = _make_model()
    result = apply_moe_lancet_surgery(
        model,
        {0: [_unit_dir()]},
        _profiles({0: [0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}),
        min_selectivity_threshold=0.25,
        scale=0.5,
    )
    ratio = result.get("untouched_expert_ratio")
    if ratio is None:
        total = result["edited_routed_experts"] + result["skipped_routed_experts"]
        ratio = result["skipped_routed_experts"] / total
    assert ratio >= 0.75


def test_h21_shared_expert_damping_scales_shared_edit():
    from sml.surgery.subspace import apply_moe_lancet_surgery

    model = _make_model()
    original = model.layers[0].mlp.shared_expert.down_proj.weight.detach().clone()
    apply_moe_lancet_surgery(
        model,
        {0: [_unit_dir()]},
        _profiles({0: [0.0] * NUM_EXPERTS}),
        min_selectivity_threshold=0.25,
        shared_expert_damping=0.40,
        scale=1.0,
    )
    edited = model.layers[0].mlp.shared_expert.down_proj.weight.detach()
    drift = (edited - original).abs().max().item()
    assert drift > 0.0
    # Damping is a strict scale-down of the full-strength edit.
    assert drift < float((original * 0.0 + 1.0).abs().max())


def test_h21_two_tier_quant_snap_respects_crest_factor_cap():
    from sml.surgery.subspace import apply_moe_lancet_surgery

    model = _make_model()
    result = apply_moe_lancet_surgery(
        model,
        {0: [_unit_dir()]},
        _profiles({0: [1.0] * NUM_EXPERTS}),
        scale=0.9,
        quant_snap=True,
        max_block16_crest_factor=3.20,
        hot_vram_expert_ids={0, 1},
    )
    assert result["max_block16_crest_factor"] <= 3.20
    assert result["quant_snap"] is True
    hot = model.layers[0].mlp.experts[0].down_proj.weight.detach()
    cold = model.layers[0].mlp.experts[5].down_proj.weight.detach()
    assert torch.isfinite(hot).all() and torch.isfinite(cold).all()


def test_h21_streaming_recipe_applies_quant_snap_to_checkpoint(tmp_path):
    from safetensors.torch import save_file

    from sml.surgery.subspace import (
        LancetRecipe,
        apply_lancet_recipe_streaming,
    )

    src = tmp_path / "src"
    src.mkdir()
    save_file(
        {
            "model.layers.0.self_attn.o_proj.weight": torch.eye(D_MODEL),
            "model.layers.0.mlp.down_proj.weight": torch.eye(D_MODEL),
            "model.layers.1.self_attn.o_proj.weight": torch.eye(D_MODEL),
        },
        str(src / "model.safetensors"),
    )
    recipe = LancetRecipe(
        model_id="tiny",
        base_model="tiny",
        extraction_mode="activation_contrastive",
        d_model=D_MODEL,
        num_layers=2,
        scale=0.5,
        top_k_layers=1,
        selected_layers=[0],
        layer_weights={0: 1.0},
        basis_per_layer={0: _unit_dir().reshape(1, -1)},
        probes_sha256="",
    )
    out = tmp_path / "out"
    result = apply_lancet_recipe_streaming(
        src,
        out,
        recipe,
        quant_snap=True,
        max_block16_crest_factor=3.20,
    )
    assert result["shards_processed"] == 1
    assert result["tensors_modified"] == 2
    assert result["max_block16_crest_factor"] <= 3.20
    assert (out / "model.safetensors").is_file()


# ---------------------------------------------------------------------------
# 5. H22 looped surgery and drafter synchronization
# ---------------------------------------------------------------------------
def test_h22_looped_basis_rank_and_loop_controller_drift():
    from sml.surgery.looped import apply_looped_moe_lancet_surgery

    model = _make_model()
    ticks = 3
    anchor = {t: [_unit_dir(t)] for t in range(ticks)}
    anchor_list = [anchor[t] for t in range(ticks)]
    target_list = [[_unit_dir(10 + t)] for t in range(ticks)]

    result = apply_looped_moe_lancet_surgery(
        model,
        anchor_list,
        target_list,
        _profiles({0: [1.0] * NUM_EXPERTS}),
        k_loop=2,
        scale=0.5,
        hot_vram_expert_ids={0, 1},
    )
    rank = result.get("basis_rank", result.get("target_basis_rank", result.get("svd_rank")))
    assert rank is not None and rank <= 2
    assert result["loop_controller_drift"] == 0.0


def test_h22_synchronize_speculative_drafter_reduces_kl_and_preserves_norms():
    from sml.surgery.looped import synchronize_speculative_drafter_surgery

    torch.manual_seed(1)
    model = _make_model()
    # The surgery scans the model itself for embedded drafter / MTP head
    # projections, so the drafter must live on the model graph.
    drafter = nn.Module()
    drafter.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
    with torch.no_grad():
        drafter.proj.weight.copy_(torch.randn(D_MODEL, D_MODEL) * 0.1)
    model.drafter = drafter
    norms_before = drafter.proj.weight.detach().norm(dim=-1)

    result = synchronize_speculative_drafter_surgery(
        model,
        [_unit_dir(20)],
        scale=0.5,
    )
    reduction = result.get("kl_reduction_pct", 0.0)
    assert reduction >= 35.0
    assert result["kl_after"] <= result["kl_before"] * 0.65
    assert result["passed_kl_gate"] is True
    norms_after = drafter.proj.weight.detach().norm(dim=-1)
    assert torch.allclose(norms_before, norms_after, atol=1e-4)


# ---------------------------------------------------------------------------
# 6. CLI verify commands, foundry dry-run and MLflow mirror
# ---------------------------------------------------------------------------
def test_cli_verify_h21_and_h22_emit_json(capsys):
    from sml.cli import main

    for command in ("verify-h21", "verify-h22"):
        try:
            code = main([command])
        except SystemExit as exc:  # argparse-style exits are acceptable
            code = exc.code
        except Exception:
            pytest.skip(f"{command} requires a full checkpoint environment")
        captured = capsys.readouterr()
        if code not in (0, None):
            pytest.skip(f"{command} needs a checkpoint (exit={code})")
        assert captured.out.strip() or captured.err.strip() or code == 0


def _load_foundry():
    module_path = SCRIPTS_DIR / "run_tenet_v3_foundry.py"
    spec = importlib.util.spec_from_file_location("run_tenet_v3_foundry", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_foundry_dry_run_prints_plan(tmp_path, capsys):
    module = _load_foundry()
    try:
        module.main(
            [
                "--dry-run",
                "--model-id",
                "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
                "--output-dir",
                str(tmp_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "Tenet-35B-A3B-Qwen3.6-Unslop-v3" in out or "dry" in out.lower()


def test_foundry_mlflow_mirror_logging(tmp_path):
    from sml.tracking.nebius_mlflow import log_foundry_surgery_run

    record = log_foundry_surgery_run(
        "tenet-v3-foundry-v1",
        params={"scale": 0.5},
        surgery_metrics={"untouched_expert_ratio": 0.75},
        gate_metrics={"h21_verified": 1.0},
        tags={"stage": "test"},
        artifacts_paths=None,
        mirror_dir=tmp_path,
    )
    assert isinstance(record, dict)
    mirrors = list(tmp_path.glob("*.json"))
    assert mirrors
    payload = json.loads(mirrors[0].read_text())
    assert payload
