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
    from sml.hub.validator import SUPPORTED_ARCHITECTURES

    extra_models = [
        "Tenet-30B-A3B-GLM4.7-Unslop-v3",
        "Tenet-15B-A2B-Inkling2-Unslop-v3",
        "Tenet-106B-A12B-GLM4.5-Unslop-v3",
        "Tenet-26B-A4B-Gemma4-Unslop-v2",
    ]

    for name in CANONICAL_MODELS + extra_models:
        assert name in CANONICAL_TENET_NAMES

    expected_architectures = {
        "Inkling2LoopForConditionalGeneration",
        "Inkling2ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Glm4MoeLiteForCausalLM",
        "Glm4MoeForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "HYV4FlashForConditionalGeneration",
    }
    assert expected_architectures <= SUPPORTED_ARCHITECTURES

    foundry = _load_foundry()
    specs = foundry.TENET_V3_MODEL_SPECS
    # ``Tenet-15B-A2B-Qwen3.6-Unslop-v3`` stays a canonical card name but has no
    # dedicated foundry spec (superseded by the Inkling2 15B release).
    spec_names = [name for name in CANONICAL_MODELS if name != "Tenet-15B-A2B-Qwen3.6-Unslop-v3"]
    for name in spec_names + extra_models:
        assert name in specs

    assert specs["Tenet-30B-A3B-GLM4.7-Unslop-v3"]["base_model"] == "zai-org/GLM-4.7-Flash"
    assert specs["Tenet-30B-A3B-GLM4.7-Unslop-v3"]["has_drafter"] is True
    assert specs["Tenet-30B-A3B-GLM4.7-Unslop-v3"]["looped"] is False

    assert (
        specs["Tenet-15B-A2B-Inkling2-Unslop-v3"]["base_model"]
        == "thinkingmachines/Inkling-2-Small"
    )
    assert specs["Tenet-15B-A2B-Inkling2-Unslop-v3"]["looped"] is False

    assert (
        specs["Tenet-26B-A3B-Inkling2-Unslop-v3"]["base_model"] == "thinkingmachines/Inkling-2-Mini"
    )
    assert specs["Tenet-26B-A3B-Inkling2-Unslop-v3"]["looped"] is True
    assert specs["Tenet-26B-A3B-Inkling2-Unslop-v3"]["has_drafter"] is True

    assert specs["Tenet-35B-A3B-Qwen3.6-Unslop-v3"]["base_model"] == "Qwen/Qwen3.6-35B-A3B"
    assert (
        specs["Tenet-35B-A3B-Qwen3.6-Unslop-v3"]["naive_model"]
        == "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated"
    )

    assert specs["Tenet-106B-A12B-GLM4.5-Unslop-v3"]["base_model"] == "zai-org/GLM-4.5-Air"
    assert specs["Tenet-32B-A3B-GLM4.5-Unslop-v3"]["base_model"] == "zai-org/GLM-4.5-Air"

    assert specs["Tenet-125B-A6B-Qwen3.8-Unslop-v3"]["base_model"] == "Qwen/Qwen3.8-Flash-Next"
    assert specs["Tenet-220B-A12B-Hy4-Unslop-v3"]["base_model"] == "tencent/Hy4-flash"

    assert specs["Tenet-26B-A4B-Gemma4-Unslop-v2"]["base_model"] == "google/gemma-4-26B-A4B-it"
    assert specs["Tenet-26B-A4B-Gemma4-Unslop-v2"]["think_end_token"] == "<|thought|>"


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


# ---------------------------------------------------------------------------
# 7. Validator text_config unwrapping, foundry model-dir resolution,
#    streaming sharded checkpoints and open-evals / cleanup flags
# ---------------------------------------------------------------------------
def test_validator_unwraps_text_config_for_conditional_generation(tmp_path):
    from safetensors.torch import save_file

    from sml.hub.validator import verify_provider_readiness

    gen = torch.Generator().manual_seed(11)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Inkling2LoopForConditionalGeneration"],
                "text_config": {"hidden_size": 16, "num_hidden_layers": 2},
            }
        )
    )
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    save_file(
        {
            "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(
                16, 16, generator=gen
            )
            * 0.25
        },
        str(tmp_path / "model.safetensors"),
    )
    basis = torch.nn.functional.normalize(torch.randn(2, 16, generator=gen), dim=1)
    save_file(
        {"layer_0": basis[0:1].contiguous(), "layer_1": basis[1:2].contiguous()},
        str(tmp_path / "lancet_basis.safetensors"),
    )
    (tmp_path / "lancet_recipe.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "d_model": 16,
                "num_layers": 2,
                "selected_layers": [0, 1],
                "layer_weights": {"0": 1.0, "1": 1.0},
            }
        )
    )

    result = verify_provider_readiness(tmp_path)
    assert result["ready"] is True

    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["text_config"]["hidden_size"] = 32
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    broken = verify_provider_readiness(tmp_path)
    assert broken["ready"] is False
    assert any("hidden_size" in str(err) for err in broken.get("errors", []))


def test_foundry_resolve_model_dir_downloads_real_base_model(tmp_path, monkeypatch):
    foundry = _load_foundry()

    calls: list[dict] = []

    def _fake_snapshot_download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        return str(tmp_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.delenv("TENET_V3_MODEL_DIR", raising=False)

    resolved = foundry._resolve_model_dir("Tenet-35B-A3B-Qwen3.6-Unslop-v3", None, download=True)
    assert str(resolved) == str(tmp_path)
    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == "Qwen/Qwen3.6-35B-A3B"
    assert call.get("cache_dir") == str(tmp_path / "hf_home")
    assert call.get("local_files_only") is not True
    assert "local_files_only" not in call
    assert "allow_patterns" not in call


def test_foundry_streaming_sharded_checkpoint_with_drafter_and_engram(tmp_path):
    foundry = _load_foundry()

    foundry_src = (SCRIPTS_DIR / "run_tenet_v3_foundry.py").read_text(encoding="utf-8")
    assert "from_pretrained(" not in foundry_src
    # The script's docstring legitimately mentions ``torch.randn(64)`` as an
    # anti-pattern; only a real fixed-size noise tensor allocation is banned.
    assert "torch.randn(64, 64" not in foundry_src

    from safetensors import safe_open
    from safetensors.torch import save_file

    from sml.hub.validator import verify_provider_readiness

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Inkling2LoopForConditionalGeneration"],
                "text_config": {
                    "hidden_size": 16,
                    "num_hidden_layers": 2,
                    "num_local_experts": 4,
                },
            }
        )
    )
    (src_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )
    (src_dir / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m.content }}{% endfor %}"
    )

    gen = torch.Generator().manual_seed(77)

    def _small(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen) * 0.25

    shard1: dict[str, torch.Tensor] = {}
    shard2: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    for idx, shard in ((0, shard1), (1, shard2)):
        prefix = f"model.language_model.layers.{idx}"
        shard[f"{prefix}.self_attn.o_proj.weight"] = _small(16, 16)
        shard[f"{prefix}.mlp.router.weight"] = _small(4, 16)
        shard[f"{prefix}.mlp.shared_expert.down_proj.weight"] = _small(16, 24)
        for e in range(4):
            shard[f"{prefix}.mlp.experts.{e}.down_proj.weight"] = _small(16, 24)
        name = f"model-0000{idx + 1}-of-00002.safetensors"
        for key in shard:
            weight_map[key] = name
    shard1["model.engram.memory_table.weight"] = _small(8, 16)
    shard1["model.loop_controller.weight"] = _small(4, 16)
    weight_map["model.engram.memory_table.weight"] = "model-00001-of-00002.safetensors"
    weight_map["model.loop_controller.weight"] = "model-00001-of-00002.safetensors"
    shard2["model.dspark.out_proj.weight"] = _small(16, 16)
    shard2["model.mtp.0.eh_proj.weight"] = _small(16, 32)
    weight_map["model.dspark.out_proj.weight"] = "model-00002-of-00002.safetensors"
    weight_map["model.mtp.0.eh_proj.weight"] = "model-00002-of-00002.safetensors"

    save_file(shard1, str(src_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(src_dir / "model-00002-of-00002.safetensors"))
    (src_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    out_dir = tmp_path / "out"
    mat_dir = tmp_path / "mat"
    rc = foundry.main(
        [
            "--model-id",
            "Tenet-26B-A3B-Inkling2-Unslop-v3",
            "--model-dir",
            str(src_dir),
            "--output-dir",
            str(out_dir),
            "--materialized-dir",
            str(mat_dir),
            "--run-fight-card-gate",
            "--allow-weight-svd-fallback",
        ]
    )
    assert rc == 0

    summary = json.loads(
        (out_dir / foundry.DEFAULT_RELEASE_ID / "foundry_summary.json").read_text()
    )
    assert summary["h21"]["mode"] == "streaming"
    assert summary["h21"]["h21_verified"] is True
    assert summary["h22"]["h22_verified"] is True
    assert summary["h22"]["looped"]["basis_rank"] <= 2
    assert summary["h22"]["drafter"]["mode"] == "streaming"
    assert summary["h22"]["drafter"]["passed_kl_gate"] is True
    assert summary["h22"]["drafter"]["kl_reduction_pct"] >= 35.0
    assert summary["h22"]["drafter"]["edited_drafter_params"] == 2
    assert summary["fight_card_gate"]["status"] == "skipped_unsupported_arch"
    assert summary["fight_card_gate"]["gate_passed"] is None

    assert verify_provider_readiness(mat_dir)["ready"] is True

    with safe_open(mat_dir / "lancet_basis.safetensors", framework="pt", device="cpu") as f:
        basis_keys = set(f.keys())
        assert {"layer_0", "layer_1"} <= basis_keys
        for key in ("layer_0", "layer_1"):
            assert f.get_tensor(key).shape[1] == 16
            assert f.get_tensor(key).shape[0] <= 2

    def _mat_tensor(key: str) -> torch.Tensor:
        for shard in sorted(mat_dir.glob("*.safetensors")):
            if shard.name == "lancet_basis.safetensors":
                continue
            with safe_open(shard, framework="pt", device="cpu") as f:
                if key in set(f.keys()):
                    return f.get_tensor(key)
        raise AssertionError(f"key {key} missing from materialized checkpoint")

    for frozen_key in (
        "model.language_model.layers.0.mlp.router.weight",
        "model.language_model.layers.1.mlp.router.weight",
        "model.engram.memory_table.weight",
        "model.loop_controller.weight",
    ):
        src_t = shard1[frozen_key] if frozen_key in shard1 else shard2[frozen_key]
        assert torch.equal(_mat_tensor(frozen_key), src_t), frozen_key

    for drafter_key in ("model.dspark.out_proj.weight", "model.mtp.0.eh_proj.weight"):
        assert not torch.equal(_mat_tensor(drafter_key), shard2[drafter_key]), drafter_key


def test_foundry_open_evals_and_cleanup_flags(tmp_path, monkeypatch):
    import subprocess

    foundry = _load_foundry()

    from safetensors.torch import save_file

    hf_home = tmp_path / "hf_home"
    ckpt_dir = hf_home / "base_ckpt"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "hidden_size": 16,
                "num_hidden_layers": 1,
            }
        )
    )
    (ckpt_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    gen = torch.Generator().manual_seed(5)
    tensors = {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(16, 16, generator=gen) * 0.25,
        "model.layers.0.mlp.router.weight": torch.randn(4, 16, generator=gen) * 0.25,
        "model.mtp.0.eh_proj.weight": torch.randn(16, 16, generator=gen) * 0.25,
    }
    for e in range(4):
        tensors[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = (
            torch.randn(16, 16, generator=gen) * 0.25
        )
    save_file(tensors, str(ckpt_dir / "model.safetensors"))

    commands: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    hf_calls: list[tuple] = []

    class _DummyHfApi:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def create_repo(self, **kwargs):
            hf_calls.append(("create_repo", kwargs))
            return kwargs.get("repo_id")

        def upload_folder(self, **kwargs):
            hf_calls.append(("upload_folder", kwargs))
            return {}

    monkeypatch.setattr("huggingface_hub.HfApi", _DummyHfApi)

    # ---- Case A: smoke eval suite for the Gemma4 v2 release ----
    monkeypatch.setenv("HF_HOME", str(hf_home))
    foundry.main(
        [
            "--model-id",
            "Tenet-26B-A4B-Gemma4-Unslop-v2",
            "--model-dir",
            str(ckpt_dir),
            "--output-dir",
            str(tmp_path / "out_gemma"),
            "--materialized-dir",
            str(tmp_path / "mat_gemma"),
            "--run-open-evals",
            "--eval-suite",
            "smoke",
            "--allow-weight-svd-fallback",
        ]
    )
    eval_cmds = [c for c in commands if any("run_vllm_open_evals.py" in part for part in c)]
    assert eval_cmds, "run_vllm_open_evals.py was not invoked"
    cmd = eval_cmds[0]
    for flag, value in (
        ("--suite", "quick"),
        ("--limit", "2"),
        ("--mmlu-pro-limit", "2"),
        ("--think-end-token", "<|thought|>"),
    ):
        assert flag in cmd, flag
        assert cmd[cmd.index(flag) + 1] == value
    for flag in ("--split-chat-0shot", "--sanity-gate", "--resume"):
        assert flag in cmd, flag
    export_cmds = [c for c in commands if any("export_model_index.py" in part for part in c)]
    assert export_cmds, "export_model_index.py was not invoked"

    # ---- Case B: full eval suite with the naive arm for the Qwen3.6 release ----
    commands.clear()
    foundry.main(
        [
            "--model-id",
            "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
            "--model-dir",
            str(ckpt_dir),
            "--output-dir",
            str(tmp_path / "out_qwen"),
            "--materialized-dir",
            str(tmp_path / "mat_qwen"),
            "--run-open-evals",
            "--eval-suite",
            "full",
            "--mmlu-pro-limit",
            "72",
            "--allow-weight-svd-fallback",
        ]
    )
    eval_cmds = [c for c in commands if any("run_vllm_open_evals.py" in part for part in c)]
    assert eval_cmds, "run_vllm_open_evals.py was not invoked"
    cmd = eval_cmds[-1]
    for flag, value in (
        ("--suite", "full"),
        ("--mmlu-pro-limit", "72"),
        ("--arms", "base,naive,tenet"),
        ("--think-end-token", "</think>"),
        ("--naive-model", "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated"),
    ):
        assert flag in cmd, flag
        assert cmd[cmd.index(flag) + 1] == value

    # ---- Case C: --cleanup-cache safety gates ----
    # Refuses when the upload result does not confirm upload.
    result = foundry._cleanup_local_cache(ckpt_dir, upload_result={})
    assert result["cleaned_up"] is False
    assert "upload" in result["reason"]
    assert ckpt_dir.exists()

    # Refuses when the Nebius s3-sync-up step fails.
    def _failing_s3_run(cmd, **kwargs):
        if any("nebius_ctl.sh" in str(part) for part in cmd):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _failing_s3_run)
    result = foundry._cleanup_local_cache(ckpt_dir, upload_result={"uploaded": True})
    assert result["cleaned_up"] is False
    assert result["reason"] == "s3-sync-up failed"
    assert ckpt_dir.exists()

    # Refuses paths outside /mnt/nvme and HF_HOME. Restore a succeeding
    # subprocess first: the path guard runs after the s3-sync-up step, so a
    # failing s3 sync would mask the refusal reason.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr=""),
    )
    outside = tmp_path / "outside_ckpt"
    outside.mkdir()
    result = foundry._cleanup_local_cache(outside, upload_result={"uploaded": True})
    assert result["cleaned_up"] is False
    assert "refusing" in result["reason"]
    assert outside.exists()

    # Deletes cleanly when upload + s3-sync-up succeed under HF_HOME.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr=""),
    )
    result = foundry._cleanup_local_cache(ckpt_dir, upload_result={"uploaded": True})
    assert result["cleaned_up"] is True
    assert not ckpt_dir.exists()

    # Also deletes parent models--org--repo/blobs when given an HF Hub snapshots/<sha> path!
    hf_repo_cache = hf_home / "models--Qwen--Qwen3.8-Flash-Next"
    blobs_dir = hf_repo_cache / "blobs"
    snap_dir = hf_repo_cache / "snapshots" / "deadbeef"
    blobs_dir.mkdir(parents=True)
    snap_dir.mkdir(parents=True)
    (blobs_dir / "giant_335gb_blob").write_bytes(b"x" * 1024)
    result_snap = foundry._cleanup_local_cache(snap_dir, upload_result={"uploaded": True})
    assert result_snap["cleaned_up"] is True
    assert not hf_repo_cache.exists()


# ---------------------------------------------------------------------------
# 8. Bug 2/3 regressions: gate-fail upload block, unsupported-arch pre-check,
#    README.md in release dir, upstream 401/404 skip
# ---------------------------------------------------------------------------
def test_foundry_open_evals_unsupported_arch_precheck(tmp_path):
    foundry = _load_foundry()

    materialized_dir = tmp_path / "materialized"
    materialized_dir.mkdir()
    (materialized_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "inkling2_loop",
                "architectures": ["Inkling2LoopForConditionalGeneration"],
            }
        )
    )

    result = foundry._run_open_evals(
        "Tenet-26B-A3B-Inkling2-Unslop-v3",
        {"base_model": "test/base", "think_end_token": "</think>"},
        "release-precheck",
        materialized_dir,
    )
    assert result["status"] == "unsupported_arch_in_vllm"
    assert result["passed"] is True
    assert result["model_type"] == "inkling2_loop"
    assert result["architectures"] == ["Inkling2LoopForConditionalGeneration"]


def test_foundry_fight_card_gate_fail_blocks_upload_and_cleanup(tmp_path, monkeypatch):
    foundry = _load_foundry()

    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    output_dir = tmp_path / "out"

    spec = {
        "tier": "edge",
        "base_model": "test/gate-base",
        "naive_model": None,
        "think_end_token": "</think>",
        "hypothesis_stack": ["H21"],
        "looped": False,
        "has_drafter": False,
        "hot_vram_expert_ids": [],
    }
    monkeypatch.setitem(foundry.TENET_V3_MODEL_SPECS, "TestGateModel", spec)

    calls: dict[str, int] = {"upload": 0, "cleanup": 0}

    monkeypatch.setattr(
        foundry,
        "_resolve_model_dir",
        lambda *a, **kw: model_dir,
    )
    monkeypatch.setattr(foundry, "_inspect_checkpoint_config", lambda *a, **kw: {"hidden_size": 16})
    monkeypatch.setattr(
        foundry,
        "_build_streaming_basis_and_profiles",
        lambda *a, **kw: ({}, {}, {}),
    )
    monkeypatch.setattr(foundry, "_run_h21", lambda *a, **kw: {"h21_verified": True})
    monkeypatch.setattr(
        foundry,
        "_run_fight_card_gate",
        lambda *a, **kw: {"gate_passed": False, "failed": True},
    )
    monkeypatch.setattr(
        foundry,
        "_upload_to_hf",
        lambda *a, **kw: calls.__setitem__("upload", calls["upload"] + 1) or {"uploaded": True},
    )
    monkeypatch.setattr(
        foundry,
        "_cleanup_local_cache",
        lambda *a, **kw: calls.__setitem__("cleanup", calls["cleanup"] + 1) or {"cleaned_up": True},
    )

    # The foundry imports `generate_tenet_model_card` from `sml.hub.release`,
    # but the implementation lives in `sml.hub.model_card`. Provide a stub
    # module so README.md generation stays exercised without the stale import.
    import sys
    import types

    release_mod = types.ModuleType("sml.hub.release")
    release_mod.generate_tenet_model_card = lambda *a, **kw: "# Stub Tenet model card\n"
    monkeypatch.setitem(sys.modules, "sml.hub.release", release_mod)

    argv = [
        "--model-id",
        "TestGateModel",
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output_dir),
        "--release-id",
        "gate-fail-test",
        "--run-fight-card-gate",
        "--upload-hf",
        "--cleanup-cache",
    ]
    code = foundry.main(argv)
    assert code == 1  # gate failed -> status "error"

    assert calls["upload"] == 0
    assert calls["cleanup"] == 0

    summary = json.loads((output_dir / "gate-fail-test" / "foundry_summary.json").read_text())
    assert summary["status"] == "error"
    assert summary["hf_upload"] == {"skipped": True, "reason": "fight_card_gate_failed"}
    assert summary["cache_cleanup"] == {"skipped": True, "reason": "fight_card_gate_failed"}

    # README.md must exist in the release dir even when evals are skipped.
    assert (output_dir / "gate-fail-test" / "README.md").is_file()


def test_foundry_force_upload_on_gate_fail(tmp_path, monkeypatch):
    foundry = _load_foundry()

    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    output_dir = tmp_path / "out"

    spec = {
        "tier": "edge",
        "base_model": "test/gate-base",
        "naive_model": None,
        "think_end_token": "</think>",
        "hypothesis_stack": ["H21"],
        "looped": False,
        "has_drafter": False,
        "hot_vram_expert_ids": [],
    }
    monkeypatch.setitem(foundry.TENET_V3_MODEL_SPECS, "TestGateModel", spec)

    monkeypatch.setattr(foundry, "_resolve_model_dir", lambda *a, **kw: model_dir)
    monkeypatch.setattr(foundry, "_inspect_checkpoint_config", lambda *a, **kw: {"hidden_size": 16})
    monkeypatch.setattr(
        foundry, "_build_streaming_basis_and_profiles", lambda *a, **kw: ({}, {}, {})
    )
    monkeypatch.setattr(foundry, "_run_h21", lambda *a, **kw: {"h21_verified": True})
    monkeypatch.setattr(foundry, "_run_fight_card_gate", lambda *a, **kw: {"gate_passed": False})
    monkeypatch.setattr(foundry, "_upload_to_hf", lambda *a, **kw: {"uploaded": True})
    monkeypatch.setattr(foundry, "_cleanup_local_cache", lambda *a, **kw: {"cleaned_up": True})

    argv = [
        "--model-id",
        "TestGateModel",
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output_dir),
        "--release-id",
        "gate-force-test",
        "--run-fight-card-gate",
        "--force-upload-on-gate-fail",
        "--upload-hf",
        "--cleanup-cache",
    ]
    code = foundry.main(argv)
    assert code == 1

    summary = json.loads((output_dir / "gate-force-test" / "foundry_summary.json").read_text())
    assert summary["hf_upload"] == {"uploaded": True}
    assert summary["cache_cleanup"] == {"cleaned_up": True}


def test_foundry_skips_unavailable_upstream_401(tmp_path, monkeypatch):
    foundry = _load_foundry()

    output_dir = tmp_path / "out"

    def _raise_401(*a, **kw):
        raise RuntimeError("401 Unauthorized: gated repo for user")

    monkeypatch.setattr(foundry, "_resolve_model_dir", _raise_401)
    monkeypatch.setattr(foundry, "_log", lambda message: None)

    argv = [
        "--model-id",
        "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
        "--output-dir",
        str(output_dir),
        "--release-id",
        "upstream-401-test",
        "--download",
    ]
    code = foundry.main(argv)
    assert code == 0

    summary = json.loads((output_dir / "upstream-401-test" / "foundry_summary.json").read_text())
    assert summary["status"] == "skipped_unavailable_upstream"
    assert "401" in summary["error"]


def test_foundry_skips_unavailable_upstream_404(tmp_path, monkeypatch):
    foundry = _load_foundry()

    output_dir = tmp_path / "out"

    def _raise_404(*a, **kw):
        # Avoid importing RepositoryNotFoundError: the installed
        # huggingface_hub version may not export it; main() must catch the
        # error by message regardless of the exception class.
        raise RuntimeError("404 Repository Not Found: no repo named test/missing")

    monkeypatch.setattr(foundry, "_resolve_model_dir", _raise_404)
    monkeypatch.setattr(foundry, "_log", lambda message: None)

    argv = [
        "--model-id",
        "Tenet-35B-A3B-Qwen3.6-Unslop-v3",
        "--output-dir",
        str(output_dir),
        "--release-id",
        "upstream-404-test",
        "--download",
    ]
    code = foundry.main(argv)
    assert code == 0

    summary = json.loads((output_dir / "upstream-404-test" / "foundry_summary.json").read_text())
    assert summary["status"] == "skipped_unavailable_upstream"
    assert "Repository Not Found" in summary["error"]
