"""Fast CPU unit tests for H20: Router-Aware & Shared-Expert Lancet Surgery.

All tests run on tiny synthetic blocks / state dicts — no real checkpoints.
"""

from __future__ import annotations

import json
import math
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from sml.cli import build_parser
from sml.hub.model_card import (
    _TENET_NAME_PATTERN,
    CANONICAL_TENET_NAMES,
    parse_tenet_model_name,
)
from sml.hub.validator import (
    _inspect_weights,
    verify_moe_router_freeze,
    verify_provider_readiness,
)
from sml.surgery.cka import (
    get_moe_layer_components,
    get_transformer_layers,
    is_router_weight_key,
)
from sml.surgery.subspace import (
    apply_moe_lancet_recipe_streaming,
    apply_moe_lancet_surgery,
    collect_moe_expert_profiles,
    project_and_snap_subspace_norm_preserving,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

D_MODEL = 32
D_FF = 64
NUM_EXPERTS = 4
TOP_K = 2
VOCAB = 16
MIN_SELECTIVE_WEIGHT = 0.05
MAX_BLOCK16 = 3.2


# ---------------------------------------------------------------------------
# Synthetic building blocks
# ---------------------------------------------------------------------------


def _directions(seed: int = 2026) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(2, D_MODEL, generator=gen)
    return [raw[i] / raw[i].norm() for i in range(raw.shape[0])]


def _expected_lancet(weight: torch.Tensor, dirs: list[torch.Tensor], scale: float) -> torch.Tensor:
    """Full-scale reference edit (H18 quant-snap, block-16 crest cap 3.2)."""
    return project_and_snap_subspace_norm_preserving(
        weight,
        dirs,
        scale=scale,
        num_iters=4,
        max_block_cf=MAX_BLOCK16,
        preserve_row_norm=True,
    )


class _RouterProj(nn.Module):
    """Wrapped router: the Linear lives in ``.proj`` (Gemma4-style)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(D_MODEL, NUM_EXPERTS, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.proj(x), dim=-1)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = q @ k.transpose(-1, -2) / math.sqrt(D_MODEL)
        return self.o_proj(torch.softmax(scores, dim=-1) @ v)


class _SharedExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FF, bias=False)
        self.up_proj = nn.Linear(D_MODEL, D_FF, bias=False)
        self.down_proj = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)


class _PackedExperts(nn.Module):
    """Gemma4-style packed experts: 3D ``down_proj`` of shape (E, d, d_ff)."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, D_FF, D_MODEL) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, D_MODEL, D_FF) * 0.02)

    def forward(
        self, flat: torch.Tensor, top_val: torch.Tensor, top_idx: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros_like(flat)
        for expert in range(NUM_EXPERTS):
            weight = (top_val * (top_idx == expert)).sum(-1, keepdim=True)
            hidden = torch.tanh(flat @ self.gate_up_proj[expert].T)
            out += weight * (hidden @ self.down_proj[expert].T)  # noqa: E501
        return out


class _QwenExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FF, bias=False)
        self.up_proj = nn.Linear(D_MODEL, D_FF, bias=False)
        self.down_proj = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)


def _topk(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    top_val, top_idx = torch.topk(probs, TOP_K, dim=-1)
    return top_val / top_val.sum(-1, keepdim=True).clamp_min(1e-9), top_idx


class _Gemma4Model(nn.Module):
    """language_model.model.layers, hybrid attention flags, packed 3D experts."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = _RouterProj()
                self.experts = _PackedExperts()
                self.shared_expert = _SharedExpert()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                flat = x.reshape(-1, D_MODEL)
                top_val, top_idx = _topk(self.router(flat))
                routed = self.experts(flat, top_val, top_idx)
                return (routed + self.shared_expert(flat)).reshape(x.shape)

        class _Block(nn.Module):
            def __init__(self, index: int) -> None:
                super().__init__()
                self.self_attn = _Attention()
                self.mlp = _MLP()
                self.attention_type = "full_attention" if index % 2 == 0 else "sliding_attention"

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.self_attn(x) + self.mlp(x)

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList(_Block(i) for i in range(num_layers))

        class _LM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Inner()

        self.language_model = _LM()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.config = SimpleNamespace(num_experts_per_tok=TOP_K, num_local_experts=NUM_EXPERTS)
        _seed_deterministic_routing(self, self.language_model.model.layers)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.language_model.model.layers:
            x = x + layer(x)
        return x


class _Qwen3MoeModel(nn.Module):
    """model.layers, ``mlp.gate`` Linear, unpacked ModuleList experts."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = nn.Linear(D_MODEL, NUM_EXPERTS, bias=False)
                self.experts = nn.ModuleList(_QwenExpert() for _ in range(NUM_EXPERTS))
                self.shared_expert = _SharedExpert()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                flat = x.reshape(-1, D_MODEL)
                top_val, top_idx = _topk(self.gate(flat))
                routed = torch.zeros_like(flat)
                for expert_idx, expert in enumerate(self.experts):
                    weight = (top_val * (top_idx == expert_idx)).sum(-1, keepdim=True)
                    routed += weight * expert(flat)
                return (routed + self.shared_expert(flat)).reshape(x.shape)

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = _Attention()
                self.mlp = _MLP()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.self_attn(x) + self.mlp(x)

        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Block() for _ in range(num_layers))
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.config = SimpleNamespace(num_experts_per_tok=TOP_K, num_local_experts=NUM_EXPERTS)
        _seed_deterministic_routing(self, self.model.layers)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = x + layer(x)
        return x


class _Mistral3Model(nn.Module):
    """language_model.model.layers with dense blocks (Devstral 24B layout)."""

    def __init__(self) -> None:
        super().__init__()

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(D_MODEL, D_FF, bias=False)
                self.up_proj = nn.Linear(D_MODEL, D_FF, bias=False)
                self.down_proj = nn.Linear(D_FF, D_MODEL, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                hidden = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
                return self.down_proj(hidden)

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = _Attention()
                self.mlp = _MLP()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.self_attn(x) + self.mlp(x)

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList(_Block() for _ in range(2))

        class _LM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Inner()

        self.language_model = _LM()
        self.embed = nn.Embedding(VOCAB, D_MODEL)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.language_model.model.layers:
            x = x + layer(x)
        return x


def _seed_deterministic_routing(model: nn.Module, layers: nn.ModuleList) -> None:
    """Token ids 0-7 route to experts 0/1 (target); ids 8-15 to experts 2/3."""
    with torch.no_grad():
        model.embed.weight.copy_(torch.eye(VOCAB, D_MODEL))
        half = VOCAB // 2
        for layer in layers:
            router_w = torch.zeros(NUM_EXPERTS, D_MODEL)
            router_w[0, :half] = 50.0
            router_w[1, :half] = 50.0
            router_w[2, half:VOCAB] = 50.0
            router_w[3, half:VOCAB] = 50.0
            router = layer.mlp.router if hasattr(layer.mlp, "router") else layer.mlp.gate
            linear = router.proj if hasattr(router, "proj") else router
            linear.weight.copy_(router_w)


def _target_anchor_ids() -> tuple[torch.Tensor, torch.Tensor]:
    half = VOCAB // 2
    return torch.arange(0, half).unsqueeze(0), torch.arange(half, VOCAB).unsqueeze(0)


# ---------------------------------------------------------------------------
# Layer/component introspection
# ---------------------------------------------------------------------------


def test_get_transformer_layers_supports_all_three_layouts() -> None:
    gemma4 = _Gemma4Model()
    layers = get_transformer_layers(gemma4)
    assert len(layers) == 2
    # Hybrid sliding/global attention flags alternate.
    assert layers[0].attention_type == "full_attention"
    assert layers[1].attention_type == "sliding_attention"

    qwen3 = _Qwen3MoeModel()
    assert len(get_transformer_layers(qwen3)) == 2

    mistral = _Mistral3Model()
    assert len(get_transformer_layers(mistral)) == 2

    bare = nn.Module()
    bare.layers = nn.ModuleList([nn.Linear(4, 4)])
    assert len(get_transformer_layers(bare)) == 1


def test_get_moe_layer_components_gemma4_packed() -> None:
    layer = get_transformer_layers(_Gemma4Model())[0]
    components = get_moe_layer_components(layer)
    assert components["is_moe"] is True
    assert components["o_proj"] is layer.self_attn.o_proj
    # Wrapped router: the wrapper is returned so hooks see the full forward.
    assert components["router"] is layer.mlp.router
    assert hasattr(components["router"], "proj")
    assert components["shared_down_proj"] is layer.mlp.shared_expert.down_proj
    assert components["routed_experts_unpacked"] == []
    packed = components["routed_experts_packed"]
    assert isinstance(packed, torch.Tensor)
    assert packed.shape == (NUM_EXPERTS, D_MODEL, D_FF)


def test_get_moe_layer_components_qwen3_unpacked() -> None:
    layer = get_transformer_layers(_Qwen3MoeModel())[0]
    components = get_moe_layer_components(layer)
    assert components["is_moe"] is True
    assert components["router"] is layer.mlp.gate
    assert components["shared_down_proj"] is layer.mlp.shared_expert.down_proj
    assert components["routed_experts_packed"] is None
    unpacked = components["routed_experts_unpacked"]
    assert [idx for idx, _ in unpacked] == list(range(NUM_EXPERTS))
    assert all(expert.down_proj.weight.shape == (D_MODEL, D_FF) for _, expert in unpacked)


def test_get_moe_layer_components_dense_mistral_block() -> None:
    layer = get_transformer_layers(_Mistral3Model())[0]
    components = get_moe_layer_components(layer)
    assert components["is_moe"] is False
    assert components["router"] is None
    assert components["routed_experts_packed"] is None
    assert components["routed_experts_unpacked"] == []
    assert components["o_proj"] is layer.self_attn.o_proj
    # Pure dense layer without routed experts: no shared down projection.
    assert components["shared_down_proj"] is None


def test_is_router_weight_key_matches_routers_not_expert_gates() -> None:
    assert is_router_weight_key("model.layers.0.mlp.gate.weight")
    assert is_router_weight_key("model.layers.3.mlp.gate.bias")
    assert is_router_weight_key("language_model.model.layers.1.mlp.router.proj.weight")
    assert is_router_weight_key("model.layers.0.router.weight")
    assert is_router_weight_key("model.layers.2.router.proj.weight")
    assert is_router_weight_key("model.layers.0.block_sparse_moe.gate.weight")
    assert is_router_weight_key("model.layers.0.mlp.gate.e_score_correction_bias")
    # Per-expert gate weights are NOT routers.
    assert not is_router_weight_key("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert not is_router_weight_key("model.layers.0.mlp.experts.1.gate_up_proj.weight")
    assert not is_router_weight_key("model.layers.0.mlp.experts.down_proj")
    assert not is_router_weight_key("model.layers.0.self_attn.q_proj.weight")
    assert not is_router_weight_key("model.layers.0.mlp.down_proj.weight")


# ---------------------------------------------------------------------------
# Expert profiling
# ---------------------------------------------------------------------------


def test_collect_moe_expert_profiles_selective_weights() -> None:
    model = _Gemma4Model()
    target_ids, anchor_ids = _target_anchor_ids()
    profiles = collect_moe_expert_profiles(
        model, target_ids, anchor_ids, beta=1.0, eps=1.0e-8, top_k=TOP_K
    )
    assert set(profiles) == {0, 1}
    for profile in profiles.values():
        assert profile["num_experts"] == NUM_EXPERTS
        assert profile["p_target"].sum().item() == pytest.approx(1.0)
        assert profile["p_anchor"].sum().item() == pytest.approx(1.0)
        selective = profile["selective_weights"]
        assert selective.shape == (NUM_EXPERTS,)
        assert bool(((selective >= 0.0) & (selective <= 1.0)).all())
        # Target-heavy experts are fully selective; anchor-heavy ones near 0.
        assert float(selective[1]) > 0.9
        assert float(selective[0]) > MIN_SELECTIVE_WEIGHT
        assert float(selective[2]) < MIN_SELECTIVE_WEIGHT
        assert float(selective[3]) < MIN_SELECTIVE_WEIGHT
    assert profiles[0]["beta"] == 1.0 and profiles[0]["eps"] == 1.0e-8


def test_collect_moe_expert_profiles_qwen3_gate_router() -> None:
    model = _Qwen3MoeModel()
    target_ids, anchor_ids = _target_anchor_ids()
    profiles = collect_moe_expert_profiles(
        model, target_ids, anchor_ids, beta=1.0, eps=1.0e-8, top_k=TOP_K
    )
    assert set(profiles) == {0, 1}
    for profile in profiles.values():
        assert profile["p_target"].sum().item() == pytest.approx(1.0)
        assert profile["p_anchor"].sum().item() == pytest.approx(1.0)
        selective = profile["selective_weights"]
        assert float(selective[2]) < MIN_SELECTIVE_WEIGHT
        assert float(selective[3]) < MIN_SELECTIVE_WEIGHT
        assert float(selective[1]) > 0.9


# ---------------------------------------------------------------------------
# apply_moe_lancet_surgery
# ---------------------------------------------------------------------------


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}


@pytest.mark.parametrize("builder", [_Gemma4Model, _Qwen3MoeModel])
def test_apply_moe_lancet_surgery_router_frozen_and_selective(builder) -> None:
    model = builder()
    dirs = _directions()
    target_ids, anchor_ids = _target_anchor_ids()
    profiles = collect_moe_expert_profiles(
        model, target_ids, anchor_ids, beta=1.0, eps=1.0e-8, top_k=TOP_K
    )
    pre = _snapshot(model)
    layers = get_transformer_layers(model)
    layer_bases = {idx: dirs for idx in range(len(layers))}

    result = apply_moe_lancet_surgery(
        model,
        layer_bases,
        profiles,
        scale=1.0,
        min_selective_weight=MIN_SELECTIVE_WEIGHT,
        min_selectivity_threshold=0.0,
        shared_expert_damping=1.0,
        quant_snap=True,
        quant_snap_iters=4,
        max_block16_crest_factor=MAX_BLOCK16,
    )

    # Router freeze is bit-for-bit exact.
    assert result["router_weight_drift"] == 0.0
    freeze = verify_moe_router_freeze(pre, model)
    assert freeze["router_freeze_ok"] is True
    assert freeze["router_weight_drift"] == 0.0
    assert freeze["checked_router_tensors"] >= 2
    assert freeze["drifted_keys"] == []

    assert result["edited_layers"] == [0, 1]
    assert result["edited_shared_experts"] == 2
    assert result["edited_routed_experts"] == 4  # experts 0/1 per layer
    assert result["skipped_routed_experts"] == 4  # anchor-only experts 2/3
    assert result["quant_snap"] is True

    prefix = "language_model.model." if builder is _Gemma4Model else "model."
    for idx in range(len(layers)):
        layer = layers[idx]
        # o_proj and shared_expert.down_proj are edited at full scale.
        o_pre = pre[f"{prefix}layers.{idx}.self_attn.o_proj.weight"]
        o_post = layer.self_attn.o_proj.weight
        assert not torch.equal(o_pre, o_post)
        assert torch.equal(o_post, _expected_lancet(o_pre, dirs, scale=1.0))

        shared_pre = pre[f"{prefix}layers.{idx}.mlp.shared_expert.down_proj.weight"]
        shared_post = layer.mlp.shared_expert.down_proj.weight
        assert not torch.equal(shared_pre, shared_post)
        assert torch.equal(shared_post, _expected_lancet(shared_pre, dirs, scale=1.0))

        selective = profiles[idx]["selective_weights"]
        for expert_idx in range(NUM_EXPERTS):
            if hasattr(layer.mlp, "router"):  # Gemma4: packed 3D experts
                down = layer.mlp.experts.down_proj
                key = f"{prefix}layers.{idx}.mlp.experts.down_proj"
                if float(selective[expert_idx]) <= MIN_SELECTIVE_WEIGHT:
                    # Anchor-only expert: exact zero weight drift.
                    assert torch.equal(down[expert_idx], pre[key][expert_idx])
                else:
                    edited = down[expert_idx]
                    assert not torch.equal(edited, pre[key][expert_idx])
                    # Projected along the surgical basis at scale s_e.
                    expected = _expected_lancet(
                        pre[key][expert_idx], dirs, scale=float(selective[expert_idx])
                    )
                    assert torch.equal(edited, expected)
            else:  # Qwen3: unpacked ModuleList experts
                down = layer.mlp.experts[expert_idx].down_proj
                key = f"{prefix}layers.{idx}.mlp.experts.{expert_idx}.down_proj.weight"
                if float(selective[expert_idx]) <= MIN_SELECTIVE_WEIGHT:
                    assert torch.equal(down.weight, pre[key])
                else:
                    assert not torch.equal(down.weight, pre[key])
                    expected = _expected_lancet(pre[key], dirs, scale=float(selective[expert_idx]))
                    assert torch.equal(down.weight, expected)


# ---------------------------------------------------------------------------
# apply_moe_lancet_recipe_streaming (safetensors checkpoint, no model)
# ---------------------------------------------------------------------------


def _write_streaming_checkpoint(model_dir: Path) -> dict[str, torch.Tensor]:
    """Sharded checkpoint with packed 3D + unpacked 2D experts and aux files."""
    tensors = {
        "model.layers.0.mlp.experts.down_proj": torch.randn(NUM_EXPERTS, D_MODEL, D_FF) * 0.05,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(D_MODEL, D_MODEL) * 0.05,
        "model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(D_MODEL, D_FF) * 0.05,
        "model.layers.0.mlp.router.proj.weight": torch.randn(NUM_EXPERTS, D_MODEL) * 0.05,
        "model.layers.1.self_attn.o_proj.weight": torch.randn(D_MODEL, D_MODEL) * 0.05,
        "model.layers.1.mlp.shared_expert.down_proj.weight": torch.randn(D_MODEL, D_FF) * 0.05,
        "model.layers.1.mlp.gate.weight": torch.randn(NUM_EXPERTS, D_MODEL) * 0.05,
    }
    for expert in range(NUM_EXPERTS):
        tensors[f"model.layers.1.mlp.experts.{expert}.down_proj.weight"] = (
            torch.randn(D_MODEL, D_FF) * 0.05
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["Gemma4ForConditionalGeneration"]}), encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"}),
        encoding="utf-8",
    )
    return tensors


def test_apply_moe_lancet_recipe_streaming_router_frozen_and_experts_edited(tmp_path: Path) -> None:
    src = tmp_path / "checkpoint"
    original = _write_streaming_checkpoint(src)
    dst = tmp_path / "edited"
    dirs = _directions()
    selective = torch.tensor([1.0, 0.9, 0.02, 0.0])
    profiles = {0: {"selective_weights": selective}, 1: {"selective_weights": selective}}

    result = apply_moe_lancet_recipe_streaming(
        src,
        dst,
        {0: dirs, 1: dirs},
        profiles,
        scale=1.0,
        min_selective_weight=MIN_SELECTIVE_WEIGHT,
        min_selectivity_threshold=0.0,
        shared_expert_damping=1.0,
        quant_snap=True,
        quant_snap_iters=4,
        max_block16_crest_factor=MAX_BLOCK16,
    )

    assert result["router_weight_drift"] == 0.0
    assert result["edited_layers"] == [0, 1]
    assert result["edited_o_proj"] == 2
    assert result["edited_shared_experts"] == 2
    assert result["edited_routed_experts"] == 4
    assert result["skipped_routed_experts"] == 4
    assert result["max_block16_crest_factor"] == MAX_BLOCK16
    assert result["output_dir"] == str(dst)
    assert result["max_row_crest_factor"] >= 1.0

    edited = load_file(str(dst / "model-00001-of-00001.safetensors"))

    # Routers copied through bit-for-bit.
    for key in ("model.layers.0.mlp.router.proj.weight", "model.layers.1.mlp.gate.weight"):
        assert torch.equal(edited[key], original[key])

    # 3D packed experts (layer 0): target-heavy edited, anchor-only untouched.
    packed = edited["model.layers.0.mlp.experts.down_proj"]
    assert not torch.equal(packed[0], original["model.layers.0.mlp.experts.down_proj"][0])
    assert not torch.equal(packed[1], original["model.layers.0.mlp.experts.down_proj"][1])
    assert torch.equal(packed[2], original["model.layers.0.mlp.experts.down_proj"][2])
    assert torch.equal(packed[3], original["model.layers.0.mlp.experts.down_proj"][3])

    # 2D unpacked experts (layer 1).
    for expert in range(NUM_EXPERTS):
        key = f"model.layers.1.mlp.experts.{expert}.down_proj.weight"
        if selective[expert] <= MIN_SELECTIVE_WEIGHT:
            assert torch.equal(edited[key], original[key])
        else:
            assert not torch.equal(edited[key], original[key])
            assert torch.equal(
                edited[key],
                _expected_lancet(original[key], dirs, scale=float(selective[expert])),
            )

    # o_proj / shared expert edited at full scale on both layers.
    for key in (
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.layers.1.mlp.shared_expert.down_proj.weight",
    ):
        assert not torch.equal(edited[key], original[key])
        assert torch.equal(edited[key], _expected_lancet(original[key], dirs, scale=1.0))

    # Aux files are copied alongside the edited shards.
    assert (dst / "config.json").is_file()
    assert (dst / "tokenizer_config.json").is_file()


# ---------------------------------------------------------------------------
# model_card.py: Tenet naming (v3 MoE names + v1/v2 backward compatibility)
# ---------------------------------------------------------------------------


def test_parse_tenet_model_name_v3_moe_and_dense() -> None:
    parsed = parse_tenet_model_name("Tenet-26B-A4B-Gemma4-Unslop-v3")
    assert parsed == {
        "total_params": 26.0,
        "active_params": "A4B",
        "family": "Gemma4",
        "version": "v3",
        "is_moe": True,
    }

    parsed = parse_tenet_model_name("Tenet-30B-A3B-Qwen3-Unslop-v3")
    assert parsed["total_params"] == 30.0
    assert parsed["active_params"] == "A3B"
    assert parsed["family"] == "Qwen3"
    assert parsed["version"] == "v3"
    assert parsed["is_moe"] is True

    parsed = parse_tenet_model_name("Tenet-24B-Mistral3.1-Unslop-v3")
    assert parsed["total_params"] == 24.0
    assert parsed["active_params"] is None
    assert parsed["family"] == "Mistral3.1"
    assert parsed["version"] == "v3"
    assert parsed["is_moe"] is False


def test_parse_tenet_model_name_v1_v2_backward_compatible() -> None:
    v1_v2_names = [
        "Tenet-3B-Qwen2.5-Unslop-v1",
        "Tenet-8B-Llama3.1-Unslop-v1",
        "Tenet-12B-Nemo-Unslop-v1",
        "Tenet-32B-Qwen2.5-Unslop-v1",
        "Tenet-70B-Llama3.3-Unslop-v1",
        "Tenet-12B-Nemo-Unslop-v2",
        "Tenet-8B-Llama3.1-Unslop-v2",
    ]
    for name in v1_v2_names:
        parsed = parse_tenet_model_name(name)
        assert parsed["active_params"] is None
        assert parsed["is_moe"] is False
        assert parsed["version"].startswith("v")
        assert isinstance(parsed["total_params"], float)

    # Every canonical name parses, and the pattern accepts exactly them.
    assert all(_TENET_NAME_PATTERN.match(name) for name in CANONICAL_TENET_NAMES)
    assert "Tenet-26B-A4B-Gemma4-Unslop-v3" in CANONICAL_TENET_NAMES
    assert "Tenet-30B-A3B-Qwen3-Unslop-v3" in CANONICAL_TENET_NAMES
    assert "Tenet-24B-Mistral3.1-Unslop-v3" in CANONICAL_TENET_NAMES


@pytest.mark.parametrize(
    "bad_name",
    [
        "Tenet-26B-A4B-Gemma4-Unslop",  # missing revision
        "Tenet-26B-A4B-Gemma4-Unslop-vX",  # non-numeric revision
        "tenet-3B-Qwen2.5-Unslop-v1",  # wrong prefix case
        "Tenet-3B-Qwen2.5-v1",  # missing Unslop token
        "Tenet-Qwen2.5-Unslop-v1",  # missing size
        "NotATenet-3B-Qwen2.5-Unslop-v1",
    ],
)
def test_parse_tenet_model_name_rejects_non_canonical(bad_name: str) -> None:
    with pytest.raises(ValueError):
        parse_tenet_model_name(bad_name)


# ---------------------------------------------------------------------------
# validator.py: readiness, packed-weight inspection, router freeze
# ---------------------------------------------------------------------------


def _write_ready_checkpoint(tmp_path: Path, architecture: str) -> Path:
    checkpoint = tmp_path / architecture
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": [architecture], "hidden_size": D_MODEL}),
        encoding="utf-8",
    )
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "User: {{ messages[-1].content }}"}),
        encoding="utf-8",
    )
    save_file(
        {"model.layers.0.mlp.experts.down_proj": torch.randn(4, 8, 16) * 0.01},
        str(checkpoint / "model.safetensors"),
    )
    return checkpoint


@pytest.mark.parametrize(
    "architecture",
    [
        "Gemma4ForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Mistral3ForConditionalGeneration",
    ],
)
def test_verify_provider_readiness_accepts_moe_v3_architectures(
    tmp_path: Path, architecture: str
) -> None:
    report = verify_provider_readiness(_write_ready_checkpoint(tmp_path, architecture))
    assert report["architectures"] == [architecture]
    assert report["checks"]["standard_architecture"] is True
    assert report["ready"] is True, report["errors"]


def test_inspect_weights_handles_packed_3d_expert_tensors(tmp_path: Path) -> None:
    packed = torch.randn(3, 8, 16) * 0.01
    packed[1, 0, 0] = float("nan")  # hidden inside the packed dimension
    packed[2, 0, 0] = 60.0  # per-expert outlier above the FP8 threshold
    weight_file = tmp_path / "model.safetensors"
    save_file({"model.layers.0.mlp.experts.down_proj": packed}, str(weight_file))

    checks = {"no_nan_inf": True, "no_unmerged_lora": True}
    errors: list[str] = []
    stats = _inspect_weights([weight_file], checks, errors)

    assert checks["no_nan_inf"] is False
    assert any("experts.down_proj" in error and "[1]" in error for error in errors)
    assert stats["max_abs_weight"] == pytest.approx(60.0)
    assert stats["max_block16_crest_factor"] > 0.0
    assert checks["no_unmerged_lora"] is True


def test_verify_moe_router_freeze_detects_drift() -> None:
    baseline = {
        "model.layers.0.mlp.gate.weight": torch.randn(NUM_EXPERTS, D_MODEL),
        "model.layers.1.mlp.router.proj.weight": torch.randn(NUM_EXPERTS, D_MODEL),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(D_MODEL, D_MODEL),
    }
    # Identical copy: zero drift, freeze ok.
    identical = {key: tensor.clone() for key, tensor in baseline.items()}
    result = verify_moe_router_freeze(baseline, identical)
    assert result["router_weight_drift"] == 0.0
    assert result["router_freeze_ok"] is True
    assert result["checked_router_tensors"] == 2
    assert result["drifted_keys"] == []

    # Perturbed gate: non-zero drift, freeze fails, drifted key reported.
    drifted = {key: tensor.clone() for key, tensor in baseline.items()}
    drifted["model.layers.0.mlp.gate.weight"] = drifted["model.layers.0.mlp.gate.weight"] + 0.5
    result = verify_moe_router_freeze(baseline, drifted)
    assert result["router_weight_drift"] > 0.0
    assert result["router_freeze_ok"] is False
    assert result["drifted_keys"] == ["model.layers.0.mlp.gate.weight"]
    # Non-router weights are ignored even when edited.
    edited_o = {key: tensor.clone() for key, tensor in baseline.items()}
    edited_o["model.layers.0.self_attn.o_proj.weight"] = torch.randn(D_MODEL, D_MODEL)
    assert verify_moe_router_freeze(baseline, edited_o)["router_freeze_ok"] is True


def test_copy_aux_files_overwrites_read_only_destinations(tmp_path: Path) -> None:
    from sml.surgery.subspace import _copy_aux_files

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    index = src / "model.safetensors.index.json"
    index.write_text("{}", encoding="utf-8")
    aux = src / "config.json"
    aux.write_text("{}", encoding="utf-8")
    shard = src / "model.safetensors"
    shard.write_bytes(b"x")

    # Pre-existing read-only 0o444 destinations (as in HF cache blobs).
    dst_index = dst / "model.safetensors.index.json"
    dst_index.write_text("old", encoding="utf-8")
    dst_index.chmod(0o444)
    dst_aux = dst / "config.json"
    dst_aux.write_text("old", encoding="utf-8")
    dst_aux.chmod(0o444)

    _copy_aux_files(src, dst, copy_index=True)

    assert dst_index.read_text(encoding="utf-8") == "{}"
    assert dst_aux.read_text(encoding="utf-8") == "{}"
    assert not (dst / "model.safetensors").exists()
    assert stat.S_IMODE(dst_index.stat().st_mode) == 0o644
    assert stat.S_IMODE(dst_aux.stat().st_mode) == 0o644

    # copy_index=False leaves the index untouched.
    _copy_aux_files(src, dst, copy_index=False)
    assert dst_index.read_text(encoding="utf-8") == "{}"


# ---------------------------------------------------------------------------
# CLI: sml verify-h20
# ---------------------------------------------------------------------------


def test_cli_verify_h20_end_to_end(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)  # default experiment/config paths are repo-relative
    args = build_parser().parse_args(["verify-h20", "--output-dir", str(tmp_path)])
    assert args.experiment == "experiments/h20_moe_router_aware_lancet.yaml"

    return_code = args.func(args)
    assert return_code == 0

    summary_path = tmp_path / "h20_moe_router_aware_lancet" / "h20_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["experiment_id"] == "h20_moe_router_aware_lancet"
    assert summary["hypothesis_id"] == "H20"
    assert summary["router_weight_drift"] == 0.0
    assert summary["router_freeze_ok"] is True
    assert summary["anchor_only_expert_weight_drift"] == 0.0
    assert summary["edited_layers"] == [0, 1]
    assert summary["skipped_routed_experts"] == 4

    printed = json.loads(capsys.readouterr().out)
    assert printed["experiment_id"] == summary["experiment_id"]
    assert printed["router_weight_drift"] == 0.0
