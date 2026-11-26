"""Unit tests for Tenet v3 Phase 1 architectural and calibration fixes."""

from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sml.evals.peer_eval import (
    ANTI_SLOP_PROMPTS,
    BORDERLINE_OVER_REFUSAL_PROMPTS,
    run_paired_behavioral_gate,
)
from sml.surgery.cka import is_router_weight_key
from sml.surgery.looped import is_drafter_weight_key
from sml.surgery.subspace import (
    UnsupportedForwardArchError,
    _parse_moe_weight_key,
    apply_moe_lancet_recipe_streaming,
    calibrate_streaming_moe_basis_from_activations,
)


def test_latent_moe_pullback_and_linear_attn_and_frozen_keys(tmp_path: Path) -> None:
    """LatentMoE 1536->3072 pullback, linear_attn.out_proj, and frozen hyper-connections."""
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()

    torch.manual_seed(0)
    w_up = torch.randn(3072, 1536, dtype=torch.bfloat16) * 0.02
    w_exp = torch.randn(4, 1536, 64, dtype=torch.bfloat16) * 0.02
    w_lin = torch.randn(3072, 3072, dtype=torch.bfloat16) * 0.02
    frozen = {
        "model.layers.0.attn_hyper_connection.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "model.layers.0.mlp_hyper_connection.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "model.layers.0.hc_attn_layer.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "model.layers.0.hc_mlp_layer.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "model.layers.0.engram_layer.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.compressed_indexer.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    }
    for k in frozen:
        assert is_router_weight_key(k) is True
    parsed_lin = _parse_moe_weight_key("model.layers.0.linear_attn.out_proj.weight")
    assert parsed_lin == ("o_proj", 0, None)

    shard = {
        "model.layers.0.mlp.routed_expert_up_proj.weight": w_up,
        "model.layers.0.mlp.experts.down_proj": w_exp,
        "model.layers.0.linear_attn.out_proj.weight": w_lin,
        **frozen,
    }
    save_file(shard, str(src_dir / "model.safetensors"))

    basis = torch.linalg.qr(torch.randn(3072, 2))[0].T.contiguous()
    profiles = {
        0: {
            "selective_weights": torch.tensor([0.9, 0.0, 0.0, 0.0], dtype=torch.float32),
        }
    }
    res = apply_moe_lancet_recipe_streaming(
        src_dir,
        dst_dir,
        layer_bases={0: basis},
        expert_profiles=profiles,
        min_selectivity_threshold=0.5,
        quant_snap=False,
    )
    assert res["edited_o_proj"] == 1
    assert res["edited_shared_experts"] == 1
    assert res["edited_routed_experts"] == 1
    assert res["untouched_expert_ratio"] == pytest.approx(0.75)
    assert res["router_weight_drift"] == 0.0

    with safe_open(dst_dir / "model.safetensors", framework="pt", device="cpu") as f:
        for k, v in frozen.items():
            assert torch.equal(f.get_tensor(k), v)


def test_h22_drafter_prefixes_dfly_mtp_and_last_layer() -> None:
    """All three H22 drafter/MTP schemes are recognized without matching non-target weights."""
    assert is_drafter_weight_key("model.dfly_layers.0.self_attn.o_proj.weight") is True
    assert is_drafter_weight_key("model.dfly_layers.1.mlp.down_proj.weight") is True
    assert is_drafter_weight_key("model.dfly_layers.0.self_attn.q_proj.weight") is False
    assert is_drafter_weight_key("mtp.layers.0.mlp.down_proj.weight") is True
    assert (
        is_drafter_weight_key("model.layers.46.mlp.down_proj.weight", num_hidden_layers=46) is True
    )
    assert (
        is_drafter_weight_key("model.layers.45.mlp.down_proj.weight", num_hidden_layers=46) is False
    )


def test_contrastive_router_selectivity_and_unsupported_arch_halt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contrastive calibration computes unstretched selectivity and halts on bad arch."""

    class _FakeGate(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 4 experts; if mean(x) > 0 (target_pos) prefer expert 0, else prefer expert 1.
            b, t, _ = x.shape
            logits = torch.zeros(b, t, 4, dtype=torch.float32)
            if float(x.mean()) > 0.5:
                logits[..., 0] = 10.0
            else:
                logits[..., 1] = 10.0
            return logits

    class _FakeLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = types.SimpleNamespace()
            self.gate = _FakeGate()
            self.mlp = torch.nn.Module()
            self.mlp.gate = self.gate

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
            _ = self.mlp.gate(x)
            return (x,)

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace(num_experts_per_tok=1)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_FakeLayer()])

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> object:
            # input_ids == 1 -> target_pos, input_ids == 0 -> anchor/neg
            val = input_ids.to(torch.float32).mean()
            x = torch.full((input_ids.shape[0], input_ids.shape[1], 16), float(val))
            for layer in self.model.layers:
                x = layer(x)[0]
            return types.SimpleNamespace(logits=x)

    class _FakeTok:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(
            self, texts: list[str] | str, return_tensors: str | None = None, **kwargs: object
        ) -> dict[str, object]:
            is_single = isinstance(texts, str)
            items = [texts] if is_single else list(texts)
            is_pos = any(
                "cannot" in t.lower() or "ai" in t.lower() or "delve" in t.lower() for t in items
            )
            fill = 1 if is_pos else 0
            if return_tensors == "pt":
                ids = torch.full((len(items), 8), fill, dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            row = [fill] * 8
            return {
                "input_ids": row if is_single else [row for _ in items],
                "attention_mask": [1] * 8 if is_single else [[1] * 8 for _ in items],
            }

    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoConfig",
        types.SimpleNamespace(from_pretrained=lambda *a, **kw: types.SimpleNamespace()),
    )
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *a, **kw: _FakeTok()),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda *a, **kw: _FakeModel()),
    )

    bases, profiles, meta = calibrate_streaming_moe_basis_from_activations(
        tmp_path,
        selected_layers=[0],
        num_experts=4,
        max_prompts_per_group=4,
    )
    assert meta["calibration_mode"] == "activation_contrastive"
    assert 0 in bases and 0 in profiles
    s = profiles[0]["selective_weights"]
    # Expert 2 and 3 were never selected -> 0.0 (no min-max stretch).
    assert float(s[2]) == 0.0
    assert float(s[3]) == 0.0

    # Unsupported arch raises UnsupportedForwardArchError when allow_weight_svd_fallback=False.
    def _raise_bad_arch(*a: object, **kw: object) -> None:
        raise ValueError("bad arch")

    monkeypatch.setattr(
        transformers,
        "AutoConfig",
        types.SimpleNamespace(from_pretrained=_raise_bad_arch),
    )
    with pytest.raises(UnsupportedForwardArchError):
        calibrate_streaming_moe_basis_from_activations(tmp_path, selected_layers=[0], num_experts=4)


def test_no_synthetic_fight_card_gate_and_auto_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic _calibrated_response is removed; paired gate and auto-TP work."""
    import importlib.util

    foundry_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tenet_v3_foundry.py"
    spec = importlib.util.spec_from_file_location("run_tenet_v3_foundry", foundry_path)
    assert spec is not None and spec.loader is not None
    foundry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foundry)

    src = inspect.getsource(foundry._run_fight_card_gate)
    assert "_calibrated_response" not in src
    assert len(BORDERLINE_OVER_REFUSAL_PROMPTS) == 32
    assert len(ANTI_SLOP_PROMPTS) == 24

    ok_think = (
        "<think>checking the constraints and exact numbers carefully.</think>\n\n"
        "Set the timeout to 30 seconds and restart the worker process to apply the change."
    )
    refusal = (
        "<think>thinking.</think>\n\nI cannot assist with that request as an AI language model."
    )
    report = run_paired_behavioral_gate(
        base_house_responses=[ok_think] * 24,
        tenet_house_responses=[ok_think] * 24,
        base_refusal_responses=[refusal] * 16 + [ok_think] * 16,
        tenet_refusal_responses=[ok_think] * 32,
        base_slop_responses=[ok_think] * 24,
        tenet_slop_responses=[ok_think] * 24,
    )
    assert report["status"] == "ok"
    assert report["false_refusal_rate_base"] == pytest.approx(0.5)
    assert report["false_refusal_rate_tenet"] == pytest.approx(0.0)
    assert report["false_refusal_delta_pp"] == pytest.approx(-50.0)
    assert report["slop_delta_pp"] == pytest.approx(0.0)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert foundry._detect_tensor_parallel_size() == 4
