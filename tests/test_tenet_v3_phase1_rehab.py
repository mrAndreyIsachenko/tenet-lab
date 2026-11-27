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


def test_fight_card_gate_uses_spawned_worker_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """vLLM runs in spawned children per checkpoint; parent stays vllm-free."""
    import importlib.util
    import inspect
    import json
    import multiprocessing
    import types

    foundry_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tenet_v3_foundry.py"
    spec = importlib.util.spec_from_file_location("run_tenet_v3_foundry", foundry_path)
    assert spec is not None and spec.loader is not None
    foundry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foundry)

    gate_src = inspect.getsource(foundry._run_fight_card_gate)
    worker_src = inspect.getsource(foundry._fight_card_vllm_worker)
    assert "from vllm import" not in gate_src
    assert "from vllm import" in worker_src
    assert 'get_context("spawn")' in gate_src

    from sml.evals.peer_eval import (
        ANTI_SLOP_PROMPTS,
        BORDERLINE_OVER_REFUSAL_PROMPTS,
        TENET_ARENA_HOUSE_SERIES,
    )

    spawned: list[tuple[str, tuple[object, ...]]] = []

    class _FakeProcess:
        def __init__(self, target: object, args: tuple[object, ...]) -> None:
            self.target = target
            self.args = args
            self.exitcode = 0

        def start(self) -> None:
            spawned.append((getattr(self.target, "__name__", "?"), self.args))
            assert callable(self.target)
            self.target(*self.args)

        def join(self) -> None:
            return None

    fake_ctx = types.SimpleNamespace(Process=_FakeProcess)

    def _fake_get_context(name: str) -> types.SimpleNamespace:
        assert name == "spawn"
        return fake_ctx

    monkeypatch.setattr(multiprocessing, "get_context", _fake_get_context)

    n_items = (
        len(TENET_ARENA_HOUSE_SERIES)
        + len(BORDERLINE_OVER_REFUSAL_PROMPTS)
        + len(ANTI_SLOP_PROMPTS)
    )
    ok_think = (
        "<think>checking the constraints and exact numbers carefully.</think>\n\n"
        "Set the timeout to 30 seconds and restart the worker process to apply the change."
    )

    def _fake_worker(
        checkpoint: str,
        tp_size: int,
        house_prompts: list[str],
        direct_prompts: list[str],
        result_path: str,
    ) -> None:
        assert tp_size == 2
        assert len(house_prompts) == len(TENET_ARENA_HOUSE_SERIES)
        assert len(direct_prompts) == len(BORDERLINE_OVER_REFUSAL_PROMPTS) + len(ANTI_SLOP_PROMPTS)
        Path(result_path).write_text(json.dumps({"status": "ok", "texts": [ok_think] * n_items}))

    monkeypatch.setattr(foundry, "_fight_card_vllm_worker", _fake_worker)
    monkeypatch.setattr(foundry, "_detect_tensor_parallel_size", lambda: 2)

    release_dir = tmp_path / "release"
    materialized = release_dir / "materialized"
    materialized.mkdir(parents=True)

    report = foundry._run_fight_card_gate(
        "tenet-test",
        {"base_model": "base-model", "think_end_token": "</think>"},
        release_dir,
        materialized_dir=materialized,
    )

    assert [name for name, _ in spawned] == ["_fake_worker", "_fake_worker"]
    assert spawned[0][1][0].endswith("materialized")
    assert spawned[1][1][0] == "base-model"

    gate = json.loads((release_dir / "fight_card_gate.json").read_text())
    assert gate["model_id"] == "tenet-test"
    assert gate["status"] == "ok"
    assert report["status"] == "ok"


def test_surgery_stage_isolated_spawn_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_run_surgery_isolated runs surgery in a spawned child and returns (h21, h22)."""
    import importlib.util
    import json
    import multiprocessing

    foundry_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tenet_v3_foundry.py"
    spec = importlib.util.spec_from_file_location("run_tenet_v3_foundry", foundry_path)
    assert spec is not None and spec.loader is not None
    foundry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foundry)

    helper_src = inspect.getsource(foundry._run_surgery_isolated)
    assert 'get_context("spawn")' in helper_src

    calls: list[tuple[str, str, float]] = []

    def _fake_worker(
        model_dir_str: str,
        materialized_dir_str: str,
        release_dir_str: str,
        spec: dict,
        args_ns_dict: dict,
        result_path: str,
    ) -> None:
        calls.append((model_dir_str, materialized_dir_str, args_ns_dict["scale"]))
        Path(result_path).write_text(
            json.dumps(
                {
                    "status": "ok",
                    "h21": {"h21_verified": True, "router_weight_drift": 0.0},
                    "h22": {"h22_verified": True},
                }
            )
        )

    monkeypatch.setattr(foundry, "_surgery_stage_worker", _fake_worker)

    # A real spawn cannot pickle a locally-defined fake worker, so replace the
    # context with a synchronous stand-in and record the requested start method.
    requested_methods: list[str] = []

    class _FakeProcess:
        def __init__(self, target: object, args: tuple[object, ...]) -> None:
            self.target = target
            self.args = args
            self.exitcode = 0

        def start(self) -> None:
            assert callable(self.target)
            self.target(*self.args)

        def join(self) -> None:
            return None

    def _fake_get_context(method: str) -> types.SimpleNamespace:
        requested_methods.append(method)
        return types.SimpleNamespace(Process=_FakeProcess)

    monkeypatch.setattr(multiprocessing, "get_context", _fake_get_context)

    args = types.SimpleNamespace(
        scale=0.35,
        min_selectivity_threshold=0.25,
        shared_expert_damping=0.40,
        strata_config=None,
        allow_weight_svd_fallback=False,
    )
    model_dir = tmp_path / "model"
    materialized = tmp_path / "materialized"
    release_dir = tmp_path / "release"
    for d in (model_dir, materialized, release_dir):
        d.mkdir()

    h21, h22 = foundry._run_surgery_isolated(
        model_dir,
        materialized,
        release_dir,
        {"looped": True, "has_drafter": False},
        args,
    )

    assert h21["h21_verified"] is True
    assert h22["h22_verified"] is True
    assert calls == [(str(model_dir), str(materialized), 0.35)]
    assert requested_methods == ["spawn"]


def test_peer_eval_quoted_think_inside_closed_envelope() -> None:
    """Quoted <think> mentions inside a closed <think>...</think> envelope still pass."""
    from sml.evals.peer_eval import run_peer_eval_gate

    quoted = (
        "<think>\nThe user prompt mentions <think> tags, so I should verify carefully.\n"
        "</think>\n\nThe verified result is 4,326 units across all three days."
    )
    rep = run_peer_eval_gate(responses=[quoted] * 24, require_think=True)
    assert rep["cot_tag_integrity_rate"] == 1.0
    assert rep["length_collapse_rate"] == 0.0
    assert rep["gate_passed"] is True


class _FakeOut:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.outputs = [_FakeOut(text)]


class _FakeSamplingParams:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _load_foundry() -> types.ModuleType:
    import importlib.util

    foundry_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tenet_v3_foundry.py"
    spec = importlib.util.spec_from_file_location("run_tenet_v3_foundry", foundry_path)
    assert spec is not None and spec.loader is not None
    foundry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foundry)
    return foundry


def test_fight_card_gate_vllm_batching_and_dynamic_require_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_model_len=16384, house max_tokens=8192, enable_thinking=False for direct."""
    import sml.evals.peer_eval as peer_eval_mod

    foundry = _load_foundry()

    import multiprocessing

    class _InProcessProcess:
        def __init__(self, target: object, args: tuple[object, ...]) -> None:
            self.target = target
            self.args = args
            self.exitcode = 0

        def start(self) -> None:
            assert callable(self.target)
            self.target(*self.args)

        def join(self) -> None:
            return None

    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        lambda name: types.SimpleNamespace(Process=_InProcessProcess),
    )

    class _FakeTok:
        chat_template = "x"

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            thinking = bool(kwargs["enable_thinking"]) if "enable_thinking" in kwargs else True
            fmt = f"<user>{messages[0]['content']}<end>\n"
            return fmt + ("<think>" if thinking else "")

    class _FakeLLM:
        instances: list[_FakeLLM] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.calls: list[tuple[list[str], _FakeSamplingParams]] = []
            _FakeLLM.instances.append(self)

        def get_tokenizer(self) -> _FakeTok:
            return _FakeTok()

        def generate(
            self, prompts: list[str], sampling: _FakeSamplingParams
        ) -> list[_FakeCompletion]:
            self.calls.append((list(prompts), sampling))
            return [_FakeCompletion("reasoned answer</think>\n\nDone.") for _ in prompts]

    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(__import__("sys").modules, "vllm", fake_vllm)

    captured: dict[str, object] = {}

    def _fake_gate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok", "gate_passed": True}

    monkeypatch.setattr(peer_eval_mod, "run_paired_behavioral_gate", _fake_gate)

    spec_cfg = {"base_model": "fake/base", "think_end_token": "</think>"}
    report = foundry._run_fight_card_gate(
        "fake/tenet", spec_cfg, Path("/tmp/fc_g"), materialized_dir=Path("/tmp/fc_g/mat")
    )
    assert report["status"] == "ok"

    llm = _FakeLLM.instances[-1]
    assert llm.kwargs["max_model_len"] == 16384
    assert len(llm.calls) == 2
    (house_prompts, house_sp), (direct_prompts, direct_sp) = llm.calls
    assert len(house_prompts) == 24
    assert len(direct_prompts) == 32 + len(ANTI_SLOP_PROMPTS)
    assert house_sp.kwargs["max_tokens"] == 8192
    assert direct_sp.kwargs["max_tokens"] == 1536  # thinking disabled branch
    assert all(p.endswith("<think>") for p in direct_prompts) is False
    # enable_thinking=False was honored: direct prompts carry no template think tag.
    assert all(not p.endswith("<think>") for p in direct_prompts)
    # House template opens <think>; generation did not start with it -> prepended.
    assert all(p.endswith("<think>") for p in house_prompts)
    assert captured["require_think"] is True
    assert all(r.startswith("<think>") for r in captured["base_house_responses"])  # type: ignore[union-attr]


def test_fight_card_gate_dynamic_require_think_direct_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A base model whose template answers directly does not force require_think."""
    import sml.evals.peer_eval as peer_eval_mod

    foundry = _load_foundry()

    import multiprocessing

    class _InProcessProcess:
        def __init__(self, target: object, args: tuple[object, ...]) -> None:
            self.target = target
            self.args = args
            self.exitcode = 0

        def start(self) -> None:
            assert callable(self.target)
            self.target(*self.args)

        def join(self) -> None:
            return None

    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        lambda name: types.SimpleNamespace(Process=_InProcessProcess),
    )

    class _FakeTok:
        chat_template = "x"

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            if "enable_thinking" in kwargs:
                return f"<user>{messages[0]['content']}<end>\n"
            raise TypeError("enable_thinking not supported")

    class _FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            pass

        def get_tokenizer(self) -> _FakeTok:
            return _FakeTok()

        def generate(self, prompts: list[str], sampling: object) -> list[_FakeCompletion]:
            # Fallback path: template without think opening -> direct answers.
            return [_FakeCompletion("Plain answer without deliberation.") for _ in prompts]

    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm",
        types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams),
    )

    captured: dict[str, object] = {}

    def _fake_gate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok", "gate_passed": True}

    monkeypatch.setattr(peer_eval_mod, "run_paired_behavioral_gate", _fake_gate)

    spec_cfg = {"base_model": "fake/base", "think_end_token": "</think>"}
    report = foundry._run_fight_card_gate(
        "fake/tenet", spec_cfg, Path("/tmp/fc_d"), materialized_dir=Path("/tmp/fc_d/mat")
    )
    assert report["status"] == "ok"
    assert captured["require_think"] is False


def test_compact_artifacts_not_length_collapse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compact valid json_config / sql_readonly answers are not length collapses."""
    from sml.evals.artifact_verifier import TENET_ARENA_HOUSE_SERIES
    from sml.evals.peer_eval import run_peer_eval_gate

    artifact_items = [
        i for i in TENET_ARENA_HOUSE_SERIES if i.artifact_type in ("json_config", "sql_readonly")
    ]
    assert artifact_items, "expected artifact-typed house prompts"

    responses = []
    prompts = []
    for item in artifact_items:
        body = '{"port": 5432}' if item.artifact_type == "json_config" else "SELECT 1"
        responses.append(f"<think>ok</think>\n{body}")
        prompts.append(item)

    monkeypatch.setattr(
        __import__("sml.evals.peer_eval", fromlist=["verify_house_artifact"]),
        "verify_house_artifact",
        lambda spec, text: types.SimpleNamespace(artifact_valid=True),
    )
    report = run_peer_eval_gate(responses=responses, prompts=prompts, require_think=True)
    assert report["length_collapse_rate"] == pytest.approx(0.0)
    assert report["gate_passed"] is True


def test_paired_gate_cot_tag_integrity_relative_non_degradation() -> None:
    """cot_tag_integrity is relative: tenet must not drop below base below 0.995."""
    from sml.evals.peer_eval import run_paired_behavioral_gate

    ok_think = (
        "<think>checking the exact constraints and numbers carefully.</think>\n\n"
        "Set the timeout to 30 seconds, then restart the worker process so the "
        "configuration change is applied across all replicas."
    )
    missing_think = (
        "Set the timeout to 30 seconds, then restart the worker process so the "
        "configuration change is applied across all replicas."
    )

    def _responses(valid: int) -> list[str]:
        return [ok_think] * valid + [missing_think] * (24 - valid)

    # Both arms at 23/24 valid think tags: equal rates pass (non-degradation).
    report = run_paired_behavioral_gate(
        base_house_responses=_responses(23),
        tenet_house_responses=_responses(23),
        base_refusal_responses=[ok_think] * 32,
        tenet_refusal_responses=[ok_think] * 32,
        base_slop_responses=[ok_think] * 24,
        tenet_slop_responses=[ok_think] * 24,
    )
    details = report["gate_details"]["cot_tag_integrity"]
    assert details["passed"] is True
    assert details["base_value"] == pytest.approx(23 / 24)
    assert details["value"] == pytest.approx(23 / 24)
    assert report["gate_passed"] is True

    # Tenet drops to 20/24 while base stays at 23/24 (< 0.995): gate fails.
    report = run_paired_behavioral_gate(
        base_house_responses=_responses(23),
        tenet_house_responses=_responses(20),
        base_refusal_responses=[ok_think] * 32,
        tenet_refusal_responses=[ok_think] * 32,
        base_slop_responses=[ok_think] * 24,
        tenet_slop_responses=[ok_think] * 24,
    )
    assert report["gate_details"]["cot_tag_integrity"]["passed"] is False
    assert report["gate_details"]["cot_tag_integrity"]["value"] < 0.995
    assert report["gate_details"]["cot_tag_integrity"]["base_value"] == pytest.approx(23 / 24)
    assert report["gate_passed"] is False


def test_normalize_deliberation_tags() -> None:
    """Alternative deliberation tag families normalize to <think>/</think>."""
    from sml.evals.peer_eval import normalize_deliberation_tags

    assert normalize_deliberation_tags("<|thought|>step<channel|>done") == "<think>step</think>done"
    assert (
        normalize_deliberation_tags("<reasoning>step</reasoning>done") == "<think>step</think>done"
    )
    assert normalize_deliberation_tags("<think>already</think>fine") == "<think>already</think>fine"
    assert normalize_deliberation_tags("no tags here") == "no tags here"


def test_borderline_prompts_are_xstest_style() -> None:
    """32 XSTest-style prompts with sensitive triggers and no self-hedging."""
    from sml.evals.peer_eval import ANTI_SLOP_PROMPTS, BORDERLINE_OVER_REFUSAL_PROMPTS

    assert len(BORDERLINE_OVER_REFUSAL_PROMPTS) == 32
    assert len(ANTI_SLOP_PROMPTS) > 0
    joined = " ".join(BORDERLINE_OVER_REFUSAL_PROMPTS)
    assert "in my own web app" not in joined
    assert "authorized audit" not in joined
    assert any("zombie processes" in p for p in BORDERLINE_OVER_REFUSAL_PROMPTS)
    assert any("192.168.1.1" in p for p in BORDERLINE_OVER_REFUSAL_PROMPTS)


def test_extended_refusal_and_slop_regex() -> None:
    """Extended regexes catch hedged refusals and fresh slop phrases."""
    from sml.evals.peer_eval import _EXTENDED_REFUSAL_RE, _EXTENDED_SLOP_RE

    assert _EXTENDED_REFUSAL_RE.search("I can’t help with that request.")
    assert _EXTENDED_REFUSAL_RE.search("As an AI, I apologize, but I'm unable to assist.")
    assert not _EXTENDED_REFUSAL_RE.search("The cannot-step loop in my pseudo-code is fine.")
    assert _EXTENDED_SLOP_RE.search("It's important to note that this is a testament to craft.")
    assert _EXTENDED_SLOP_RE.search("Let's dive in and embark on a journey.")
    assert not _EXTENDED_SLOP_RE.search("Set the timeout to 30 seconds.")
