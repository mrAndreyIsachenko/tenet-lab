"""Fast, fully-mocked unit tests for the OpenEvals runners, model card
model-index generation, and GGUF repo-root sanitization.

No real model downloads, no vllm/lm-eval execution: scripts are loaded via
importlib (with heavy third-party deps stubbed out where needed) and every
network/API interaction is faked.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_MODULE_CACHE: dict[str, Any] = {}


def _load_module(name: str, path: Path, stubs: dict[str, Any] | None = None) -> Any:
    """Import a repo script by file path, optionally stubbing sys.modules entries.

    Stubs are only present for the duration of the import (they satisfy the
    module-level imports of heavy deps such as torch/transformers) and are
    removed afterwards so the rest of the test suite is unaffected. The
    imported module object keeps its own references to the stubs, which is
    fine for the pure helpers under test.
    """
    cached = _MODULE_CACHE.get(name)
    if cached is not None:
        return cached
    inserted: list[str] = []
    try:
        for mod_name, stub in (stubs or {}).items():
            if mod_name not in sys.modules:
                sys.modules[mod_name] = stub
                inserted.append(mod_name)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULE_CACHE[name] = module
        return module
    finally:
        for mod_name in inserted:
            sys.modules.pop(mod_name, None)
        sys.modules.pop(name, None)


def _stub_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture(scope="module")
def vllm_runner():
    return _load_module(
        "test_open_evals_run_vllm_open_evals",
        REPO_ROOT / "scripts" / "eval" / "run_vllm_open_evals.py",
    )


@pytest.fixture(scope="module")
def external_runner():
    torch_stub = _stub_module("torch")
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.backends = _stub_module(
        "torch.backends", mps=_stub_module("x", is_available=lambda: False)
    )
    torch_stub.cuda = _stub_module("x", is_available=lambda: False)
    transformers_stub = _stub_module(
        "transformers",
        AutoModelForCausalLM=type("AutoModelForCausalLM", (), {}),
        AutoTokenizer=type("AutoTokenizer", (), {}),
    )
    safetensors_stub = _stub_module("safetensors")
    safetensors_stub.torch = _stub_module("safetensors.torch", load_file=lambda *a, **k: {})
    hf_stub = _stub_module("huggingface_hub", hf_hub_download=lambda *a, **k: "")
    harness_stub = _stub_module(
        "sml.evals.harness",
        _run_sql_incident_eval=lambda *a, **k: {},
        _run_unslop_audit=lambda *a, **k: {},
        build_hf_chat_responder=lambda *a, **k: None,
        run_lm_eval_suite=lambda *a, **k: {},
    )
    surgery_stub = _stub_module(
        "sml.surgery.subspace",
        apply_lancet_biprojected_surgery=lambda *a, **k: None,
    )
    return _load_module(
        "test_open_evals_run_external_benchmark_3way",
        REPO_ROOT / "scripts" / "eval" / "run_external_benchmark_3way.py",
        stubs={
            "torch": torch_stub,
            "transformers": transformers_stub,
            "safetensors": safetensors_stub,
            "safetensors.torch": safetensors_stub.torch,
            "huggingface_hub": hf_stub,
            "sml.evals.harness": harness_stub,
            "sml.surgery.subspace": surgery_stub,
        },
    )


@pytest.fixture(scope="module")
def model_card():
    return _load_module(
        "test_open_evals_model_card",
        REPO_ROOT / "src" / "sml" / "hub" / "model_card.py",
    )


# ---------------------------------------------------------------------------
# scripts/eval/run_vllm_open_evals.py
# ---------------------------------------------------------------------------


class TestParseThinkingResponse:
    def test_closed_think_block(self, vllm_runner):
        assert vllm_runner.parse_thinking_response("<think>reasoning</think>42") == ("42", False)

    def test_unclosed_think_block(self, vllm_runner):
        assert vllm_runner.parse_thinking_response("<think>cutoff") == ("", True)

    def test_plain_response(self, vllm_runner):
        assert vllm_runner.parse_thinking_response("42") == ("42", False)

    def test_custom_think_end_token(self, vllm_runner):
        text = "<think>reasoning||END||42"
        assert vllm_runner.parse_thinking_response(
            text, think_start_token="<think>", think_end_token="||END||"
        ) == ("42", False)

    def test_answer_is_stripped_after_end_token(self, vllm_runner):
        assert vllm_runner.parse_thinking_response("<think>r</think>  42\n") == ("42", False)


class TestComputeUnclosedThinkRate:
    def test_mixed_responses_exact_fraction(self, vllm_runner):
        responses = [
            "<think>closed</think>answer",
            "<think>never closed",
            "plain answer",
            "</think>no start token",
        ]
        assert vllm_runner.compute_unclosed_think_rate(responses) == pytest.approx(0.25)

    def test_empty_responses(self, vllm_runner):
        assert vllm_runner.compute_unclosed_think_rate([]) == 0.0

    def test_all_unclosed(self, vllm_runner):
        responses = ["<think>a", "<think>b"]
        assert vllm_runner.compute_unclosed_think_rate(responses) == pytest.approx(1.0)


class TestExtractLmEvalSamplesUnclosedThinkRate:
    def test_list_of_samples_with_resps(self, vllm_runner):
        samples = [
            {"resps": [["</think>ok"]], "filtered_resps": ["ok"]},
            {"resps": [["<think>cutoff"]], "filtered_resps": ["<think>cutoff"]},
        ]
        assert vllm_runner.extract_lm_eval_samples_unclosed_think_rate(samples) == pytest.approx(
            0.5
        )

    def test_results_dict_with_samples_list(self, vllm_runner):
        payload = {
            "results": {"ifeval": {"acc": 0.5}},
            "samples": [
                {"filtered_resps": ["plain"]},
                {"filtered_resps": ["plain2"]},
                {"resps": [["<think>truncated"]]},
                {"resps": [["</think>done"]]},
            ],
        }
        assert vllm_runner.extract_lm_eval_samples_unclosed_think_rate(payload) == pytest.approx(
            0.25
        )

    def test_single_sample_dict(self, vllm_runner):
        sample = {"resps": [["<think>truncated"]], "filtered_resps": ["<think>truncated"]}
        assert vllm_runner.extract_lm_eval_samples_unclosed_think_rate(sample) == pytest.approx(1.0)

    def test_task_keyed_samples_dict_is_handled(self, vllm_runner):
        # lm-eval nests samples per task name; the helper only reads a flat
        # sample list / top-level resps, so a task-keyed mapping degrades to a
        # safe 0.0 instead of crashing (callers pass the flattened list).
        task_keyed = {
            "ifeval": [{"resps": [["</think>ok"]]}],
            "gsm8k_cot": [{"resps": [["<think>cutoff"]]}],
        }
        result = vllm_runner.extract_lm_eval_samples_unclosed_think_rate(task_keyed)
        assert isinstance(result, float)
        assert result == 0.0

    def test_custom_tokens_forwarded(self, vllm_runner):
        samples = [{"resps": [["<think>r||END||ok"]], "filtered_resps": ["ok"]}]
        assert vllm_runner.extract_lm_eval_samples_unclosed_think_rate(
            samples, think_end_token="||END||"
        ) == pytest.approx(0.0)

    def test_vllm_system_instruction_flag(self, vllm_runner, tmp_path):
        cmd = vllm_runner.build_lm_eval_command(
            "org/model",
            tasks=("ifeval",),
            output_path=tmp_path / "out",
            system_instruction=(
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        )
        assert cmd[cmd.index("--system_instruction") + 1].startswith("Please reason step by step")
        # Omitted -> no flag.
        cmd_plain = vllm_runner.build_lm_eval_command(
            "org/model", tasks=("ifeval",), output_path=tmp_path / "out2"
        )
        assert "--system_instruction" not in cmd_plain


class TestSuiteProfiles:
    def test_full_profile_no_limit_and_all_tasks(self, vllm_runner):
        profile = vllm_runner.SUITE_PROFILES["full"]
        assert profile["limit"] is None
        expected = {
            "ifeval",
            "hendrycks_math500",
            "gsm8k_cot",
            "gpqa_diamond_zeroshot",
            "mmlu_pro",
            "humaneval_plus",
            "mbpp_plus",
            "eqbench_creative_writing",
            "slop_score",
        }
        all_tasks = {
            *profile["tasks"],
            *profile["evalplus_tasks"],
            *profile["eqbench_tasks"],
        }
        assert expected <= all_tasks

    def test_quick_profile_limit_40(self, vllm_runner):
        assert vllm_runner.SUITE_PROFILES["quick"]["limit"] == 40


class TestBuildLmEvalCommand:
    def test_vllm_bf16(self, vllm_runner, tmp_path):
        cmd = vllm_runner.build_lm_eval_command(
            "org/model",
            tasks=["ifeval", "gsm8k_cot"],
            output_path=tmp_path / "raw" / "tenet",
            backend="vllm",
            precision="bf16",
            limit=None,
        )
        joined = " ".join(cmd)
        assert "--model vllm" in joined
        assert "think_end_token=</think>" in joined
        assert "max_gen_toks=8192" in joined
        assert "--limit" not in cmd
        assert "dtype=bfloat16" in joined
        model_args = cmd[cmd.index("--model_args") + 1]
        assert "pretrained=org/model" in model_args

    def test_vllm_bf16_with_limit(self, vllm_runner, tmp_path):
        cmd = vllm_runner.build_lm_eval_command(
            "org/model",
            tasks=["ifeval"],
            output_path=tmp_path,
            backend="vllm",
            precision="bf16",
            limit=40,
        )
        assert cmd[cmd.index("--limit") + 1] == "40"

    def test_vllm_fp8_uses_fp8_dtype(self, vllm_runner, tmp_path):
        cmd = vllm_runner.build_lm_eval_command(
            "org/model",
            tasks=["ifeval"],
            output_path=tmp_path,
            backend="vllm",
            precision="fp8",
            limit=None,
        )
        model_args = cmd[cmd.index("--model_args") + 1]
        assert "dtype=fp8" in model_args
        assert "fp8" in model_args

    def test_openai_compatible_gguf(self, vllm_runner, tmp_path):
        cmd = vllm_runner.build_lm_eval_command(
            "Tenet.gguf",
            tasks=["ifeval"],
            output_path=tmp_path,
            backend="openai-compatible",
            precision="gguf",
            api_base="http://localhost:8080/",
        )
        assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
        model_args = cmd[cmd.index("--model_args") + 1]
        assert "base_url=http://localhost:8080/v1" in model_args
        assert "--apply_chat_template" in cmd

    def test_vllm_apply_chat_template_disables_thinking(self, vllm_runner, tmp_path):
        cmd = vllm_runner.build_lm_eval_command(
            "org/model",
            tasks=["ifeval"],
            output_path=tmp_path,
            backend="vllm",
            precision="bf16",
            apply_chat_template=True,
            enable_thinking=False,
        )
        joined = " ".join(cmd)
        assert "--apply_chat_template" in cmd
        model_args = cmd[cmd.index("--model_args") + 1]
        assert "enable_thinking=False" in model_args
        # With thinking disabled the think_end_token must be dropped.
        assert "think_end_token" not in model_args
        assert "think_end_token=</think>" not in joined

    def test_vllm_default_keeps_think_end_token(self, vllm_runner, tmp_path):
        cmd = vllm_runner.build_lm_eval_command(
            "org/model",
            tasks=["ifeval"],
            output_path=tmp_path,
            backend="vllm",
            precision="bf16",
        )
        model_args = cmd[cmd.index("--model_args") + 1]
        assert "think_end_token=</think>" in model_args
        assert "enable_thinking" not in model_args
        assert "--apply_chat_template" not in cmd

    def test_unsupported_backend_raises(self, vllm_runner, tmp_path):
        with pytest.raises(ValueError):
            vllm_runner.build_lm_eval_command(
                "org/model",
                tasks=["ifeval"],
                output_path=tmp_path,
                backend="nonsense",
            )


class TestFormatThreeWayMarkdownTable:
    def test_renders_arms_and_delta_columns(self, vllm_runner):
        payload = {
            "model_id": "Tenet-14B",
            "base_model": "base-ref",
            "naive_model": "naive-ref",
            "tenet_model": "tenet-ref",
            "arms": {
                "base": {"unclosed_think_rate": 0.25},
                "naive": {"unclosed_think_rate": 0.5},
                "tenet": {"unclosed_think_rate": 0.0},
            },
        }
        report = vllm_runner.format_three_way_markdown_table(payload)
        assert "| Metric | Base | Naive Abliteration | Tenet |" in report
        assert "Δ Tenet" in report
        assert "unclosed_think_rate" in report
        assert "| unclosed_think_rate | 0.2500 | 0.5000 | 0.0000 | -0.2500 | -0.5000 |" in report

    def test_uses_supplied_deltas(self, vllm_runner):
        payload = {
            "arms": {
                "base": {"score": 0.1},
                "naive": {"score": 0.2},
                "tenet": {"score": 0.3},
            },
            "deltas_vs_base": {"score": 0.2},
            "deltas_vs_naive": {"score": 0.1},
        }
        report = vllm_runner.format_three_way_markdown_table(payload)
        assert "| score | 0.1000 | 0.2000 | 0.3000 | +0.2000 | +0.1000 |" in report


# ---------------------------------------------------------------------------
# scripts/eval/run_external_benchmark_3way.py
# ---------------------------------------------------------------------------


class TestResolveLimitExternalRunner:
    def test_full_suite_none_limit(self, external_runner):
        assert external_runner._resolve_limit(None, "full") is None

    def test_zero_limit_treated_as_none(self, external_runner):
        assert external_runner._resolve_limit(0, "quick") is None

    def test_quick_suite_default_40(self, external_runner):
        assert external_runner._resolve_limit(None, "quick") == 40

    def test_explicit_limit_overrides_suite(self, external_runner):
        assert external_runner._resolve_limit(100, "full") == 100

    def test_negative_limit_treated_as_none(self, external_runner):
        assert external_runner._resolve_limit(-5, "quick") is None


class TestResolveMmluProLimit:
    def test_none_is_full_split(self, vllm_runner):
        assert vllm_runner._resolve_mmlu_pro_limit(None) is None

    def test_zero_is_full_split(self, vllm_runner):
        assert vllm_runner._resolve_mmlu_pro_limit(0) is None

    def test_negative_is_full_split(self, vllm_runner):
        assert vllm_runner._resolve_mmlu_pro_limit(-5) is None

    def test_positive_limit_returns_int(self, vllm_runner):
        resolved = vllm_runner._resolve_mmlu_pro_limit(1000)
        assert resolved == 1000
        assert isinstance(resolved, int)


class TestMath500Recompute:
    def test_last_boxed_and_equivalence(self, vllm_runner):
        text = "Reasoning with $(0,3)$ and other $x$ steps.\n\\boxed{(3, \\frac{\\pi}{2})}"
        assert vllm_runner._last_boxed_only_string(text) == "(3, \\frac{\\pi}{2})"
        assert vllm_runner._remove_boxed("(3, \\frac{\\pi}{2})") == "(3, \\frac{\\pi}{2})"
        assert vllm_runner._math_answers_equiv(
            "(3, \\frac{\\pi}{2})", "\\left( 3, \\frac{\\pi}{2} \\right)"
        )

    def test_recompute_from_samples_real_world_case(self, vllm_runner, tmp_path):
        rows = [
            {
                "target": "\\left( 3, \\frac{\\pi}{2} \\right)",
                "resps": [["$(0,3)$ ... intermediate $x$ steps ... \\boxed{(3, \\frac{\\pi}{2})"]],
            },
            {"target": "2", "resps": [["\\boxed{2}"]]},
            {"target": "5", "resps": [["\\boxed{4}"]]},
        ]
        sub = tmp_path / "chat_0shot_math"
        sub.mkdir()
        with (sub / "samples_hendrycks_math500_2026.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        assert vllm_runner.recompute_hendrycks_math500_from_samples(sub) == round(2 / 3, 6)

    def test_recompute_real_release_dir(self, vllm_runner):
        real_dir = (
            REPO_ROOT
            / "results"
            / "external_evals"
            / "Tenet-35B-A3B-Qwen3.6-Unslop-v3"
            / "raw"
            / "tenet"
            / "chat_0shot_math"
        )
        if not (real_dir.is_dir() and list(real_dir.rglob("samples_hendrycks_math500_*.jsonl"))):
            pytest.skip("real chat_0shot_math samples not present on disk")
        score = vllm_runner.recompute_hendrycks_math500_from_samples(real_dir)
        assert score is not None
        assert score > 0.70

    @pytest.mark.parametrize("marker", ["401 Unauthorized", "inkling2_loop"])
    def test_fatal_markers_stop_on_first_attempt(self, vllm_runner, tmp_path, monkeypatch, marker):
        calls: list[list[str]] = []

        def fake_run(cmd, log_path=None):
            calls.append(cmd)
            return 1, "", f"Traceback: {marker} failed"

        def boom(seconds):
            raise AssertionError("sleep must not be called on fatal markers")

        monkeypatch.setattr(vllm_runner, "_run_subtask_with_cleanup", fake_run)
        monkeypatch.setattr(vllm_runner.time, "sleep", boom)
        record = vllm_runner._eval_arm(
            "base",
            "org/model",
            tasks=("hendrycks_math500",),
            evalplus_tasks=(),
            eqbench_tasks=(),
            raw_dir=tmp_path / "raw",
            limit=None,
            backend="vllm",
            precision="bf16",
            batch_size="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
            max_gen_toks=8192,
            think_start_token="<think>",
            think_end_token="</think>",
            api_base=None,
            dry_run=False,
            max_retries=2,
        )
        assert record["status"] == "error"
        assert len(calls) == 1


def _eval_arm_kwargs(vllm_runner: Any, raw_dir: Path, **overrides: Any) -> dict[str, Any]:
    """Default keyword arguments for ``_eval_arm`` (full suite profile)."""
    profile = vllm_runner.SUITE_PROFILES["full"]
    kwargs: dict[str, Any] = {
        "tasks": profile["tasks"],
        "evalplus_tasks": profile["evalplus_tasks"],
        "eqbench_tasks": profile["eqbench_tasks"],
        "raw_dir": raw_dir,
        "limit": None,
        "mmlu_pro_limit": 1000,
        "backend": "vllm",
        "precision": "bf16",
        "batch_size": "auto",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 16384,
        "max_gen_toks": 8192,
        "think_start_token": "<think>",
        "think_end_token": "</think>",
        "api_base": None,
        "dry_run": True,
    }
    kwargs.update(overrides)
    return kwargs


class TestEvalArmMmluProSplit:
    def test_dry_run_full_tasks_splits_mmlu_pro_command(self, vllm_runner, tmp_path):
        raw_dir = tmp_path / "raw" / "tenet"
        record = vllm_runner._eval_arm(
            "tenet", "org/model", **_eval_arm_kwargs(vllm_runner, raw_dir)
        )

        assert record["status"] == "dry-run"
        assert record["arm"] == "tenet"
        assert record["model"] == "org/model"
        cmds = record["commands"]
        assert len(cmds) == 2
        main_cmd, mmlu_cmd = cmds

        # Main run: everything except mmlu_pro, at the suite limit (None here).
        assert main_cmd[main_cmd.index("--tasks") + 1] == (
            "ifeval,hendrycks_math500,gsm8k_cot,gpqa_diamond_zeroshot"
        )
        assert "--limit" not in main_cmd
        assert main_cmd[main_cmd.index("--output_path") + 1] == str(raw_dir)
        assert record["command"] == main_cmd

        # mmlu_pro run: alone, with the override limit, in its own subdir.
        assert mmlu_cmd[mmlu_cmd.index("--tasks") + 1] == "mmlu_pro"
        assert mmlu_cmd[mmlu_cmd.index("--limit") + 1] == "1000"
        assert mmlu_cmd[mmlu_cmd.index("--output_path") + 1] == str(raw_dir / "mmlu_pro")

    def test_run_merges_scores_from_main_and_mmlu_pro_dirs(
        self, vllm_runner, tmp_path, monkeypatch
    ):
        raw_dir = tmp_path / "raw" / "tenet"
        mmlu_dir = raw_dir / "mmlu_pro"
        mmlu_dir.mkdir(parents=True)
        (raw_dir / "main_results.json").write_text(
            json.dumps({"results": {"ifeval": {"acc,strict-match": 0.5, "alias": "ifeval"}}}),
            encoding="utf-8",
        )
        (mmlu_dir / "mmlu_results.json").write_text(
            json.dumps({"results": {"mmlu_pro": {"acc": 0.612}}}),
            encoding="utf-8",
        )

        invoked: list[list[str]] = []

        class _FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **kwargs):
            invoked.append(list(cmd))
            return _FakeProc()

        monkeypatch.setattr(vllm_runner.subprocess, "run", fake_run)

        record = vllm_runner._eval_arm(
            "tenet",
            "org/model",
            **_eval_arm_kwargs(
                vllm_runner,
                raw_dir,
                tasks=("ifeval", "mmlu_pro"),
                evalplus_tasks=(),
                eqbench_tasks=(),
                mmlu_pro_limit=25,
                dry_run=False,
                resume=False,
            ),
        )

        assert record["status"] == "ok"
        assert len(invoked) == 2
        first_cmd = record["commands"][0]
        second_cmd = record["commands"][1]
        assert first_cmd[first_cmd.index("--output_path") + 1] == str(raw_dir)
        assert second_cmd[second_cmd.index("--output_path") + 1] == str(mmlu_dir)
        # Scores from both raw_dir and raw_dir / 'mmlu_pro' are merged.
        assert record["scores"] == {"ifeval_acc": 0.5, "mmlu_pro_acc": 0.612}
        assert record["unclosed_think_rate"] == 0.0


class TestValidateOpenEvalSanity:
    def test_near_zero_math_raises(self, vllm_runner):
        with pytest.raises(vllm_runner.SanityGateError):
            vllm_runner.validate_open_evals_sanity({"math_500": 0.0, "ifeval_strict": 0.6})

    def test_near_zero_hendrycks_alias_raises(self, vllm_runner):
        with pytest.raises(vllm_runner.SanityGateError):
            vllm_runner.validate_open_evals_sanity(
                {"hendrycks_math500_exact_match": 0.01, "ifeval_strict": 0.6}
            )

    def test_near_zero_ifeval_raises(self, vllm_runner):
        with pytest.raises(vllm_runner.SanityGateError):
            vllm_runner.validate_open_evals_sanity({"math_500": 0.5, "ifeval_strict": 0.05})

    def test_healthy_scores_pass(self, vllm_runner):
        vllm_runner.validate_open_evals_sanity({"math_500": 0.5, "ifeval_strict": 0.6})

    def test_runaway_mean_response_length_raises(self, vllm_runner):
        samples = [{"resps": [["x" * 5000]]} for _ in range(10)]
        with pytest.raises(vllm_runner.SanityGateError):
            vllm_runner.validate_open_evals_sanity(
                {"math_500": 0.5, "ifeval_strict": 0.6}, samples, max_gen_toks=1280
            )

    def test_short_samples_pass(self, vllm_runner):
        samples = [{"resps": [["x" * 100]]} for _ in range(10)]
        vllm_runner.validate_open_evals_sanity(
            {"math_500": 0.5, "ifeval_strict": 0.6}, samples, max_gen_toks=1280
        )


class TestEvalArmChat0ShotSplit:
    def test_dry_run_splits_chat_0shot_command(self, vllm_runner, tmp_path):
        raw_dir = tmp_path / "raw" / "tenet"
        record = vllm_runner._eval_arm(
            "tenet",
            "org/model",
            **_eval_arm_kwargs(vllm_runner, raw_dir, split_chat_0shot=True),
        )

        assert record["status"] == "dry-run"
        cmds = record["commands"]
        # main (non-chat, non-mmlu) + mmlu_pro split + ifeval chat_0shot
        # + hendrycks_math500 chat_0shot_math split.
        assert len(cmds) == 4
        ifeval_cmd = cmds[-2]
        assert ifeval_cmd[ifeval_cmd.index("--tasks") + 1] == "ifeval"
        assert ifeval_cmd[ifeval_cmd.index("--output_path") + 1] == str(raw_dir / "chat_0shot")
        assert "--apply_chat_template" in ifeval_cmd
        assert "--system_instruction" not in ifeval_cmd
        ifeval_model_args = ifeval_cmd[ifeval_cmd.index("--model_args") + 1]
        assert "enable_thinking=False" in ifeval_model_args
        assert "think_end_token" not in ifeval_model_args
        assert f"max_gen_toks={vllm_runner.DEFAULT_CHAT_0SHOT_MAX_GEN_TOKS}" in ifeval_cmd

        math_cmd = cmds[-1]
        assert math_cmd[math_cmd.index("--tasks") + 1] == "hendrycks_math500"
        assert math_cmd[math_cmd.index("--output_path") + 1] == str(raw_dir / "chat_0shot_math")
        assert "--apply_chat_template" in math_cmd
        assert math_cmd[math_cmd.index("--system_instruction") + 1] == (
            vllm_runner.DEFAULT_MATH_500_SYSTEM_INSTRUCTION
        )
        assert "boxed{}" in math_cmd[math_cmd.index("--system_instruction") + 1]
        math_model_args = math_cmd[math_cmd.index("--model_args") + 1]
        assert "enable_thinking=False" in math_model_args
        assert f"max_gen_toks={vllm_runner.DEFAULT_CHAT_0SHOT_MATH_MAX_GEN_TOKS}" in math_cmd

        main_cmd = cmds[0]
        assert main_cmd[main_cmd.index("--tasks") + 1] == "gsm8k_cot,gpqa_diamond_zeroshot"
        assert "--apply_chat_template" not in main_cmd

    def test_no_split_keeps_single_main_command(self, vllm_runner, tmp_path):
        raw_dir = tmp_path / "raw" / "tenet"
        record = vllm_runner._eval_arm(
            "tenet",
            "org/model",
            **_eval_arm_kwargs(
                vllm_runner,
                raw_dir,
                tasks=("ifeval", "gsm8k_cot"),
                evalplus_tasks=(),
                eqbench_tasks=(),
                mmlu_pro_limit=None,
                split_chat_0shot=False,
            ),
        )
        cmds = record["commands"]
        assert len(cmds) == 1
        assert cmds[0][cmds[0].index("--tasks") + 1] == "ifeval,gsm8k_cot"
        assert "--apply_chat_template" not in cmds[0]

    def test_loaders_ignore_chat_0shot_subdir(self, vllm_runner, tmp_path):
        raw_dir = tmp_path / "raw" / "tenet"
        chat_dir = raw_dir / "chat_0shot"
        chat_dir.mkdir(parents=True)
        (raw_dir / "main_results.json").write_text(
            json.dumps({"results": {"ifeval": {"acc,strict-match": 0.6}}}), encoding="utf-8"
        )
        (chat_dir / "chat_results.json").write_text(
            json.dumps({"results": {"ifeval": {"acc,strict-match": 0.9}}}), encoding="utf-8"
        )

        payload = vllm_runner._load_lm_eval_payload(raw_dir)
        assert payload is not None
        assert payload["results"]["ifeval"]["acc,strict-match"] == 0.6

        chat_payload = vllm_runner._load_lm_eval_payload(chat_dir)
        assert chat_payload is not None
        assert chat_payload["results"]["ifeval"]["acc,strict-match"] == 0.9

        assert vllm_runner._find_valid_subtask_payload(chat_dir) is not None
        # Scanning the parent must not pick up the nested chat_0shot payload.
        parent_payload = vllm_runner._find_valid_subtask_payload(raw_dir)
        assert parent_payload is not None
        assert parent_payload["results"]["ifeval"]["acc,strict-match"] == 0.6


class TestParseArms:
    def test_valid_subset(self, vllm_runner):
        assert vllm_runner._parse_arms("tenet,base") == ("tenet", "base")

    def test_whitespace_and_dedupe_preserve_order(self, vllm_runner):
        assert vllm_runner._parse_arms(" base , tenet, base ,tenet") == ("base", "tenet")

    def test_invalid_arm_raises(self, vllm_runner):
        with pytest.raises(ValueError, match="Unknown arm"):
            vllm_runner._parse_arms("tenet,quantized")

    def test_empty_raises(self, vllm_runner):
        with pytest.raises(ValueError, match="No valid arms"):
            vllm_runner._parse_arms("  ,  ")


class TestEvalArmResume:
    @staticmethod
    def _prepare_dirs(tmp_path: Path) -> tuple[Path, Path]:
        raw_dir = tmp_path / "raw" / "tenet"
        mmlu_dir = raw_dir / "mmlu_pro"
        mmlu_dir.mkdir(parents=True)
        (raw_dir / "results_ifeval.json").write_text(
            json.dumps({"results": {"ifeval": {"acc,strict-match": 0.5}}}),
            encoding="utf-8",
        )
        (mmlu_dir / "results_mmlu_pro.json").write_text("{corrupted json", encoding="utf-8")
        return raw_dir, mmlu_dir

    def _kwargs(self, vllm_runner: Any, raw_dir: Path, **overrides: Any) -> dict[str, Any]:
        kwargs = _eval_arm_kwargs(
            vllm_runner,
            raw_dir,
            tasks=("ifeval", "mmlu_pro"),
            evalplus_tasks=(),
            eqbench_tasks=(),
            mmlu_pro_limit=25,
            dry_run=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_resume_only_runs_missing_subtask(self, vllm_runner, tmp_path, monkeypatch):
        raw_dir, mmlu_dir = self._prepare_dirs(tmp_path)
        invoked: list[list[str]] = []

        class _FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **kwargs):
            invoked.append(list(cmd))
            (mmlu_dir / "results_mmlu_pro.json").write_text(
                json.dumps({"results": {"mmlu_pro": {"acc": 0.612}}}),
                encoding="utf-8",
            )
            return _FakeProc()

        monkeypatch.setattr(vllm_runner.subprocess, "run", fake_run)

        record = vllm_runner._eval_arm(
            "tenet", "org/model", **self._kwargs(vllm_runner, raw_dir, resume=True)
        )

        assert record["status"] == "ok"
        assert len(invoked) == 1
        assert invoked[0][invoked[0].index("--output_path") + 1] == str(mmlu_dir)
        assert record["scores"] == {"ifeval_acc": 0.5, "mmlu_pro_acc": 0.612}

    def test_no_resume_reruns_all_subtasks(self, vllm_runner, tmp_path, monkeypatch):
        raw_dir, mmlu_dir = self._prepare_dirs(tmp_path)
        invoked: list[list[str]] = []

        class _FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **kwargs):
            invoked.append(list(cmd))
            if "mmlu_pro" in str(kwargs or "") or cmd[cmd.index("--output_path") + 1].endswith(
                "mmlu_pro"
            ):
                (mmlu_dir / "results_mmlu_pro.json").write_text(
                    json.dumps({"results": {"mmlu_pro": {"acc": 0.612}}}),
                    encoding="utf-8",
                )
            return _FakeProc()

        monkeypatch.setattr(vllm_runner.subprocess, "run", fake_run)

        record = vllm_runner._eval_arm(
            "tenet", "org/model", **self._kwargs(vllm_runner, raw_dir, resume=False)
        )

        assert record["status"] == "ok"
        assert len(invoked) == 2


class TestEvalArmRetries:
    def test_retry_succeeds_with_process_group_cleanup(self, vllm_runner, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw" / "tenet"
        raw_dir.mkdir(parents=True)
        killed: list[tuple[int, Any]] = []
        slept: list[float] = []
        calls: list[list[str]] = []

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                calls.append(list(cmd))
                self.pid = 4242
                self.returncode = 0

            def communicate(self):
                if len(calls) == 1:
                    self.returncode = 1
                    return "", "boom"
                (raw_dir / "results_ifeval.json").write_text(
                    json.dumps({"results": {"ifeval": {"acc,strict-match": 0.7}}}),
                    encoding="utf-8",
                )
                return "", ""

        def fake_killpg(pgid, sig):
            killed.append((pgid, sig))

        def fake_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(vllm_runner.subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(vllm_runner.os, "killpg", fake_killpg)
        monkeypatch.setattr(vllm_runner.time, "sleep", fake_sleep)

        record = vllm_runner._eval_arm(
            "tenet",
            "org/model",
            **_eval_arm_kwargs(
                vllm_runner,
                raw_dir,
                tasks=("ifeval",),
                evalplus_tasks=(),
                eqbench_tasks=(),
                dry_run=False,
                resume=False,
                max_retries=2,
                retry_delay_sec=3.5,
            ),
        )

        assert record["status"] == "ok"
        assert record["scores"] == {"ifeval_acc": 0.7}
        assert len(calls) == 2
        assert killed == [(4242, vllm_runner.signal.SIGKILL)]
        assert slept == [3.5]

    def test_retries_exhausted_reports_error(self, vllm_runner, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw" / "tenet"
        raw_dir.mkdir(parents=True)
        killed: list[tuple[int, Any]] = []
        slept: list[float] = []
        calls: list[list[str]] = []

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                calls.append(list(cmd))
                self.pid = 777
                self.returncode = 0

            def communicate(self):
                self.returncode = 1
                return "", "boom"

        monkeypatch.setattr(vllm_runner.subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(vllm_runner.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        monkeypatch.setattr(vllm_runner.time, "sleep", slept.append)

        record = vllm_runner._eval_arm(
            "tenet",
            "org/model",
            **_eval_arm_kwargs(
                vllm_runner,
                raw_dir,
                tasks=("ifeval",),
                evalplus_tasks=(),
                eqbench_tasks=(),
                dry_run=False,
                resume=False,
                max_retries=1,
                retry_delay_sec=0.0,
            ),
        )

        assert record["status"] == "error"
        assert "boom" in record["error"]
        assert len(calls) == 2
        assert killed == [(777, vllm_runner.signal.SIGKILL)] * 2
        assert slept == [0.0]


# ---------------------------------------------------------------------------
# src/sml/hub/model_card.py
# ---------------------------------------------------------------------------

_FULL_OPEN_EVAL_METRICS = {
    "ifeval_strict": 0.812,
    "math_500": 0.734,
    "gsm8k_cot_strict": 0.901,
    "gpqa_diamond": 0.415,
    "mmlu_pro": 0.623,
    "humaneval_plus": 0.774,
    "mbpp_plus": 0.811,
    "eqbench_creative_writing": 1245.5,
    "slop_score": 3.2,
}


def _frontmatter_yaml(card: str) -> dict[str, Any]:
    import yaml

    lines = card.split("\n")
    assert lines[0] == "---", "model card must start with YAML frontmatter"
    end = lines.index("---", 1)
    return yaml.safe_load("\n".join(lines[1:end]))


class TestGenerateTenetModelCardModelIndex:
    def test_full_metrics_emits_model_index(self, model_card):
        card = model_card.generate_tenet_model_card(
            "Tenet-14B-Qwen3-Unslop-v2",
            "Qwen/Qwen3-14B",
            dict(_FULL_OPEN_EVAL_METRICS),
        )
        meta = _frontmatter_yaml(card)
        assert "model-index" in meta
        index = meta["model-index"]
        assert index[0]["name"] == "Tenet-14B-Qwen3-Unslop-v2"

        results = index[0]["results"]
        datasets = {entry["dataset"]["type"]: entry for entry in results}
        expected_datasets = {
            "google/IFEval",
            "HuggingFaceH4/MATH-500",
            "openai/gsm8k",
            "Idavidrein/gpqa",
            "TIGER-Lab/MMLU-Pro",
            "evalplus/humanevalplus",
            "evalplus/mbppplus",
            "eqbench/creative-writing",
        }
        assert expected_datasets <= set(datasets)
        for ds_type in expected_datasets:
            metrics = datasets[ds_type]["metrics"]
            assert metrics, f"no metrics for {ds_type}"
            for metric in metrics:
                assert isinstance(metric["value"], (int, float))

        values = {
            metric["name"]: metric["value"] for entry in results for metric in entry["metrics"]
        }
        assert values["Prompt-level Strict Accuracy"] == pytest.approx(0.812)
        assert values["Accuracy"] >= 0  # MATH-500 / GPQA / MMLU-Pro / EvalPlus rows

    def test_empty_metrics_omits_model_index(self, model_card):
        card = model_card.generate_tenet_model_card(
            "Tenet-14B-Qwen3-Unslop-v2",
            "Qwen/Qwen3-14B",
            {},
        )
        meta = _frontmatter_yaml(card)
        assert "model-index" not in meta

    def test_non_numeric_metric_is_skipped(self, model_card):
        card = model_card.generate_tenet_model_card(
            "Tenet-14B-Qwen3-Unslop-v2",
            "Qwen/Qwen3-14B",
            {"ifeval_strict": "n/a"},
        )
        meta = _frontmatter_yaml(card)
        assert "model-index" not in meta


# ---------------------------------------------------------------------------
# scripts/hub/sanitize_gguf_roots.py
# ---------------------------------------------------------------------------

_SAMPLE_FILES = [
    "Tenet-14B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf",
    "Tenet-14B-Qwen3-Unslop-v2-cka-q4_k_m.gguf",
    "Tenet-14B-Qwen3-Unslop-v2-f16.gguf",
    "experimental_ablation/Tenet-14B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf",
    "README.md",
]


def _sanitize_module():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:  # pragma: no cover - stub the tiny surface we need
        hf = _stub_module(
            "huggingface_hub",
            CommitOperationCopy=type("CommitOperationCopy", (), {}),
            CommitOperationDelete=type("CommitOperationDelete", (), {}),
            HfApi=type("HfApi", (), {}),
        )
        return (
            _load_module(
                "test_open_evals_sanitize_gguf_roots",
                REPO_ROOT / "scripts" / "hub" / "sanitize_gguf_roots.py",
                stubs={"huggingface_hub": hf},
            ),
            hf.CommitOperationCopy,
            hf.CommitOperationDelete,
        )
    module = _load_module(
        "test_open_evals_sanitize_gguf_roots",
        REPO_ROOT / "scripts" / "hub" / "sanitize_gguf_roots.py",
    )
    return module, None, None


class TestPlanGgufSanitization:
    def test_classification(self):
        module, _, _ = _sanitize_module()
        plan = module.plan_gguf_sanitization(_SAMPLE_FILES)
        assert plan["moves"] == [
            {
                "src": "Tenet-14B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf",
                "dst": "experimental_ablation/Tenet-14B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf",
            }
        ]
        assert plan["kept_root_ggufs"] == [
            "Tenet-14B-Qwen3-Unslop-v2-cka-q4_k_m.gguf",
            "Tenet-14B-Qwen3-Unslop-v2-f16.gguf",
        ]
        assert plan["already_in_subdir"] == [
            "experimental_ablation/Tenet-14B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf"
        ]
        assert plan["skipped"] == ["README.md"]


class _FakeCommit:
    def __init__(self) -> None:
        self.commit_url = "https://huggingface.co/fake/commit/abc"


class _FakeHfApi:
    """Stands in for HfApi; records create_commit calls."""

    instances: list[_FakeHfApi] = []

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.create_commit_calls: list[dict[str, Any]] = []
        _FakeHfApi.instances.append(self)

    def list_repo_files(self, repo_id: str, repo_type: str = "model") -> list[str]:
        assert repo_type == "model"
        return list(_SAMPLE_FILES)

    def create_commit(self, **kwargs: Any) -> _FakeCommit:
        self.create_commit_calls.append(kwargs)
        return _FakeCommit()


@pytest.fixture()
def fake_api():
    _FakeHfApi.instances = []
    api = _FakeHfApi()
    yield api
    _FakeHfApi.instances = []


class TestSanitizeRepoGgufRoot:
    def test_dry_run_default_does_not_commit(self, fake_api):
        module, _, _ = _sanitize_module()
        report = module.sanitize_repo_gguf_root(
            "mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2", api=fake_api
        )
        assert report["dry_run"] is True
        assert report["executed"] is False
        assert report["moves"], "expected planned moves"
        assert fake_api.create_commit_calls == []

    def test_execute_commits_copy_then_delete(self, fake_api):
        module, copy_cls, delete_cls = _sanitize_module()
        if copy_cls is None or delete_cls is None:
            try:
                from huggingface_hub import (  # type: ignore
                    CommitOperationCopy,
                    CommitOperationDelete,
                )

                copy_cls, delete_cls = CommitOperationCopy, CommitOperationDelete
            except ImportError:  # pragma: no cover - stub path
                copy_cls = module.CommitOperationCopy
                delete_cls = module.CommitOperationDelete

        report = module.sanitize_repo_gguf_root(
            "mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2",
            dry_run=False,
            api=fake_api,
        )
        assert report["executed"] is True
        assert report["dry_run"] is False
        assert len(fake_api.create_commit_calls) == 1
        kwargs = fake_api.create_commit_calls[0]
        assert kwargs["repo_id"] == "mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2"
        ops = kwargs["operations"]
        assert len(ops) == 2
        assert isinstance(ops[0], copy_cls)
        assert isinstance(ops[1], delete_cls)
        assert ops[0].src_path_in_repo == _SAMPLE_FILES[0]
        assert ops[0].path_in_repo == f"experimental_ablation/{_SAMPLE_FILES[0]}"
        assert ops[1].path_in_repo == _SAMPLE_FILES[0]
        assert report["commit_url"] == "https://huggingface.co/fake/commit/abc"


class TestSanitizeCliMain:
    def test_main_defaults_to_dry_run(self, fake_api, monkeypatch, capsys):
        module, _, _ = _sanitize_module()
        monkeypatch.setattr(module, "HfApi", _FakeHfApi)
        monkeypatch.setattr(module, "load_hf_token_from_env", lambda start=None: None)
        exit_code = module.main(
            ["--repos", "mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2", "--json"]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "--execute" in out
        # The CLI path must construct its own (faked) HfApi instance.
        assert _FakeHfApi.instances and _FakeHfApi.instances[-1] is not fake_api
        cli_api = _FakeHfApi.instances[-1]
        assert cli_api.create_commit_calls == []
        assert "experimental_ablation" in out
        assert "Tenet-14B-Qwen3-Unslop-v2-i1-cka-q4_k_m.gguf" in out


class TestMainTwoWayReport:
    def test_base_tenet_arms_write_report_with_empty_naive_deltas(
        self, vllm_runner, tmp_path, monkeypatch
    ):
        release_id = "release-2way"
        raw_root = tmp_path / "results" / "external_evals" / release_id / "raw"
        for arm in ("base", "tenet"):
            summary = raw_root / arm / "arm_summary.json"
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text(
                json.dumps(
                    {"arm": arm, "status": "ok", "scores": {"ifeval": 0.5, "math_500": 0.5}}
                ),
                encoding="utf-8",
            )
        monkeypatch.setattr(vllm_runner, "REPO_ROOT", tmp_path)

        exit_code = vllm_runner.main(
            [
                "--release-id",
                release_id,
                "--base-model",
                "org/base",
                "--naive-model",
                "org/naive",
                "--tenet-repo",
                "org/tenet",
                "--arms",
                "base,tenet",
            ]
        )

        assert exit_code == 0
        canonical = tmp_path / "results" / "external_evals" / f"{release_id}.json"
        payload = json.loads(canonical.read_text(encoding="utf-8"))
        assert set(payload["arms"]) == {"base", "tenet"}
        assert payload["deltas_vs_naive"] == {}
        assert payload["deltas_vs_base"] == payload["deltas"]


class TestMainResumeArmSummaryCache:
    @staticmethod
    def _main_argv(release_id: str) -> list[str]:
        return [
            "--release-id",
            release_id,
            "--base-model",
            "org/base",
            "--naive-model",
            "org/naive",
            "--tenet-repo",
            "org/tenet",
            "--arms",
            "tenet",
            "--resume",
        ]

    def test_resume_stale_arm_summary_reruns_arm(self, vllm_runner, tmp_path, monkeypatch):
        release_id = "release-resume-stale"
        raw_dir = tmp_path / "results" / "external_evals" / release_id / "raw" / "tenet"
        raw_dir.mkdir(parents=True)
        summary = raw_dir / "arm_summary.json"
        summary.write_text(
            json.dumps(
                {"arm": "tenet", "status": "ok", "scores": {"ifeval": 0.8, "math_500": 0.0}}
            ),
            encoding="utf-8",
        )
        chat_math = raw_dir / "chat_0shot_math"
        chat_math.mkdir()
        (chat_math / "stale_results.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(vllm_runner, "REPO_ROOT", tmp_path)

        calls: list[tuple[str, str]] = []

        def fake_eval_arm(arm, model, **kwargs):
            # arm_summary.json and the stale chat_0shot_math dir must already
            # be gone by the time the arm is re-evaluated.
            assert not summary.exists()
            assert not chat_math.exists()
            calls.append((arm, model))
            return {
                "arm": arm,
                "model": model,
                "status": "ok",
                "scores": {"ifeval_acc": 0.5, "math_500_acc": 0.4},
                "commands": [],
            }

        monkeypatch.setattr(vllm_runner, "_eval_arm", fake_eval_arm)

        exit_code = vllm_runner.main(self._main_argv(release_id))

        assert exit_code == 0
        assert calls == [("tenet", "org/tenet")]

    def test_resume_healthy_arm_summary_is_reused(self, vllm_runner, tmp_path, monkeypatch):
        release_id = "release-resume-cached"
        raw_dir = tmp_path / "results" / "external_evals" / release_id / "raw" / "tenet"
        raw_dir.mkdir(parents=True)
        summary = raw_dir / "arm_summary.json"
        summary.write_text(
            json.dumps(
                {"arm": "tenet", "status": "ok", "scores": {"ifeval": 0.8, "math_500": 0.5}}
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(vllm_runner, "REPO_ROOT", tmp_path)

        def fail_eval_arm(arm, model, **kwargs):
            raise AssertionError("_eval_arm must not run when arm_summary is healthy")

        monkeypatch.setattr(vllm_runner, "_eval_arm", fail_eval_arm)

        exit_code = vllm_runner.main(self._main_argv(release_id))

        assert exit_code == 0
        assert summary.exists()
