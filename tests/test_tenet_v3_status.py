import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "tenet_v3_status.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("tenet_v3_status", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tenet_v3_status"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_mod()


def test_progress_tqdm_cr(tmp_path):
    text = "garbage\r\n\x1b[2K\rProcessed prompts:  40%|####    | 20/50 [00:10]"
    assert mod.extract_live_progress(text) == "Processed prompts:  40%|####    | 20/50 [00:10]"


def test_progress_various():
    assert mod.extract_live_progress("[supervisor] step done").startswith("[supervisor]")
    assert mod.extract_live_progress("hello\nERROR: boom").startswith("ERROR:")
    assert mod.extract_live_progress("nothing useful") == "(no active log)"


def _mk_foundry(root: Path, name: str, summary, gate, strata):
    d = root / "results" / "tenet_v3_foundry" / name
    d.mkdir(parents=True)
    if summary:
        (d / "foundry_summary.json").write_text(json.dumps(summary))
    if gate:
        (d / "fight_card_gate.json").write_text(json.dumps(gate))
    if strata:
        (d / "strata_bench.json").write_text(json.dumps(strata))


def _mk_ext(root: Path, name: str, scores):
    d = root / "results" / "external_evals"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({"after": {"scores": scores}}))


def test_collect_rows(tmp_path):
    _mk_foundry(
        tmp_path,
        "qwen-mx",
        {"status": "RUNNING", "h21_verified": True, "h22_verified": False},
        {"gate_passed": True, "cot_tag_integrity_rate": 0.9, "length_collapse_rate": 0.1},
        {"passed": True},
    )
    _mk_ext(
        tmp_path,
        "qwen-mx",
        {
            "mmlu_pro": 0.739,
            "hendrycks_math500": 0.61,
            "gsm8k_math_acc": 0.88,
            "gpqa_diamond": 0.42,
            "ifeval_strict": 0.55,
        },
    )
    _mk_foundry(
        tmp_path,
        "tenet-v3-foundry-v1",
        {"status": "done", "h21_verified": True, "h22_verified": True},
        None,
        None,
    )
    _mk_ext(tmp_path, "old-model-v1", {"mmlu_pro": 0.1})

    rows = mod.collect_model_rows(tmp_path / "results")
    names = {r["model"] for r in rows}
    assert "tenet-v3-foundry-v1" not in names
    assert "qwen-mx" in names
    assert "old-model-v1" not in names

    r = next(x for x in rows if x["model"] == "qwen-mx")
    assert r["status"] == "RUNNING"
    assert r["h21_22"] == "✓/✗"
    assert r["gate"] == "pass"
    assert r["strata"] == "pass"
    assert r["mmlu_pro"] == "73.9%"
    assert r["math_500"] == "61.0%"
    assert r["gsm8k"] == "88.0%"
    assert r["gpqa"] == "42.0%"
    assert r["ifeval"] == "55.0%"


def test_render_dashboard(tmp_path, capsys):
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / ".pod_status.json").write_text(json.dumps({"status": "RUNNING"}))
    (tmp_path / "results" / "worker.log").write_text(
        "\r\x1b[2K[supervisor] workers ready\nProcessed prompts: 10%|  5/50 "
    )
    out = mod.render_dashboard(tmp_path / "results")
    assert "Pod status: RUNNING" in out
    assert "Live: Processed prompts: 10%|  5/50" in out
    assert "Model" in out and "MMLU-Pro" in out
