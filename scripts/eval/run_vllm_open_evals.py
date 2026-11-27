#!/usr/bin/env python3
"""2026-standard 3-way OpenEvals runner: Base vs Naive Abliteration vs Tenet.

Evaluates three model references with ``lm-eval --model vllm`` (or
``local-chat-completions`` for GGUF served behind llama-server / ``vllm serve``):

1. Base model
2. Naive Abliteration control (pre-built abliterated reference)
3. Tenet release

Suite profiles (``--suite full | quick``; ``full`` has NO default limit):
- ``full``:  ifeval, hendrycks_math500, gsm8k_cot, gpqa_diamond_zeroshot,
  mmlu_pro + evalplus (humaneval_plus, mbpp_plus) + eqbench
  (eqbench_creative_writing, slop_score)
- ``quick``: ifeval, gsm8k_cot at limit=40

Reasoning-model handling (Qwen3 etc.): responses are parsed for an unclosed
``<think>`` block and the per-arm ``unclosed_think_rate`` is reported.

Outputs:
- raw lm-eval logs: ``results/external_evals/<release_id>/raw/<arm>/``
- canonical JSON:   ``results/external_evals/<release_id>.json``
- copy:             ``results/tenet_releases/<release_id>/external_benchmarks.json``
- markdown report:  ``results/external_evals/<release_id>/report.md``

This script does NOT perform surgery or load weights itself; it only drives
lm-eval (and optional evalplus/eqbench entrypoints when installed). Use
``--dry-run`` to print the commands without executing anything.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:  # optional dependency: streaming subtask metrics is best-effort
    from sml.tracking.nebius_mlflow import log_open_evals_subtask
except ImportError:  # pragma: no cover - exercised only when src is not importable
    log_open_evals_subtask = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_THINK_START_TOKEN = "<think>"
DEFAULT_THINK_END_TOKEN = "</think>"
DEFAULT_MAX_GEN_TOKS = 8192
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_API_BASE = "http://localhost:8080"

ARMS = ("base", "naive", "tenet")

# Score keys updated together when hendrycks_math500 is recomputed from raw
# samples (cache rescue in main() and _recompute_math500_scores callers).
MATH500_SCORE_KEYS = ("math_500", "hendrycks_math500", "hendrycks_math500_exact_match")

# Zero-shot chat tasks that need the model's chat template with thinking
# disabled (run separately under ``chat_0shot/`` when ``--split-chat-0shot``).
ZERO_SHOT_CHAT_TASKS: tuple[str, ...] = ("ifeval", "hendrycks_math500")
DEFAULT_CHAT_0SHOT_MAX_GEN_TOKS = 1280
DEFAULT_CHAT_0SHOT_MATH_MAX_GEN_TOKS = 2048
DEFAULT_MATH_500_SYSTEM_INSTRUCTION = (
    "Please reason step by step without using $ delimiters for intermediate "
    "steps, and put only your final answer within \\boxed{}."
)

# Reference to the pristine subprocess.run so tests that monkeypatch
# ``subprocess.run`` can be detected (they get the simple code path).
_ORIG_SUBPROCESS_RUN = subprocess.run
_ORIG_SUBPROCESS_POPEN = subprocess.Popen

SUITE_PROFILES: dict[str, dict[str, Any]] = {
    "full": {
        "limit": None,
        "tasks": (
            "ifeval",
            "hendrycks_math500",
            "gsm8k_cot",
            "gpqa_diamond_zeroshot",
            "mmlu_pro",
        ),
        "evalplus_tasks": ("humaneval_plus", "mbpp_plus"),
        "eqbench_tasks": ("eqbench_creative_writing", "slop_score"),
    },
    "quick": {
        "limit": 40,
        "tasks": ("ifeval", "gsm8k_cot"),
        "evalplus_tasks": (),
        "eqbench_tasks": (),
    },
}


# ---------------------------------------------------------------------------
# Pure helpers: reasoning-response parsing
# ---------------------------------------------------------------------------


def parse_thinking_response(
    text: str,
    *,
    think_start_token: str = DEFAULT_THINK_START_TOKEN,
    think_end_token: str = DEFAULT_THINK_END_TOKEN,
) -> tuple[str, bool]:
    """Split a reasoning-model response into (answer, is_unclosed).

    - If ``think_end_token`` is present: split on the *last* occurrence and
      return ``(after.strip(), False)``.
    - Else if ``think_start_token`` is present: the think block never closed,
      return ``("", True)``.
    - Otherwise return ``(text.strip(), False)``.
    """
    if think_end_token and think_end_token in text:
        answer = text.rsplit(think_end_token, 1)[1]
        return answer.strip(), False
    if think_start_token and think_start_token in text:
        return "", True
    return text.strip(), False


def compute_unclosed_think_rate(
    responses: Sequence[str],
    *,
    think_start_token: str = DEFAULT_THINK_START_TOKEN,
    think_end_token: str = DEFAULT_THINK_END_TOKEN,
) -> float:
    """Fraction of responses whose ``<think>`` block never closed (0.0 if empty)."""
    items = list(responses)
    if not items:
        return 0.0
    unclosed = sum(
        1
        for r in items
        if parse_thinking_response(
            r, think_start_token=think_start_token, think_end_token=think_end_token
        )[1]
    )
    return unclosed / len(items)


def _iter_sample_texts(sample: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("resps", "filtered_resps"):
        value = sample.get(key)
        if value is None:
            continue
        for entry in value:
            if isinstance(entry, str):
                texts.append(entry)
            elif isinstance(entry, (list, tuple)):
                texts.extend(e for e in entry if isinstance(e, str))
    return texts


def extract_lm_eval_samples_unclosed_think_rate(
    raw_samples: dict[str, Any] | list[dict[str, Any]],
    *,
    think_start_token: str = DEFAULT_THINK_START_TOKEN,
    think_end_token: str = DEFAULT_THINK_END_TOKEN,
) -> float:
    """Unclosed-think rate over lm-eval ``log_samples`` payloads.

    Accepts a list of sample dicts, a single sample dict, or a results dict
    with a ``samples`` key; reads text from ``resps`` / ``filtered_resps``.
    """
    samples: list[dict[str, Any]] = []
    if isinstance(raw_samples, dict):
        if "resps" in raw_samples or "filtered_resps" in raw_samples:
            samples.append(raw_samples)
        elif isinstance(raw_samples.get("samples"), list):
            samples = [s for s in raw_samples["samples"] if isinstance(s, dict)]
    elif isinstance(raw_samples, list):
        samples = [s for s in raw_samples if isinstance(s, dict)]
    texts: list[str] = []
    for sample in samples:
        texts.extend(_iter_sample_texts(sample))
    return compute_unclosed_think_rate(
        texts, think_start_token=think_start_token, think_end_token=think_end_token
    )


# ---------------------------------------------------------------------------
# Pure helpers: deterministic boxed-answer extraction & math equivalence
# ---------------------------------------------------------------------------


def _last_boxed_only_string(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}`` / ``\\fbox{...}`` (balanced braces)."""
    best: tuple[int, str] | None = None
    for cmd in ("\\boxed", "\\fbox"):
        idx = text.find(cmd)
        while idx != -1:
            i = idx + len(cmd)
            while i < len(text) and text[i] != "{":
                if not text[i].isspace():
                    break
                i += 1
            if i < len(text) and text[i] == "{":
                depth = 0
                j = i
                closed = False
                while j < len(text):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                        if depth == 0:
                            closed = True
                            break
                    j += 1
                # A truncated response may open ``\boxed{`` without closing it;
                # fall back to the remainder of the text so the answer is still
                # recoverable instead of dropping the sample.
                content = text[i + 1 : j] if closed else text[i + 1 :]
                if best is None or idx > best[0]:
                    best = (idx, content)
            idx = text.find(cmd, idx + len(cmd))
    return best[1] if best is not None else None


def _remove_boxed(text: str) -> str:
    """Strip a leading ``\\boxed{`` / ``\\fbox{`` and the trailing ``}``."""
    for cmd in ("\\boxed{", "\\fbox{"):
        if text.startswith(cmd) and text.endswith("}"):
            return text[len(cmd) : -1]
    return text


def _normalize_math_answer(expr: str) -> str:
    """Canonicalize a LaTeX math answer for string comparison."""
    if expr is None:
        return ""
    s = str(expr).strip().replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{[^{}]*\}", "", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = re.sub(r"[,.;:!]+\s*$", "", s.strip())
    s = re.sub(r"\s+", "", s)
    # Rewrite every ``\frac{a}{b}`` occurrence (not just full-string matches)
    # so tuples like ``(3,\frac{\pi}{2})`` canonicalize deterministically.
    for _ in range(3):
        rewritten = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        if rewritten == s:
            break
        s = rewritten
    return s


def _math_answers_equiv(pred: str, target: str) -> bool:
    """Best-effort equivalence check for math answers (lm-eval is_equiv plus fallbacks)."""
    if pred is None or target is None:
        return False
    pred_s, target_s = str(pred).strip(), str(target).strip()
    if pred_s == target_s and pred_s:
        return True
    try:
        from lm_eval.tasks.hendrycks_math.utils import is_equiv

        if is_equiv(pred_s, target_s):
            return True
    except Exception:
        pass
    norm_pred, norm_target = _normalize_math_answer(pred_s), _normalize_math_answer(target_s)
    if norm_pred and norm_pred == norm_target:
        return True

    def _numeric(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            pass
        m = re.fullmatch(r"\((-?[\d.]+)\)/\((-?[\d.]+)\)", value) or re.fullmatch(
            r"(-?[\d.]+)/(-?[\d.]+)", value
        )
        if m and float(m.group(2)) != 0.0:
            return float(m.group(1)) / float(m.group(2))
        return None

    pred_n, target_n = _numeric(norm_pred), _numeric(norm_target)
    if pred_n is not None and target_n is not None:
        return abs(pred_n - target_n) < 1e-6
    return False


def recompute_hendrycks_math500_from_samples(sub_dir: Path) -> float | None:
    """Recompute the hendrycks_math500 exact-match score from raw lm-eval samples.

    lm-eval's hendrycks_math500 extraction slices between the first and last
    ``$`` in a CoT response, which yields 0.0 for responses that use ``$`` in
    intermediate steps. This recomputes the score deterministically from the
    last ``\\boxed{...}`` of each response instead. Returns ``None`` when no
    valid sample rows are found.
    """
    if not sub_dir.is_dir():
        return None
    paths = sorted(sub_dir.rglob("samples_hendrycks_math500_*.jsonl"))
    if not paths:
        return None
    correct = 0
    total = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            target = row.get("target")
            resp: Any = None
            resps = row.get("resps") or []
            if resps and isinstance(resps[0], (list, tuple)) and resps[0]:
                resp = resps[0][0]
            elif resps and isinstance(resps[0], str):
                resp = resps[0]
            if resp is None:
                filtered = row.get("filtered_resps") or []
                if filtered:
                    resp = filtered[0]
            if not isinstance(resp, str) or not isinstance(target, str):
                continue
            total += 1
            boxed = _last_boxed_only_string(resp)
            pred = _remove_boxed(boxed) if boxed is not None else None
            if pred is not None and _math_answers_equiv(pred, target):
                correct += 1
    if total == 0:
        return None
    return round(correct / total, 6)


def _recompute_math500_scores(sub_dir: Path, scores: dict[str, float]) -> None:
    """Upgrade a near-zero math_500 score in-place from raw samples when possible."""
    if "hendrycks_math500" not in scores and "math_500" not in scores:
        return
    current = float(scores.get("math_500", 0.0) or 0.0)
    if current > 0.05:
        return
    recomputed = recompute_hendrycks_math500_from_samples(sub_dir)
    if recomputed is None or recomputed <= 0.0 or recomputed <= current:
        return
    print(
        f"Recomputed hendrycks_math500 from samples under {sub_dir}: {current} -> {recomputed}",
        flush=True,
    )
    for key in MATH500_SCORE_KEYS:
        if key in scores or key == "math_500":
            scores[key] = recomputed


def _load_release_json(path: Path) -> dict[str, Any]:
    """Best-effort JSON loader returning {} for missing/invalid files."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _num(value: Any) -> bool:
    """True for finite real numbers (bools excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


HEADLINE_EXT_KEYS: tuple[str, ...] = (
    "ifeval_strict",
    "ifeval_inst_strict",
    "math_500",
    "gsm8k_cot_strict",
    "gsm8k_math_acc",
    "gpqa_diamond",
    "mmlu_pro",
    "humaneval_pass1",
    "mbpp_pass1",
    "bfcl_tool_acc",
    "eqbench_creative_writing",
    "slop_score",
    "unclosed_think_rate",
)


def _flat_arm_scores(block: Any) -> dict[str, float]:
    """Extract flat numeric headline metrics from an arm payload block."""
    raw_scores = (
        block.get("scores")
        if isinstance(block, dict) and isinstance(block.get("scores"), dict)
        else (block if isinstance(block, dict) else {})
    )
    out = {k: float(raw_scores[k]) for k in HEADLINE_EXT_KEYS if _num(raw_scores.get(k))}
    if isinstance(block, dict) and _num(block.get("unclosed_think_rate")):
        out["unclosed_think_rate"] = float(block["unclosed_think_rate"])
    return out


def recompute_local_eval_releases(
    results_root: Path,
    releases_root: Path | None = None,
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Backfill math_500 scores for existing local release JSONs from raw samples.

    Iterates ``*-v3.json`` / ``*-v2.json`` payloads under ``results_root``,
    recomputes ``hendrycks_math500`` from ``samples_hendrycks_math500_*.jsonl``
    for each arm, and updates arm summaries, payload scores, deltas and
    downstream artifacts (release copies, model index, model cards).
    """
    updated: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return updated
    paths = sorted(
        p
        for p in list(results_root.glob("*-v3.json")) + list(results_root.glob("*-v2.json"))
        if p.is_file()
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        release_id = str(payload.get("release_id") or payload.get("model_id") or path.stem)
        release_raw = results_root / release_id / "raw"
        changed = False
        recomputed_by_arm: dict[str, float] = {}
        for arm in ARMS:
            arm_dir = release_raw / arm
            if not arm_dir.is_dir():
                continue
            sub_dir = arm_dir / "chat_0shot_math"
            if not (sub_dir.is_dir() and list(sub_dir.rglob("samples_hendrycks_math500_*.jsonl"))):
                alt = arm_dir / "chat_0shot"
                if alt.is_dir() and list(alt.rglob("samples_hendrycks_math500_*.jsonl")):
                    sub_dir = alt
                else:
                    continue
            score = recompute_hendrycks_math500_from_samples(sub_dir)
            if score is None or score <= 0.0:
                continue
            recomputed_by_arm[arm] = score
            records: list[Any] = [payload.get("arms", {}).get(arm)]
            if arm == "base":
                records.append(payload.get("before"))
            if arm == "tenet":
                records.append(payload.get("after"))
            for record in records:
                if not isinstance(record, dict):
                    continue
                scores = record.get("scores")
                if not isinstance(scores, dict):
                    continue
                old = float(scores.get("math_500", 0.0) or 0.0)
                if score <= old:
                    continue
                for key in MATH500_SCORE_KEYS:
                    if key in scores or key == "math_500":
                        scores[key] = score
                changed = True
            summary_path = arm_dir / "arm_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    s_scores = summary.get("scores") if isinstance(summary, dict) else None
                    if isinstance(s_scores, dict) and score > float(
                        s_scores.get("math_500", 0.0) or 0.0
                    ):
                        for key in MATH500_SCORE_KEYS:
                            if key in s_scores or key == "math_500":
                                s_scores[key] = score
                        if not dry_run:
                            tmp = summary_path.with_suffix(".json.tmp")
                            tmp.write_text(
                                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8",
                            )
                            os.replace(tmp, summary_path)
                        changed = True
                except (json.JSONDecodeError, OSError):
                    pass
        rel_readme = (
            (releases_root / release_id / "README.md") if releases_root is not None else None
        )
        needs_card_repair = rel_readme is not None and (
            not rel_readme.exists()
            or "Max row crest-factor: n/a" in rel_readme.read_text(encoding="utf-8")
        )
        if not changed and not needs_card_repair:
            continue
        arms = payload.get("arms")
        if isinstance(arms, dict) and {"base", "tenet"} <= set(arms):
            tenet_scores = arms["tenet"].get("scores", {})
            base_scores = arms["base"].get("scores", {})
            delta = {
                k: round(float(tenet_scores[k]) - float(base_scores[k]), 6)
                for k in MATH500_SCORE_KEYS
                if isinstance(tenet_scores.get(k), (int, float))
                and not isinstance(tenet_scores.get(k), bool)
                and isinstance(base_scores.get(k), (int, float))
                and not isinstance(base_scores.get(k), bool)
            }
            for dkey in ("deltas", "deltas_vs_base"):
                deltas = payload.get(dkey)
                if isinstance(deltas, dict):
                    deltas.update(delta)
            naive_scores = arms.get("naive", {}).get("scores", {})
            dvn = payload.get("deltas_vs_naive")
            if isinstance(dvn, dict) and naive_scores:
                dvn.update(
                    {
                        k: round(float(tenet_scores[k]) - float(naive_scores[k]), 6)
                        for k in delta
                        if isinstance(naive_scores.get(k), (int, float))
                        and not isinstance(naive_scores.get(k), bool)
                    }
                )
        ext = payload.get("external_benchmarks")
        if isinstance(ext, dict) and isinstance(arms, dict) and "tenet" in arms:
            ext_scores = ext.get("scores")
            tenet_scores = arms["tenet"].get("scores", {})
            if isinstance(ext_scores, dict):
                for key in MATH500_SCORE_KEYS:
                    if key in ext_scores and key in tenet_scores:
                        ext_scores[key] = tenet_scores[key]
        updated.append(
            {
                "release_id": release_id,
                "path": str(path),
                "recomputed": recomputed_by_arm,
                "dry_run": dry_run,
            }
        )
        if changed and not dry_run:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if releases_root is not None:
                rel_dir = releases_root / release_id
                if rel_dir.is_dir():
                    (rel_dir / "external_benchmarks.json").write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
        generate_tenet_model_card = None  # type: ignore[assignment]
        render_model_index_yaml = None  # type: ignore[assignment]
        with contextlib.suppress(Exception):
            from sml.hub.model_card import generate_tenet_model_card, render_model_index_yaml
        index_model_id = str(payload.get("tenet_model") or payload.get("model_id") or release_id)
        index_metrics = payload.get("after") or payload.get("external_benchmarks") or payload
        if render_model_index_yaml is not None:
            with contextlib.suppress(Exception):
                yaml_text = render_model_index_yaml(index_model_id, index_metrics)
            if yaml_text:
                for root in (results_root, releases_root):
                    if root is None:
                        continue
                    target_dir = root / release_id
                    if target_dir.is_dir():
                        (target_dir / "model_index.yaml").write_text(yaml_text, encoding="utf-8")
        if (
            releases_root is not None
            and (releases_root / release_id).is_dir()
            and generate_tenet_model_card is not None
        ):
            rel_dir = releases_root / release_id
            if (rel_dir / "foundry_summary.json").exists():
                summary = _load_release_json(rel_dir / "foundry_summary.json")
                v3_foundry_summary = _load_release_json(
                    REPO_ROOT / "results" / "tenet_v3_foundry" / release_id / "foundry_summary.json"
                )
                h21 = dict(summary.get("h21") or {})
                if isinstance(v3_foundry_summary.get("h21"), dict) and v3_foundry_summary[
                    "h21"
                ].get("h21_verified"):
                    h21.update(v3_foundry_summary["h21"])

                def _num(value: Any) -> bool:
                    return isinstance(value, (int, float)) and not isinstance(value, bool)

                card_metrics: dict[str, Any] = {}
                flat_candidates = {
                    "max_row_crest_factor": (
                        h21.get("max_row_crest_factor")
                        or summary.get("max_row_crest_factor")
                        or (summary.get("h11") or {}).get("max_row_crest_factor")
                    ),
                    "max_block16_crest_factor": h21.get("max_block16_crest_factor"),
                    "untouched_expert_ratio": h21.get("untouched_expert_ratio"),
                    "router_weight_drift": h21.get("router_weight_drift"),
                    "peak_host_ram_mb": (
                        h21.get("peak_memory_mb") or summary.get("peak_host_ram_mb")
                    ),
                }
                for key, value in flat_candidates.items():
                    if _num(value):
                        card_metrics[key] = value
                strata_data = _load_release_json(rel_dir / "strata_bench.json")
                strata_metrics = (
                    (strata_data.get("report") or {}).get("metrics")
                    or strata_data.get("metrics")
                    or {}
                )
                for key in (
                    "expert_cache_hit_rate",
                    "coactivation_graph_drift",
                    "speculative_acceptance_rate",
                ):
                    if _num(strata_metrics.get(key)):
                        card_metrics[key] = strata_metrics[key]
                flat_ext = _flat_arm_scores(payload.get("after"))
                if flat_ext:
                    card_metrics["external_benchmarks"] = flat_ext
                comparison: dict[str, Any] = {}
                if isinstance(payload.get("before"), dict) and isinstance(
                    payload.get("after"), dict
                ):
                    b_flat = _flat_arm_scores(payload["before"])
                    a_flat = _flat_arm_scores(payload["after"])
                    if b_flat and a_flat:
                        comparison["before"] = b_flat
                        comparison["after"] = a_flat
                        comparison["deltas"] = {
                            k: round(a_flat[k] - b_flat[k], 6) for k in a_flat if k in b_flat
                        }
                if isinstance(payload.get("arms"), dict) and all(
                    a in payload["arms"] for a in ("base", "naive", "tenet")
                ):
                    arms_flat = {
                        a: _flat_arm_scores(payload["arms"][a]) for a in ("base", "naive", "tenet")
                    }
                    if all(arms_flat.values()):
                        comparison["arms"] = arms_flat
                        comparison["deltas_vs_naive"] = {
                            k: round(arms_flat["tenet"][k] - arms_flat["naive"][k], 6)
                            for k in arms_flat["tenet"]
                            if k in arms_flat["naive"]
                        }
                if comparison:
                    card_metrics["external_benchmarks_comparison"] = comparison
                with contextlib.suppress(Exception):
                    card_md = generate_tenet_model_card(
                        str(payload.get("model_id") or release_id),
                        str(
                            payload.get("base_model")
                            or summary.get("plan", {}).get("base_model")
                            or ""
                        ),
                        card_metrics,
                        recipe_metadata=summary.get("plan"),
                    )
                    (rel_dir / "README.md").write_text(card_md, encoding="utf-8")
                    (rel_dir / f"{release_id}.md").write_text(card_md, encoding="utf-8")
    return updated


# ---------------------------------------------------------------------------
# Pure helpers: command building
# ---------------------------------------------------------------------------


def _shell_safe(value: Any) -> str:
    """Render a model_args value as an lm-eval-safe string."""
    text = str(value)
    for ch in (",", "=", "\n"):
        text = text.replace(ch, " ")
    return text.strip()


def build_lm_eval_command(
    model_ref: str,
    *,
    tasks: Sequence[str],
    output_path: Path,
    backend: str = "vllm",
    precision: str = "bf16",
    limit: int | None = None,
    batch_size: int | str = "auto",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    think_end_token: str = DEFAULT_THINK_END_TOKEN,
    api_base: str | None = None,
    apply_chat_template: bool = False,
    enable_thinking: bool | None = None,
    system_instruction: str | None = None,
) -> list[str]:
    """Build the ``lm_eval`` argv for one arm.

    ``backend='vllm'`` runs the in-process vLLM backend (precision bf16/fp8);
    ``backend='openai-compatible'`` runs ``local-chat-completions`` against an
    OpenAI-compatible server (llama-server / ``vllm serve``), which is how GGUF
    precision is evaluated. ``precision='gguf'`` callers should pass
    ``backend='openai-compatible'``.
    """
    cmd = ["lm_eval", "--tasks", ",".join(tasks)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    cmd += ["--batch_size", str(batch_size)]
    cmd += ["--output_path", str(output_path)]
    cmd += ["--log_samples", "--gen_kwargs", f"max_gen_toks={int(max_gen_toks)}"]
    if system_instruction:
        cmd += ["--system_instruction", system_instruction]

    if backend == "vllm":
        dtype = "bfloat16" if precision == "bf16" else precision
        model_args = [
            f"pretrained={_shell_safe(model_ref)}",
            f"dtype={dtype}",
            f"tensor_parallel_size={int(tensor_parallel_size)}",
            f"gpu_memory_utilization={float(gpu_memory_utilization)}",
            f"max_model_len={int(max_model_len)}",
        ]
        if enable_thinking is not None:
            model_args.append(f"enable_thinking={bool(enable_thinking)}")
        if think_end_token and enable_thinking is not False:
            model_args.append(f"think_end_token={think_end_token}")
        cmd += ["--model", "vllm", "--model_args", ",".join(model_args)]
        if apply_chat_template:
            cmd.append("--apply_chat_template")
        return cmd

    if backend == "openai-compatible":
        base = (api_base or DEFAULT_API_BASE).rstrip("/")
        model_args = [
            f"model={_shell_safe(model_ref)}",
            f"base_url={base}/v1",
            "tokenized_requests=False",
        ]
        if think_end_token:
            model_args.append("enable_thinking=True")
        cmd += ["--model", "local-chat-completions", "--model_args", ",".join(model_args)]
        cmd += ["--apply_chat_template"]
        return cmd

    raise ValueError(f"Unsupported backend: {backend!r} (use 'vllm' or 'openai-compatible')")


def build_evalplus_command(
    model_ref: str,
    *,
    tasks: Sequence[str],
    backend: str = "vllm",
    api_base: str | None = None,
) -> list[str]:
    """Build the ``evalplus.evaluate`` argv (best-effort; entrypoint-dependent)."""
    datasets = [t.removesuffix("_plus") for t in tasks]
    cmd = ["evalplus.evaluate", "--model", model_ref, "--datasets", *datasets, "--greedy"]
    if backend == "openai-compatible":
        cmd += ["--backend", "openai-compat", "--base-url", (api_base or DEFAULT_API_BASE)]
    else:
        cmd += ["--backend", "vllm"]
    return cmd


def build_eqbench_command(
    task: str,
    model_ref: str,
    *,
    api_base: str | None = None,
) -> list[str]:
    """Build the EqBench 3 argv for one component (best-effort)."""
    component = task.removeprefix("eqbench_") or task
    return [
        "python",
        "run_eqbench.py",
        "--model",
        model_ref,
        "--components",
        component,
        "--url",
        (api_base or DEFAULT_API_BASE),
    ]


# ---------------------------------------------------------------------------
# Pure helper: reporting
# ---------------------------------------------------------------------------


def _fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_delta(value: Any) -> str:
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    return f"{value:+.4f}"


def format_three_way_markdown_table(payload: dict[str, Any]) -> str:
    """Render Base vs Naive Abliteration vs Tenet metrics as markdown."""
    arms: dict[str, dict[str, Any]] = payload.get("arms", {}) or {}
    base_arm = arms.get("base", {}) or {}
    naive_arm = arms.get("naive", {}) or {}
    tenet_arm = arms.get("tenet", {}) or {}

    keys: list[str] = []
    for key in ("unclosed_think_rate",):
        if any(key in a for a in (base_arm, naive_arm, tenet_arm)) and key not in keys:
            keys.append(key)
    for arm in (base_arm, naive_arm, tenet_arm):
        for key in arm:
            if key in ("status", "elapsed_seconds", "error") or key.startswith("_"):
                continue
            if key not in keys:
                keys.append(key)

    deltas_base = payload.get("deltas_vs_base")
    deltas_naive = payload.get("deltas_vs_naive")
    if deltas_base is None:
        deltas_base = {
            k: round(float(tenet_arm.get(k, 0.0)) - float(base_arm.get(k, 0.0)), 6)
            for k in keys
            if isinstance(tenet_arm.get(k), (int, float))
            and isinstance(base_arm.get(k), (int, float))
        }
    if deltas_naive is None:
        deltas_naive = {
            k: round(float(tenet_arm.get(k, 0.0)) - float(naive_arm.get(k, 0.0)), 6)
            for k in keys
            if isinstance(tenet_arm.get(k), (int, float))
            and isinstance(naive_arm.get(k), (int, float))
        }

    lines = [
        f"# Three-way external eval: "
        f"{payload.get('model_id', payload.get('release_id', 'unknown'))}",
        "",
        f"- Suite: `{payload.get('suite', 'full')}` | Backend: `{payload.get('backend', 'vllm')}`"
        f" | Precision: `{payload.get('precision', 'bf16')}` | Limit: `{payload.get('limit')}`",
        f"- Base: `{payload.get('base_model', '-')}`",
        f"- Naive Abliteration: `{payload.get('naive_model', '-')}`",
        f"- Tenet: `{payload.get('tenet_model', '-')}`",
        f"- Think-end token: `{payload.get('think_end_token', DEFAULT_THINK_END_TOKEN)}`"
        f" | Max gen toks: `{payload.get('max_gen_toks', DEFAULT_MAX_GEN_TOKS)}`",
        "",
        "| Metric | Base | Naive Abliteration | Tenet | Δ Tenet−Base | Δ Tenet−Naive |",
        "|---|---|---|---|---|---|",
    ]
    for key in keys:
        lines.append(
            f"| {key} | {_fmt_value(base_arm.get(key))} | {_fmt_value(naive_arm.get(key))} | "
            f"{_fmt_value(tenet_arm.get(key))} | {_fmt_delta(deltas_base.get(key))} | "
            f"{_fmt_delta(deltas_naive.get(key))} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _resolve_limit(cli_limit: int | None, suite: str) -> int | None:
    if cli_limit is None:
        return SUITE_PROFILES[suite]["limit"]
    return cli_limit if cli_limit > 0 else None


def _resolve_mmlu_pro_limit(cli_limit: int | None) -> int | None:
    """Resolve the mmlu_pro-only limit override (0/negative/absent = full split)."""
    if cli_limit is not None and cli_limit > 0:
        return cli_limit
    return None


def _flatten_lm_eval_scores(results: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task, metrics in (results.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if key in ("alias",):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            val = round(float(value), 6)
            clean = key.split(",")[0]
            scores[f"{task}_{clean}"] = val
            if task == "gsm8k_cot" and key == "exact_match,strict-match":
                scores["gsm8k_cot_strict"] = val
            elif task == "gsm8k_cot" and key == "exact_match,flexible-extract":
                scores["gsm8k_cot_flexible"] = val
                scores["gsm8k_math_acc"] = val
            elif task == "ifeval" and clean == "prompt_level_strict_acc":
                scores["ifeval_strict"] = val
            elif task == "ifeval" and clean == "inst_level_strict_acc":
                scores["ifeval_inst_strict"] = val
            elif task == "hendrycks_math500" and clean == "exact_match":
                scores["math_500"] = val
                scores["hendrycks_math500"] = val
            elif task == "gpqa_diamond_zeroshot" and clean == "acc":
                scores["gpqa_diamond"] = val
                scores["gpqa_diamond_zeroshot"] = val
            elif task == "mmlu_pro" and clean == "exact_match":
                scores["mmlu_pro"] = val
    return scores


def _load_lm_eval_payload(raw_dir: Path) -> dict[str, Any] | None:
    """Load the main lm-eval results JSON (the one holding a ``results`` dict)."""
    candidates = sorted(
        (
            p
            for p in raw_dir.rglob("*.json")
            if not (
                {"mmlu_pro", "chat_0shot", "chat_0shot_math"} & set(p.relative_to(raw_dir).parts)
            )
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            return data
    return None


def _load_sample_dicts(raw_dir: Path, results: dict[str, Any] | None) -> list[dict[str, Any]]:
    embedded = (results or {}).get("samples")
    if isinstance(embedded, list) and embedded:
        return [s for s in embedded if isinstance(s, dict)]
    samples: list[dict[str, Any]] = []
    sample_files = sorted(
        p
        for p in raw_dir.rglob("samples_*.jsonl")
        if not ({"mmlu_pro", "chat_0shot", "chat_0shot_math"} & set(p.relative_to(raw_dir).parts))
    )
    for path in sample_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                samples.append(record)
    return samples


def _find_valid_subtask_payload(sub_dir: Path) -> dict[str, Any] | None:
    """Find a reusable lm-eval ``results_*.json`` payload for ``--resume``.

    Only lm-eval's own ``results_*.json`` files count (``arm_summary.json`` and
    hand-written caches do not). mmlu_pro and chat_0shot payloads live in their
    own subtask directories, so they are excluded by name here.
    """
    for path in sorted(sub_dir.rglob("*results*.json")):
        if {"mmlu_pro", "chat_0shot", "chat_0shot_math"} & set(path.relative_to(sub_dir).parts):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), dict) and data["results"]:
            return data
    return None


class SanityGateError(RuntimeError):
    """Raised when eval scores or response lengths look implausible."""


def validate_open_evals_sanity(
    scores: dict[str, float],
    samples: Sequence[dict[str, Any]] = (),
    *,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
) -> None:
    """Raise ``SanityGateError`` on near-zero scores or runaway responses."""
    if scores.get("math_500", 1.0) <= 0.05 or (
        scores.get("hendrycks_math500_exact_match", 1.0) <= 0.05
    ):
        raise SanityGateError(f"math_500 score near zero: {scores}")
    if scores.get("ifeval_strict", 1.0) <= 0.10 or (
        scores.get("ifeval_prompt_level_strict_acc", 1.0) <= 0.10
    ):
        raise SanityGateError(f"ifeval strict score near zero: {scores}")
    if len(samples) >= 10:
        lengths: list[int] = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            resps = sample.get("resps")
            if (
                isinstance(resps, list)
                and resps
                and isinstance(resps[0], list)
                and resps[0]
                and isinstance(resps[0][0], str)
                and resps[0][0]
            ):
                lengths.append(len(resps[0][0]))
        if lengths:
            mean_chars = sum(lengths) / len(lengths)
            if mean_chars > max_gen_toks * 3.5:
                raise SanityGateError(
                    f"mean response length {mean_chars:.0f} chars exceeds "
                    f"3.5x max_gen_toks ({max_gen_toks * 3.5:.0f})"
                )


def _parse_arms(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated arm subset; dedupe preserving order."""
    seen: list[str] = []
    for part in raw.split(","):
        arm = part.strip()
        if not arm:
            continue
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; valid arms: {', '.join(ARMS)}")
        if arm not in seen:
            seen.append(arm)
    if not seen:
        raise ValueError("No valid arms parsed from --arms")
    return tuple(seen)


def _kill_process_group(pid: int) -> None:
    """Kill a launched subtask's whole process group (vLLM engine included)."""
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pid, signal.SIGKILL)


def _run_subtask_with_cleanup(
    cmd: list[str], *, log_path: Path | None = None
) -> tuple[int, str, str]:
    """Run one lm-eval subtask, killing the process group on failure.

    Returns ``(returncode, stdout, stderr)``. When ``subprocess.run`` has been
    monkeypatched (tests), fall back to the simple ``subprocess.run`` call.
    With the real ``Popen`` path, ``start_new_session=True`` puts lm-eval and
    its ``VLLM::EngineCore`` children in their own process group so a failure
    never leaks VRAM-holding zombies. When ``log_path`` is provided, combined
    output is streamed to that file in real time so live tqdm progress is
    visible on disk.
    """
    if subprocess.run is not _ORIG_SUBPROCESS_RUN:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    if log_path is not None and subprocess.Popen is _ORIG_SUBPROCESS_POPEN:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        proc = None
        try:
            env = {
                **os.environ,
                "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
            }
            with log_path.open("w", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    start_new_session=True,
                )
                returncode = proc.wait()
        except Exception as exc:  # noqa: BLE001 - report any launch/IO failure
            if proc is not None:
                _kill_process_group(proc.pid)
            return 1, "", f"{type(exc).__name__}: {exc}"
        if returncode != 0 and proc is not None:
            _kill_process_group(proc.pid)
        tail = ""
        with contextlib.suppress(OSError):
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        return returncode, "", tail
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate()
        returncode = proc.returncode
    except Exception as exc:  # noqa: BLE001 - report any launch/IO failure
        if proc is not None:
            _kill_process_group(proc.pid)
        return 1, "", f"{type(exc).__name__}: {exc}"
    if returncode != 0:
        _kill_process_group(proc.pid)
    return returncode, stdout or "", stderr or ""


def _run_optional_tool(
    cmd: list[str], *, cwd: Path | None, timeout_s: int = 3600
) -> dict[str, Any]:
    """Run an optional external entrypoint, degrading to a skip record."""
    if shutil.which(cmd[0]) is None:
        return {"status": "skipped", "reason": f"entrypoint not found: {cmd[0]}", "command": cmd}
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "reason": str(exc), "command": cmd}
    if proc.returncode != 0:
        return {
            "status": "error",
            "reason": (proc.stderr or proc.stdout or "")[-2000:],
            "command": cmd,
        }
    return {"status": "ok", "command": cmd, "stdout_tail": (proc.stdout or "")[-2000:]}


def _eval_arm(
    arm: str,
    model_ref: str,
    *,
    tasks: Sequence[str],
    evalplus_tasks: Sequence[str],
    eqbench_tasks: Sequence[str],
    raw_dir: Path,
    limit: int | None,
    mmlu_pro_limit: int | None = None,
    backend: str,
    precision: str,
    batch_size: int | str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_gen_toks: int,
    think_start_token: str,
    think_end_token: str,
    api_base: str | None,
    dry_run: bool,
    resume: bool = True,
    max_retries: int = 2,
    retry_delay_sec: float = 15.0,
    split_chat_0shot: bool = False,
    sanity_gate: bool = False,
    release_id: str | None = None,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _build_cmd(
        run_tasks: Sequence[str],
        output_path: Path,
        run_limit: int | None,
        **overrides: Any,
    ) -> list[str]:
        return build_lm_eval_command(
            model_ref,
            tasks=run_tasks,
            output_path=output_path,
            backend=backend,
            precision=precision,
            limit=run_limit,
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_gen_toks=overrides.pop("max_gen_toks", max_gen_toks),
            think_end_token=think_end_token,
            api_base=api_base,
            **overrides,
        )

    # Optionally split the zero-shot chat tasks into their own run: chat
    # template applied, thinking disabled, capped generation length.
    chat_tasks = tuple(t for t in tasks if t in ZERO_SHOT_CHAT_TASKS) if split_chat_0shot else ()
    non_chat_tasks = tuple(t for t in tasks if t not in chat_tasks)
    runs: list[tuple[Sequence[str], Path, int | None, dict[str, Any]]] = []
    if mmlu_pro_limit is not None and mmlu_pro_limit > 0 and "mmlu_pro" in non_chat_tasks:
        # Split the run: everything except mmlu_pro at the suite limit, and
        # mmlu_pro alone into its own subdirectory with the override limit.
        other_tasks = tuple(t for t in non_chat_tasks if t != "mmlu_pro")
        mmlu_dir = raw_dir / "mmlu_pro"
        if other_tasks:
            runs.append((other_tasks, raw_dir, limit, {}))
        runs.append((("mmlu_pro",), mmlu_dir, mmlu_pro_limit, {}))
    elif non_chat_tasks:
        runs.append((non_chat_tasks, raw_dir, limit, {}))
    if chat_tasks:
        # ifeval keeps the short generation cap and NO system instruction so
        # strict instruction-following constraints are unaffected.
        ifeval_tasks = tuple(t for t in chat_tasks if t != "hendrycks_math500")
        if ifeval_tasks:
            runs.append(
                (
                    ifeval_tasks,
                    raw_dir / "chat_0shot",
                    limit,
                    {
                        "apply_chat_template": True,
                        "enable_thinking": False,
                        "max_gen_toks": min(max_gen_toks, DEFAULT_CHAT_0SHOT_MAX_GEN_TOKS),
                    },
                )
            )
        # hendrycks_math500 gets a CoT system instruction and a larger
        # generation budget so step-by-step reasoning fits.
        if "hendrycks_math500" in chat_tasks:
            runs.append(
                (
                    ("hendrycks_math500",),
                    raw_dir / "chat_0shot_math",
                    limit,
                    {
                        "apply_chat_template": True,
                        "enable_thinking": False,
                        "max_gen_toks": DEFAULT_CHAT_0SHOT_MATH_MAX_GEN_TOKS,
                        "system_instruction": DEFAULT_MATH_500_SYSTEM_INSTRUCTION,
                    },
                )
            )
    cmds = [
        _build_cmd(run_tasks, out_dir, run_limit, **overrides)
        for run_tasks, out_dir, run_limit, overrides in runs
    ]
    sub_dirs = [out_dir for _, out_dir, _, _ in runs]

    record: dict[str, Any] = {"arm": arm, "model": model_ref}
    if cmds:
        record["command"] = cmds[0]
        record["commands"] = cmds
    if dry_run:
        record["status"] = "dry-run"
        return record

    elapsed = 0.0
    scores: dict[str, float] = {}
    samples: list[dict[str, Any]] = []
    lm_eval_version: str | None = None
    payload_found = False
    for cmd, sub_dir in zip(cmds, sub_dirs, strict=True):
        if resume and not dry_run:
            cached_payload = _find_valid_subtask_payload(sub_dir)
            if cached_payload is not None:
                print(f"[{arm}] Resuming: reusing cached results under {sub_dir}", flush=True)
                payload_found = True
                scores.update(_flatten_lm_eval_scores(cached_payload))
                if lm_eval_version is None:
                    lm_eval_version = cached_payload.get("git_hash") or cached_payload.get(
                        "config", {}
                    ).get("lm_eval_version")
                samples.extend(_load_sample_dicts(sub_dir, cached_payload))
                _recompute_math500_scores(sub_dir, scores)
                continue
        print(f"[{arm}] Running: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        attempts = max(1, 1 + max(0, int(max_retries)))
        returncode, stdout, stderr = 1, "", ""
        for attempt in range(attempts):
            returncode, stdout, stderr = _run_subtask_with_cleanup(
                cmd, log_path=sub_dir / "lm_eval_live.log"
            )
            if returncode == 0:
                break
            failure_text = ((stderr or "") + "\n" + (stdout or ""))[-4000:]
            if not failure_text.strip():
                with contextlib.suppress(OSError):
                    failure_text = (sub_dir / "lm_eval_live.log").read_text(
                        encoding="utf-8", errors="replace"
                    )[-4000:]
            if attempt == 0 and any(
                m in failure_text
                for m in (
                    "Inkling2ForConditionalGeneration",
                    "Inkling2LoopForConditionalGeneration",
                    "inkling2_loop",
                    "inkling2",
                    "is not supported for now",
                    "401",
                    "403",
                    "Repository Not Found",
                    "Unauthorized",
                    "Qwen4ExpForConditionalGeneration",
                    "HYV4FlashForConditionalGeneration",
                    "are not supported",
                    "does not recognize",
                )
            ):
                print(
                    f"[{arm}] attempt {attempt + 1}/{attempts} failed (rc={returncode}); "
                    "unsupported architecture — not retrying",
                    flush=True,
                )
                break
            if attempt < attempts - 1:
                print(
                    f"[{arm}] attempt {attempt + 1}/{attempts} failed (rc={returncode}); "
                    f"retrying in {retry_delay_sec}s",
                    flush=True,
                )
                time.sleep(retry_delay_sec)
        elapsed += time.time() - t0
        if returncode != 0:
            record["elapsed_seconds"] = round(elapsed, 1)
            record["status"] = "error"
            record["error"] = (stderr or stdout or "")[-4000:]
            print(f"[{arm}] lm-eval FAILED in {round(elapsed, 1)}s", flush=True)
            return record
        results = _load_lm_eval_payload(sub_dir)
        if results is None:
            print(f"[{arm}] WARNING: no lm-eval results JSON found under {sub_dir}", flush=True)
            continue
        payload_found = True
        sub_scores = _flatten_lm_eval_scores(results)
        # Recompute math_500 BEFORE publishing scores so MLflow and the merged
        # arm scores receive the corrected value, not the broken 0.0.
        _recompute_math500_scores(sub_dir, sub_scores)
        scores.update(sub_scores)
        if log_open_evals_subtask is not None:
            try:
                log_open_evals_subtask(
                    release_id or raw_dir.parent.name,
                    arm,
                    sub_dir.name,
                    sub_scores,
                )
            except Exception as exc:  # best-effort streaming; never break evals
                print(f"[{arm}] WARNING: MLflow subtask logging failed: {exc}", flush=True)
        if lm_eval_version is None:
            lm_eval_version = results.get("git_hash") or results.get("config", {}).get(
                "lm_eval_version"
            )
        samples.extend(_load_sample_dicts(sub_dir, results))

    if sanity_gate and payload_found:
        validate_open_evals_sanity(scores, samples, max_gen_toks=max_gen_toks)

    record["elapsed_seconds"] = round(elapsed, 1)
    record["status"] = "ok"
    if payload_found:
        record["scores"] = scores
        record["lm_eval_version"] = lm_eval_version
        record["num_sampled_responses"] = len(samples)
        record["unclosed_think_rate"] = round(
            extract_lm_eval_samples_unclosed_think_rate(
                samples,
                think_start_token=think_start_token,
                think_end_token=think_end_token,
            ),
            6,
        )
    else:
        record["scores"] = {}
        record["unclosed_think_rate"] = None

    if evalplus_tasks:
        record["evalplus"] = {
            task: _run_optional_tool(
                build_evalplus_command(
                    model_ref, tasks=evalplus_tasks, backend=backend, api_base=api_base
                ),
                cwd=REPO_ROOT,
            )
            for task in evalplus_tasks
        }
    if eqbench_tasks:
        record["eqbench"] = {
            task: _run_optional_tool(
                build_eqbench_command(task, model_ref, api_base=api_base), cwd=REPO_ROOT
            )
            for task in eqbench_tasks
        }
    print(
        f"[{arm}] Done in {elapsed}s: unclosed_think_rate={record.get('unclosed_think_rate')}",
        flush=True,
    )
    return record


def _deltas(tenet: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        k: round(float(tenet[k]) - float(reference[k]), 6)
        for k in tenet
        if k in reference
        and isinstance(tenet[k], (int, float))
        and isinstance(reference[k], (int, float))
        and not isinstance(tenet[k], bool)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="3-way OpenEvals runner (vLLM / OpenAI-compatible)."
    )
    parser.add_argument("--release-id", required=False, default=None)
    parser.add_argument("--base-model", required=False, default=None)
    parser.add_argument(
        "--naive-model",
        required=False,
        default=None,
        help="Naive Abliteration control model ref",
    )
    parser.add_argument("--tenet-repo", required=False, default=None)
    parser.add_argument("--suite", choices=["full", "quick"], default="full")
    parser.add_argument(
        "--limit", type=int, default=None, help="Override suite limit (0/negative = None)"
    )
    parser.add_argument(
        "--mmlu-pro-limit",
        type=int,
        default=None,
        help="Override limit only for mmlu_pro (0/negative = full split)",
    )
    parser.add_argument("--backend", choices=["vllm", "openai-compatible", "hf"], default="vllm")
    parser.add_argument("--precision", choices=["bf16", "fp8", "gguf"], default="bf16")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION
    )
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-gen-toks", type=int, default=DEFAULT_MAX_GEN_TOKS)
    parser.add_argument("--think-end-token", default=DEFAULT_THINK_END_TOKEN)
    parser.add_argument(
        "--arms",
        default="base,naive,tenet",
        help="Comma-separated subset of arms to run (default: base,naive,tenet)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached subtask/arm results when present (default: yes)",
    )
    parser.add_argument(
        "--split-chat-0shot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Split zero-shot chat tasks (ifeval, hendrycks_math500) into a "
            "separate chat-template run with thinking disabled"
        ),
    )
    parser.add_argument(
        "--sanity-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail the arm when scores or response lengths look implausible",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per failed lm-eval subtask (total attempts = 1 + retries)",
    )
    parser.add_argument(
        "--retry-delay-sec",
        type=float,
        default=15.0,
        help="Seconds to sleep between subtask retry attempts",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands and exit without running"
    )
    parser.add_argument(
        "--recompute-local-math500",
        action="store_true",
        help=(
            "Recompute near-zero hendrycks_math500 scores for existing local "
            "release JSONs from raw lm-eval samples and exit"
        ),
    )
    args = parser.parse_args(argv)

    if args.recompute_local_math500:
        updates = recompute_local_eval_releases(
            REPO_ROOT / "results" / "external_evals",
            REPO_ROOT / "results" / "tenet_releases",
            dry_run=args.dry_run,
        )
        for item in updates:
            print(json.dumps(item), flush=True)
        print(f"Recompute complete: {len(updates)} release payload(s) updated.", flush=True)
        return 0

    missing = [
        flag
        for flag, value in (
            ("--release-id", args.release_id),
            ("--base-model", args.base_model),
            ("--naive-model", args.naive_model),
            ("--tenet-repo", args.tenet_repo),
        )
        if not value
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    if args.backend == "hf":
        parser.error("--backend hf is not supported here; use run_external_benchmark_3way.py")
    try:
        selected_arms = _parse_arms(args.arms)
    except ValueError as exc:
        parser.error(str(exc))
        raise SystemExit(2) from None  # unreachable; parser.error exits
    backend = "openai-compatible" if args.precision == "gguf" else args.backend
    if args.precision == "gguf" and args.backend != "openai-compatible":
        print(
            f"precision=gguf -> forcing backend 'openai-compatible' against {args.api_base}",
            flush=True,
        )

    profile = SUITE_PROFILES[args.suite]
    limit = _resolve_limit(args.limit, args.suite)
    mmlu_pro_limit = _resolve_mmlu_pro_limit(args.mmlu_pro_limit)
    tasks = profile["tasks"]
    evalplus_tasks = profile["evalplus_tasks"]
    eqbench_tasks = profile["eqbench_tasks"]

    base_raw = REPO_ROOT / "results" / "external_evals" / args.release_id / "raw"
    arms: dict[str, dict[str, Any]] = {}
    model_refs = {"base": args.base_model, "naive": args.naive_model, "tenet": args.tenet_repo}
    for arm in selected_arms:
        summary_path = base_raw / arm / "arm_summary.json"
        if (
            arm == "naive"
            and model_refs.get("naive") == model_refs.get("base")
            and "base" in arms
            and arms["base"].get("status") == "ok"
            and not args.dry_run
        ):
            print(
                f"[naive] naive_model == base_model ({model_refs['naive']}); "
                "reusing completed base arm scores",
                flush=True,
            )
            arms["naive"] = {**arms["base"], "arm": "naive"}
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = summary_path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(arms["naive"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(tmp_path, summary_path)
            continue
        summary_invalidated = False
        if args.resume and summary_path.exists() and not args.dry_run:
            try:
                cached = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    isinstance(cached, dict)
                    and cached.get("status") == "ok"
                    and cached.get("scores")
                ):
                    cached_scores = cached.get("scores", {})
                    math_score = float(cached_scores.get("math_500", 0.0) or 0.0)
                    math_raw = base_raw / arm / "chat_0shot_math"
                    math_required = "hendrycks_math500" in tasks
                    math_missing = "math_500" not in cached_scores or math_score <= 0.05
                    # Invalidate cached arm_summary whenever math_500 is required
                    # but missing or near-zero, so _eval_arm runs chat_0shot_math
                    # while other cached subtasks remain reusable via --resume.
                    if math_required and math_missing:
                        # Rescue first: if raw math samples exist on disk, the
                        # near-zero score is likely the $-extraction bug, not a
                        # broken run. Recompute from samples and reuse the
                        # cached summary without deleting anything.
                        recomputed: float | None = None
                        for cand in (math_raw, base_raw / arm):
                            if cand.is_dir():
                                recomputed = recompute_hendrycks_math500_from_samples(cand)
                                if recomputed is not None:
                                    break
                        if recomputed is not None and recomputed > 0.05:
                            print(
                                f"[{arm}] arm_summary.json math_500={math_score} "
                                f"rescued from raw samples: {recomputed}; "
                                "reusing cached summary without rerunning",
                                flush=True,
                            )
                            for key in MATH500_SCORE_KEYS:
                                cached_scores[key] = recomputed
                            summary_path.parent.mkdir(parents=True, exist_ok=True)
                            tmp_path = summary_path.with_suffix(".json.tmp")
                            tmp_path.write_text(
                                json.dumps(cached, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8",
                            )
                            os.replace(tmp_path, summary_path)
                            arms[arm] = cached
                            continue
                        # Do not reuse the summary as a whole when math_500 is
                        # absent/near-zero and no samples could rescue it:
                        # drop it, clear a broken math raw dir, and rerun only
                        # chat_0shot_math while other cached subtasks remain
                        # reusable via --resume.
                        print(
                            f"[{arm}] arm_summary.json lacks usable math_500 "
                            f"(score={math_score}); invalidating cache and "
                            "rerunning chat_0shot_math only",
                            flush=True,
                        )
                        summary_path.unlink(missing_ok=True)
                        summary_invalidated = True
                        if math_score <= 0.05:
                            shutil.rmtree(math_raw, ignore_errors=True)
                    else:
                        print(
                            f"[{arm}] Reusing completed arm_summary.json under {summary_path}",
                            flush=True,
                        )
                        arms[arm] = cached
                        continue
            except (json.JSONDecodeError, OSError):
                pass
        arms[arm] = _eval_arm(
            arm,
            model_refs[arm],
            tasks=tasks,
            evalplus_tasks=evalplus_tasks,
            eqbench_tasks=eqbench_tasks,
            raw_dir=base_raw / arm,
            limit=limit,
            mmlu_pro_limit=mmlu_pro_limit,
            backend=backend,
            precision=args.precision,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_gen_toks=args.max_gen_toks,
            think_start_token=DEFAULT_THINK_START_TOKEN,
            think_end_token=args.think_end_token,
            api_base=args.api_base,
            dry_run=args.dry_run,
            resume=args.resume,
            max_retries=args.max_retries,
            retry_delay_sec=args.retry_delay_sec,
            split_chat_0shot=args.split_chat_0shot,
            sanity_gate=args.sanity_gate,
            release_id=args.release_id,
        )
        if not args.dry_run and not summary_invalidated:
            tmp_path = summary_path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(arms[arm], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(tmp_path, summary_path)

    if args.dry_run:
        print("Dry run complete; no lm-eval/vllm processes were launched.", flush=True)
        return 0

    # Partial --arms runs: try to fill missing arms from cached summaries so a
    # complete 3-way report can still be produced.
    for arm in (a for a in ARMS if a not in arms):
        summary_path = base_raw / arm / "arm_summary.json"
        if not summary_path.exists():
            continue
        try:
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(cached, dict) and cached.get("status") == "ok" and cached.get("scores"):
            print(f"[{arm}] Filling missing arm from cached {summary_path}", flush=True)
            arms[arm] = cached

    failed = [arm for arm, rec in arms.items() if rec.get("status") == "error"]
    if failed:
        print(f"ERROR: arms failed: {', '.join(failed)}", flush=True)
        return 1

    required_report_arms = set(selected_arms) | {"base", "tenet"}
    if not required_report_arms.issubset(arms):
        missing = [arm for arm in sorted(required_report_arms) if arm not in arms]
        print(
            f"Partial run: completed arms {sorted(arms)}; missing {missing}; "
            "skipping report (per-arm summaries saved).",
            flush=True,
        )
        return 0

    payload: dict[str, Any] = {
        "model_id": args.release_id,
        "release_id": args.release_id,
        "base_model": args.base_model,
        "naive_model": args.naive_model,
        "tenet_model": args.tenet_repo,
        "suite": args.suite,
        "backend": backend,
        "precision": args.precision,
        "limit": limit,
        "mmlu_pro_limit": mmlu_pro_limit,
        "max_gen_toks": args.max_gen_toks,
        "max_model_len": args.max_model_len,
        "think_end_token": args.think_end_token,
        "api_base": args.api_base if backend == "openai-compatible" else None,
        "tasks": list(tasks),
        "evalplus_tasks": list(evalplus_tasks),
        "eqbench_tasks": list(eqbench_tasks),
        "arms": arms,
        "before": arms["base"],
        "after": arms["tenet"],
        "external_benchmarks": arms["tenet"],
        "deltas": _deltas(arms["tenet"], arms["base"]),
        "deltas_vs_base": _deltas(arms["tenet"], arms["base"]),
        "deltas_vs_naive": _deltas(arms["tenet"], arms["naive"]) if "naive" in arms else {},
    }
    payload["markdown_report"] = format_three_way_markdown_table(payload)

    canonical = REPO_ROOT / "results" / "external_evals" / f"{args.release_id}.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    release_dir = REPO_ROOT / "results" / "tenet_releases" / args.release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "external_benchmarks.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_path = REPO_ROOT / "results" / "external_evals" / args.release_id / "report.md"
    report_path.write_text(format_three_way_markdown_table(payload), encoding="utf-8")
    print(f"Saved canonical evals to {canonical}", flush=True)
    print(f"Saved markdown report to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
