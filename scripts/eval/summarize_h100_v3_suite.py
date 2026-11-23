#!/usr/bin/env python3
"""Summarize the H100 Tenet v3 suite into a paste-ready Markdown report.

Reads (skipping anything missing):
  - results/external_evals/*.json                (Tenet-*-Unslop-v2, *v3*)
  - results/h18_nvfp4_quant_snap/qwen3_4b_audit.json
  - results/h18_nvfp4_quant_snap/h18_summary.json
  - results/h19_loop_unrolled_lancet/h19_summary.json
  - results/tenet_releases/*v3*/foundry_summary.json

Sections:
  1. Tenet v2 Full vLLM OpenEvals (Base vs Naive vs Tenet v2)
  2. H18 Quant-Snap Audit on Qwen/Qwen3-4B (NVFP4 block-16, INT4 block-32, FP8 per-channel)
  3. H18 & H19 Standard Benchmarks
  4. Tenet v3 (H20 + H18 Quant-Snap) Non-Qwen/Meta Pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

H18_DIR = "h18_nvfp4_quant_snap"
H19_DIR = "h19_loop_unrolled_lancet"
AUDIT_JSON = f"{H18_DIR}/qwen3_4b_audit.json"
H18_SUMMARY = f"{H18_DIR}/h18_summary.json"
H19_SUMMARY = f"{H19_DIR}/h19_summary.json"

MAX_TABLE_COLS = 14


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"_load_error": repr(exc)}


def flatten(obj: Any, prefix: str = "") -> dict[str, float | int | str | bool]:
    """Flatten nested dicts/lists into 'a.b.c' -> scalar for tabulation."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            flat.update(flatten(val, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            flat.update(flatten(val, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        flat[prefix or "value"] = obj
    return flat


def fmt(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}".rstrip("0").rstrip(".") or "0"
    return str(val)


def md_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Rows of flat dicts -> markdown table over the union of scalar keys."""
    flat_rows = [flatten(r) for r in rows]
    if columns is None:
        seen: list[str] = []
        for row in flat_rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen[:MAX_TABLE_COLS]
    cols = [c for c in columns if any(c in r for r in flat_rows)]
    if not cols or not flat_rows:
        return "_No data._\n"
    cols = [c if len(c) <= 32 else c[:29] + "…" for c in cols]
    out = ["| file | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for row in flat_rows:
        cells = [fmt(row.get(c)) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def kv_table(flat: dict[str, Any], title_keys: list[str] | None = None) -> str:
    if not flat:
        return "_No data._\n"
    out = ["| metric | value |", "|---|---|"]
    keys = list(flat)
    if title_keys:
        keys.sort(key=lambda k: (0 if any(t in k for t in title_keys) else 1, k))
    for key in keys:
        out.append(f"| {key} | {fmt(flat[key])} |")
    return "\n".join(out) + "\n"


def section1(results: Path) -> str:
    lines = ["## 1. Tenet v2 Full vLLM OpenEvals (Base vs Naive vs Tenet v2)", ""]
    evals_dir = results / "external_evals"
    files = sorted(evals_dir.glob("*.json")) if evals_dir.exists() else []
    v2 = [p for p in files if "unslop-v2" in p.name.lower()]
    v3 = [p for p in files if "v3" in p.name.lower() and p not in v2]
    if not files:
        lines.append("_No `results/external_evals/*.json` found._\n")
        return "\n".join(lines)

    def role_of(path: Path, data: Any) -> str:
        text = path.name.lower()
        if isinstance(data, dict):
            text += " " + " ".join(str(k).lower() for k in data)
        if "base" in text:
            return "base"
        if "naive" in text:
            return "naive"
        return "tenet"

    rows = []
    for path in files:
        data = load_json(path)
        if data is None:
            continue
        row = {"file": path.name, "role": role_of(path, data)}
        if isinstance(data, dict):
            for key, val in flatten(data).items():
                row[key] = val
            rows.append(row)
    if rows:
        title = "### External eval JSONs" + (" (v2 highlighted, v3 included)" if v2 or v3 else "")
        lines.append(title)
        lines.append("")
        lines.append(md_table(rows, columns=["file", "role"]))
        for path in v2:
            v2_data = load_json(path)
            if isinstance(v2_data, dict) and isinstance(v2_data.get("markdown_report"), str):
                lines.append("")
                lines.append(v2_data["markdown_report"].strip())
                lines.append("")
    else:
        lines.append("_External eval files present but none parseable._\n")
    if v3:
        lines.append("### v3-era external eval files")
        lines.append("")
        lines.append(md_table([{"file": p.name} for p in v3], columns=["file"]))
    return "\n".join(lines)


def section2(results: Path) -> str:
    lines = [
        "## 2. H18 Quant-Snap Audit on `Qwen/Qwen3-4B` "
        "(NVFP4 block-16, INT4 block-32, FP8 per-channel)",
        "",
    ]
    audit = load_json(results / AUDIT_JSON)
    if audit is None:
        lines.append(f"_Missing `{AUDIT_JSON}`._\n")
    else:
        flat = flatten(audit)
        groups: dict[str, dict[str, Any]] = {}
        for key, val in flat.items():
            bucket = "general"
            for tag in ("nvfp4", "int4", "fp8"):
                if tag in key.lower():
                    bucket = tag
                    break
            groups.setdefault(bucket, {})[key] = val
        order = [
            ("NVFP4 block-16", "nvfp4"),
            ("INT4 block-32", "int4"),
            ("FP8 per-channel", "fp8"),
            ("general", "general"),
        ]
        for label, bucket in order:
            if bucket in groups:
                lines.append(f"### {label}")
                lines.append("")
                lines.append(kv_table(groups[bucket]))
                lines.append("")
    summary = load_json(results / H18_SUMMARY)
    lines.append("### H18 summary")
    lines.append("")
    lines.append(
        kv_table(
            flatten(summary) if summary else {}, title_keys=("quant", "snap", "kl", "ppl", "cos")
        )
    )
    lines.append("")
    return "\n".join(lines)


def section3(results: Path) -> str:
    lines = ["## 3. H18 & H19 Standard Benchmarks", ""]
    for label, rel in (("H18", H18_SUMMARY), ("H19", H19_SUMMARY)):
        data = load_json(results / rel)
        lines.append(f"### {label} (`{rel}`)")
        lines.append("")
        if data is None:
            lines.append("_Missing._")
        else:
            lines.append(
                kv_table(
                    flatten(data),
                    title_keys=("mmlu", "gsm8k", "hellaswag", "arc", "winogrande", "ifeval"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def section4(results: Path) -> str:
    lines = ["## 4. Tenet v3 (H20 + H18 Quant-Snap) Non-Qwen/Meta Pilot", ""]
    releases_root = results / "tenet_releases"
    summaries = (
        sorted(releases_root.glob("*v3*/foundry_summary.json")) if releases_root.exists() else []
    )
    if not summaries:
        lines.append(
            "_No `results/tenet_releases/*v3*/foundry_summary.json` found — "
            "run `scripts/run_tenet_v3_pilot.py` first._\n"
        )
        return "\n".join(lines)
    rows = []
    for path in summaries:
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        flat = flatten(data)
        rows.append(
            {
                "release_id": data.get("release_id", path.parent.name),
                "target": data.get("target"),
                "model_id": data.get("model_id"),
                "status": data.get("status"),
                "quant_snap": data.get("quant_snap"),
                "beta": data.get("beta"),
                "eval_suite": data.get("eval_suite"),
                "_path": str(path.relative_to(results)),
                "_flat": flat,
            }
        )
    pilot_cols = ["release_id", "target", "model_id", "status", "quant_snap", "beta", "eval_suite"]
    lines.append("### Pilot releases")
    lines.append("")
    lines.append(md_table(rows, columns=pilot_cols))
    lines.append("")
    for row in rows:
        lines.append(f"#### `{row['release_id']}` (full summary)")
        lines.append("")
        lines.append(kv_table(row["_flat"]))
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize H100 Tenet v2/v3 + H18/H19 results into Markdown"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/h100_v3_suite_summary.md")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the report to stdout instead of writing --output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = Path(args.results_dir)
    if not results.is_absolute():
        results = REPO_ROOT / results

    report = [
        "# H100 Suite Summary — Tenet v2 → v3 (H18/H19/H20)",
        "",
        f"_Generated from `{results.name}/` — missing sources are skipped._",
        "",
        section1(results),
        section2(results),
        section3(results),
        section4(results),
    ]
    text = "\n".join(report)
    if args.dry_run:
        print(text)
        return 0
    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
