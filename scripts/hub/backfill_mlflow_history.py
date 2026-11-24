"""Backfill MLflow run history from the local ``results/`` tree.

Discovers tenet releases, external eval JSON files, and h18/h19 hypothesis
summaries, then logs them through ``sml.tracking.nebius_mlflow``. In
``--dry-run`` mode nothing is sent to MLflow and no mirror files are written
(unless ``--mirror-dir`` is explicitly provided); a JSON discovery summary is
printed instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sml.tracking.nebius_mlflow import (  # noqa: E402
    log_foundry_surgery_run,
    log_open_evals_subtask,
    log_strata_bench_run,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _infer_generation(release_name: str, manifest: dict[str, Any] | None) -> str:
    for source in (
        (manifest or {}).get("generation"),
        (manifest or {}).get("recipe", {}).get("generation")
        if isinstance((manifest or {}).get("recipe"), dict)
        else None,
        release_name,
    ):
        if not isinstance(source, str):
            continue
        lower = source.lower()
        if lower in {"v1", "v2", "v3"}:
            return lower
        for gen in ("v3", "v2", "v1"):
            if f"-{gen}" in lower or f"_{gen}" in lower or lower.endswith(gen):
                return gen
    return "v1"


def _numeric_metrics(data: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def backfill_releases(
    results_dir: Path,
    mirror_dir: str | Path | None,
    experiment_name: str | None,
    dry_run: bool,
    summary: dict[str, Any],
) -> None:
    releases_root = results_dir / "tenet_releases"
    if not releases_root.is_dir():
        summary["releases"].append({"error": f"missing directory: {releases_root}"})
        return
    for release_dir in sorted(p for p in releases_root.iterdir() if p.is_dir()):
        release_name = release_dir.name
        foundry = _read_json(release_dir / "foundry_summary.json") or {}
        manifest = _read_json(release_dir / "release_manifest.json") or {}
        recipe = _read_json(release_dir / "lancet_recipe.json") or {}
        quant_recipe = _read_json(release_dir / "cka_quant_recipe.json") or {}

        generation = _infer_generation(release_name, manifest)
        params: dict[str, Any] = {
            "base_model": manifest.get("base_model") or recipe.get("base_model"),
            "generation": generation,
            "selected_layers": recipe.get("selected_layers"),
            "scale": recipe.get("scale"),
            "min_selectivity_threshold": recipe.get("min_selectivity_threshold"),
            "shared_expert_damping": recipe.get("shared_expert_damping"),
            "quant_snap_iters": quant_recipe.get("snap_iters")
            or quant_recipe.get("quant_snap_iters"),
        }
        surgery_metrics = _numeric_metrics(foundry.get("surgery_metrics", foundry))
        gate_metrics = _numeric_metrics(foundry.get("gate_metrics", {}))
        artifacts = [
            str(release_dir / name)
            for name in ("lancet_recipe.json", "cka_quant_recipe.json", "release_manifest.json")
            if (release_dir / name).is_file()
        ]
        record = {
            "release": release_name,
            "generation": generation,
            "surgery_metrics_keys": sorted(surgery_metrics),
            "gate_metrics_keys": sorted(gate_metrics),
            "artifacts": artifacts,
        }
        if dry_run and mirror_dir is None:
            summary["releases"].append(record)
            continue
        tags = {"generation": generation, "experiment_name": experiment_name}
        result = log_foundry_surgery_run(
            release_id=release_name,
            params=params,
            surgery_metrics=surgery_metrics,
            gate_metrics=gate_metrics,
            artifacts_paths=artifacts,
            mirror_dir=mirror_dir,
            tags=tags,
        )
        record["remote_logged"] = result.get("remote_logged")
        record["mirror_path"] = result.get("mirror_path")
        summary["releases"].append(record)


def backfill_external_evals(
    results_dir: Path,
    mirror_dir: str | Path | None,
    experiment_name: str | None,
    dry_run: bool,
    summary: dict[str, Any],
) -> None:
    evals_root = results_dir / "external_evals"
    if not evals_root.is_dir():
        summary["external_evals"].append({"error": f"missing directory: {evals_root}"})
        return
    for eval_path in sorted(evals_root.glob("*.json")):
        data = _read_json(eval_path) or {}
        generation = "v2" if "v2" in eval_path.stem.lower() else "v1"
        arms: dict[str, dict[str, Any]] = {}
        if isinstance(data.get("arms"), dict):
            arms = data["arms"]
        else:
            if isinstance(data.get("after"), dict):
                arms["after"] = data["after"]
            if isinstance(data.get("before"), dict):
                arms["before"] = data["before"]
        tasks = data.get("tasks") if isinstance(data.get("tasks"), dict) else None
        entries: list[dict[str, Any]] = []
        if not arms and tasks:
            arms = {"summary": {"tasks": tasks}}
        for arm, payload in arms.items():
            subtasks = (
                payload.get("tasks")
                if isinstance(payload, dict) and isinstance(payload.get("tasks"), dict)
                else {"summary": payload}
            )
            for subtask_name, scores in subtasks.items():
                score_dict = scores if isinstance(scores, dict) else {"score": scores}
                record = {
                    "eval": eval_path.name,
                    "arm": arm,
                    "subtask": subtask_name,
                    "generation": generation,
                    "scores": _numeric_metrics(score_dict),
                }
                entries.append(record)
                if dry_run and mirror_dir is None:
                    continue
                result = log_open_evals_subtask(
                    release_id=eval_path.stem,
                    arm=arm,
                    subtask_name=subtask_name,
                    scores_dict=score_dict,
                    mirror_dir=mirror_dir,
                    tags={"generation": generation, "experiment_name": experiment_name},
                )
                record["remote_logged"] = result.get("remote_logged")
                record["mirror_path"] = result.get("mirror_path")
        summary["external_evals"].extend(entries)


def backfill_hypotheses(
    results_dir: Path,
    mirror_dir: str | Path | None,
    experiment_name: str | None,
    dry_run: bool,
    summary: dict[str, Any],
) -> None:
    for hypothesis in ("h18", "h19"):
        for hyp_dir in sorted(p for p in results_dir.glob(f"{hypothesis}_*") if p.is_dir()):
            for json_path in sorted(hyp_dir.glob("*.json")):
                data = _read_json(json_path) or {}
                metrics = _numeric_metrics(data.get("summary_metrics", data.get("metrics", data)))
                record = {
                    "hypothesis": hypothesis,
                    "file": str(json_path),
                    "metrics": metrics,
                }
                summary["hypotheses"].append(record)
                if dry_run and mirror_dir is None:
                    continue
                result = log_strata_bench_run(
                    release_id=f"{hypothesis}_{hyp_dir.name}_{json_path.stem}",
                    strata_metrics=metrics,
                    strata_toml_path=None,
                    mirror_dir=mirror_dir,
                    tags={
                        "generation": "v2",
                        "hypothesis": hypothesis,
                        "experiment_name": experiment_name,
                    },
                )
                record["remote_logged"] = result.get("remote_logged")
                record["mirror_path"] = result.get("mirror_path")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill MLflow history from the local results tree."
    )
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--mirror-dir", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    mirror_dir: str | Path | None = args.mirror_dir
    dry_run = args.dry_run

    summary: dict[str, Any] = {
        "results_dir": str(results_dir),
        "dry_run": dry_run,
        "releases": [],
        "external_evals": [],
        "hypotheses": [],
    }
    backfill_releases(results_dir, mirror_dir, args.experiment_name, dry_run, summary)
    backfill_external_evals(results_dir, mirror_dir, args.experiment_name, dry_run, summary)
    backfill_hypotheses(results_dir, mirror_dir, args.experiment_name, dry_run, summary)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
