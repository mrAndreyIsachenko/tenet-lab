"""MLflow tracking helpers with a graceful local JSON mirror fallback.

Remote logging to a Nebius-hosted MLflow server is best-effort: any failure
(missing tracking URI, missing ``mlflow`` package, network/API errors) never
raises. Every call always writes a local JSON mirror record under ``mirror_dir``
so that run history survives offline or degraded environments.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # optional dependency: tracking works only when installed
    import mlflow  # type: ignore

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without mlflow installed
    mlflow = None
    _MLFLOW_AVAILABLE = False

DEFAULT_EXPERIMENT_NAME = "tenet-lab-v1-v2-v3"


def _load_dotenv_if_present(dotenv_path: Path | str = Path(".env")) -> None:
    """Load simple ``KEY=VALUE`` lines from a dotenv file into ``os.environ``.

    Existing environment variables always win; nothing is overwritten.
    """
    path = Path(dotenv_path)
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _mirror_dir(mirror_dir: str | Path | None) -> Path:
    return (
        Path(mirror_dir)
        if mirror_dir is not None
        else Path(os.environ.get("MLFLOW_MIRROR_DIR", "results/mlflow_mirror"))
    )


def _write_mirror(
    mirror_dir: str | Path | None,
    record: dict[str, Any],
) -> Path:
    target = _mirror_dir(mirror_dir)
    target.mkdir(parents=True, exist_ok=True)
    run_name = record.get("run_name", "run")
    release_id = (
        record.get("release_id")
        or record.get("subtask_name")
        or record.get("release_id", "release")
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
    path = target / f"{run_name}_{release_id}_{stamp}.json".replace(" ", "_")
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def _artifact_paths(paths: list[str | Path] | None) -> list[str]:
    if not paths:
        return []
    return [str(Path(p)) for p in paths]


def _resolve_tracking_uri() -> str | None:
    _load_dotenv_if_present()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    return uri or None


def _configure_auth() -> None:
    token = os.environ.get("MLFLOW_TRACKING_TOKEN", "").strip()
    if token:
        os.environ.setdefault("MLFLOW_TRACKING_TOKEN", token)
        return
    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "").strip()
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "").strip()
    if username and password:
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", username)
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", password)
        return
    iam = os.environ.get("NEBIUS_IAM_TOKEN", "").strip()
    if iam and not os.environ.get("MLFLOW_TRACKING_TOKEN"):
        os.environ["MLFLOW_TRACKING_TOKEN"] = iam


def _safe_mlflow_call(fn, *args, **kwargs):
    """Run an MLflow call, returning its result or ``None`` on any failure."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _log_run(
    *,
    run_name: str,
    release_id: str,
    experiment_name: str | None,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None,
    artifacts_paths: list[str | Path] | None,
    mirror_dir: str | Path | None,
) -> dict[str, Any]:
    tracking_uri = _resolve_tracking_uri()
    remote_logged = False

    if tracking_uri and _MLFLOW_AVAILABLE:
        _configure_auth()
        exp_name = experiment_name or os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME
        )
        _safe_mlflow_call(mlflow.set_tracking_uri, tracking_uri)
        exp_id = _safe_mlflow_call(mlflow.set_experiment, exp_name)
        run = _safe_mlflow_call(
            mlflow.start_run,
            run_name=run_name,
            experiment_id=getattr(exp_id, "experiment_id", None),
        )
        if run is not None:
            for key, value in params.items():
                _safe_mlflow_call(mlflow.log_param, key, value)
            for key, value in metrics.items():
                _safe_mlflow_call(mlflow.log_metric, key, float(value))
            if tags:
                _safe_mlflow_call(mlflow.set_tags, tags)
            for artifact in _artifact_paths(artifacts_paths):
                if Path(artifact).is_file():
                    _safe_mlflow_call(mlflow.log_artifact, artifact)
            _safe_mlflow_call(mlflow.end_run, status="FINISHED")
            remote_logged = True

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_name": run_name,
        "release_id": release_id,
        "experiment_name": experiment_name
        or os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
        "params": params,
        "metrics": metrics,
        "tags": tags or {},
        "artifacts": _artifact_paths(artifacts_paths),
        "remote_logged": remote_logged,
    }
    record["mirror_path"] = str(_write_mirror(mirror_dir, record))
    return record


def log_foundry_surgery_run(
    release_id: str,
    params: dict[str, Any],
    surgery_metrics: dict[str, float],
    gate_metrics: dict[str, Any],
    artifacts_paths: list[str | Path] | None = None,
    *,
    mirror_dir: str | Path | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Log a Foundry surgery release: recipe params, surgery + gate metrics, artifacts."""
    run_name = f"foundry_surgery_{release_id}"
    merged_params: dict[str, Any] = {
        "base_model": params.get("base_model"),
        "generation": params.get("generation"),
        "selected_layers": params.get("selected_layers"),
        "scale": params.get("scale"),
        "min_selectivity_threshold": params.get("min_selectivity_threshold"),
        "shared_expert_damping": params.get("shared_expert_damping"),
        "quant_snap_iters": params.get("quant_snap_iters"),
    }
    merged_params.update(
        {k: v for k, v in params.items() if k not in merged_params or merged_params[k] is None}
    )
    metrics: dict[str, float] = {}
    metrics.update(
        {
            k: float(v)
            for k, v in {
                "router_weight_drift": surgery_metrics.get("router_weight_drift"),
                "untouched_expert_ratio": surgery_metrics.get("untouched_expert_ratio"),
                "max_row_crest_factor": surgery_metrics.get("max_row_crest_factor"),
                "max_block16_crest_factor": surgery_metrics.get("max_block16_crest_factor"),
                "fp8_releakage_pct": surgery_metrics.get("fp8_releakage_pct"),
                "drafter_kl_reduction": surgery_metrics.get("drafter_kl_reduction"),
                "peak_host_ram_mb": surgery_metrics.get("peak_host_ram_mb"),
                "wall_time_sec": surgery_metrics.get("wall_time_sec"),
            }.items()
            if v is not None
        }
    )
    metrics.update({k: float(v) for k, v in gate_metrics.items() if isinstance(v, (int, float))})
    return _log_run(
        run_name=run_name,
        release_id=release_id,
        experiment_name=tags.get("experiment_name") if tags else None,
        params=merged_params,
        metrics=metrics,
        tags=tags,
        artifacts_paths=artifacts_paths,
        mirror_dir=mirror_dir,
    )


def log_open_evals_subtask(
    release_id: str,
    arm: str,
    subtask_name: str,
    scores_dict: dict[str, Any],
    *,
    mirror_dir: str | Path | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Stream numeric scores for a single open-evals subtask / arm."""
    run_name = f"open_evals_{release_id}_{arm}_{subtask_name}"
    metrics = {k: float(v) for k, v in scores_dict.items() if isinstance(v, (int, float))}
    merged_tags = dict(tags or {})
    merged_tags.setdefault("arm", arm)
    merged_tags.setdefault("subtask", subtask_name)
    return _log_run(
        run_name=run_name,
        release_id=release_id,
        experiment_name=merged_tags.get("experiment_name"),
        params={"release_id": release_id, "arm": arm, "subtask_name": subtask_name},
        metrics=metrics,
        tags=merged_tags,
        artifacts_paths=None,
        mirror_dir=mirror_dir,
    )


def log_strata_bench_run(
    release_id: str,
    strata_metrics: dict[str, float],
    strata_toml_path: str | Path | None = None,
    *,
    mirror_dir: str | Path | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Log strata benchmark metrics and attach the strata TOML if present."""
    run_name = f"strata_bench_{release_id}"
    metrics = {
        k: float(v)
        for k, v in {
            "ttft_ms": strata_metrics.get("ttft_ms"),
            "decode_tok_per_s": strata_metrics.get("decode_tok_per_s"),
            "expert_cache_hit_rate": strata_metrics.get("expert_cache_hit_rate"),
            "coactivation_graph_drift": strata_metrics.get("coactivation_graph_drift"),
            "speculative_acceptance_rate": strata_metrics.get("speculative_acceptance_rate"),
        }.items()
        if v is not None
    }
    metrics.update(
        {
            k: float(v)
            for k, v in strata_metrics.items()
            if k not in metrics and isinstance(v, (int, float))
        }
    )
    artifacts: list[str | Path] | None = (
        [strata_toml_path] if strata_toml_path and Path(strata_toml_path).is_file() else None
    )
    return _log_run(
        run_name=run_name,
        release_id=release_id,
        experiment_name=tags.get("experiment_name") if tags else None,
        params={"release_id": release_id},
        metrics=metrics,
        tags=tags,
        artifacts_paths=artifacts,
        mirror_dir=mirror_dir,
    )
