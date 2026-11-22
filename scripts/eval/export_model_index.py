"""Export a model_index.yaml from an external evaluation JSON payload.

Usage:
    python scripts/eval/export_model_index.py --eval-json results/external_evals/<release_id>/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sml.hub.model_card import render_model_index_yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render model_index.yaml from an external eval JSON payload.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        required=True,
        help="Path to the external eval JSON payload.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model id for the index (default: payload tenet_model or release_id).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML path (default: results/external_evals/<release_id>/model_index.yaml).",
    )
    parser.add_argument(
        "--also-release-dir",
        action="store_true",
        help="Also write results/tenet_releases/<release_id>/model_index.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rendered YAML instead of writing files.",
    )
    args = parser.parse_args()

    eval_json: Path = args.eval_json
    payload = json.loads(eval_json.read_text(encoding="utf-8"))

    model_id = args.model_id or payload.get("tenet_model") or payload.get("release_id")
    release_id = payload.get("release_id") or model_id

    metrics = payload.get("after") or payload.get("external_benchmarks") or payload

    yaml_text = render_model_index_yaml(model_id, metrics)
    if not yaml_text:
        print(f"render_model_index_yaml produced no content for {eval_json}; nothing written.")
        return 0

    if args.dry_run:
        print(yaml_text)
        return 0

    output: Path = args.output or Path("results/external_evals") / str(release_id) / "model_index.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {output}")

    if args.also_release_dir and release_id:
        release_output = Path("results/tenet_releases") / str(release_id) / "model_index.yaml"
        release_output.parent.mkdir(parents=True, exist_ok=True)
        release_output.write_text(yaml_text, encoding="utf-8")
        print(f"Wrote {release_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
