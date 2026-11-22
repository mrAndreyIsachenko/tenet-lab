#!/usr/bin/env python3
"""HF repo hygiene for the OnlyModels.gg auto-crawler.

The OnlyModels.gg auto-crawler builds Fight Card rows directly from each HF
repo's root listing, so the repo root must contain only the recommended GGUF
artifacts: ``*-cka-q4_k_m.gguf`` and ``*-f16.gguf``. Experimental imatrix-overlay
ablations (``*-i1-cka-q4_k_m.gguf``) must live in ``experimental_ablation/``.

This script moves stray root-level experimental GGUFs into the experimental
subfolder using a single ``create_commit`` per repo (copy + delete), defaulting
to a dry run so nothing is written until ``--execute`` / ``--no-dry-run``.

Usage:
    .venv/bin/python scripts/hub/sanitize_gguf_roots.py            # dry run
    .venv/bin/python scripts/hub/sanitize_gguf_roots.py --execute  # write
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi

DEFAULT_REPOS = (
    'mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2',
    'mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2',
)

EXPERIMENTAL_SUBDIR = 'experimental_ablation'

_I1_CKA_SUFFIX = '-i1-cka-q4_k_m.gguf'
_CKA_SUFFIX = '-cka-q4_k_m.gguf'
_F16_SUFFIX = '-f16.gguf'
_GGUF_SUFFIX = '.gguf'


def is_experimental_i1_gguf(filename: str) -> bool:
    """True iff ``filename`` is a root-level experimental i1 CKA GGUF."""
    name = filename.lower()
    return '/' not in filename and name.endswith(_I1_CKA_SUFFIX)


def is_recommended_root_gguf(filename: str) -> bool:
    """True iff ``filename`` is a root-level GGUF the crawler treats as recommended."""
    name = filename.lower()
    if '/' in filename:
        return False
    if name.endswith(_F16_SUFFIX):
        return True
    return name.endswith(_CKA_SUFFIX) and not name.endswith(_I1_CKA_SUFFIX)


def plan_gguf_sanitization(
    files: Sequence[str], *, subdir: str = EXPERIMENTAL_SUBDIR
) -> dict[str, Any]:
    """Classify repo files and plan the root -> experimental subdir moves."""
    prefix = f'{subdir}/'
    moves: list[dict[str, str]] = []
    kept_root_ggufs: list[str] = []
    already_in_subdir: list[str] = []
    skipped: list[str] = []
    for filename in files:
        if filename.startswith(prefix):
            already_in_subdir.append(filename)
        elif is_experimental_i1_gguf(filename):
            moves.append({'src': filename, 'dst': f'{subdir}/{filename}'})
        elif '/' not in filename and filename.lower().endswith(_GGUF_SUFFIX):
            kept_root_ggufs.append(filename)
        else:
            skipped.append(filename)
    return {
        'moves': moves,
        'kept_root_ggufs': kept_root_ggufs,
        'already_in_subdir': already_in_subdir,
        'skipped': skipped,
    }


def sanitize_repo_gguf_root(
    repo_id: str,
    *,
    dry_run: bool = True,
    api: Any = None,
    token: str | None = None,
    subdir: str = EXPERIMENTAL_SUBDIR,
) -> dict[str, Any]:
    """Move stray experimental i1 GGUFs out of a repo root (dry run by default)."""
    if api is None:
        api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type='model')
    plan = plan_gguf_sanitization(files, subdir=subdir)
    moves = plan['moves']
    report: dict[str, Any] = {
        'repo_id': repo_id,
        'dry_run': dry_run,
        'executed': False,
        'moves': moves,
        'kept_root_ggufs': plan['kept_root_ggufs'],
        'already_in_subdir': plan['already_in_subdir'],
        'skipped': plan['skipped'],
        'subdir': subdir,
    }
    if dry_run or not moves:
        return report

    ops: list[Any] = []
    for move in moves:
        ops.append(CommitOperationCopy(src_path_in_repo=move['src'], path_in_repo=move['dst']))
    for move in moves:
        ops.append(CommitOperationDelete(path_in_repo=move['src']))

    commit_message = (
        f'Sanitize GGUF root for OnlyModels.gg crawler: move {len(moves)} '
        f'experimental i1 GGUF file(s) to {subdir}/ '
        '(root keeps only recommended *-cka-q4_k_m.gguf and *-f16.gguf)'
    )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type='model',
        operations=ops,
        commit_message=commit_message,
    )
    report['executed'] = True
    report['commit_url'] = getattr(commit, 'commit_url', None) or getattr(commit, 'url', None)
    return report


def load_hf_token_from_env(start: Path | None = None) -> str | None:
    """Load ``HF_TOKEN`` from a ``.env`` file if present (like create_tenet_collection.py).

    Walks up from the script directory to the repo root; existing environment
    variables always win over ``.env`` values.
    """
    directory = (start or Path(__file__)).resolve().parent
    for candidate in (directory, *directory.parents):
        env_path = candidate / '.env'
        if not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)
        break
    return os.environ.get('HF_TOKEN') or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Move experimental i1 GGUFs out of HF repo roots '
            '(OnlyModels.gg auto-crawler hygiene).'
        )
    )
    parser.add_argument(
        '--repos',
        nargs='+',
        default=list(DEFAULT_REPOS),
        help='HF repo ids to sanitize (default: 14B + 32B Tenet v2 repos).',
    )
    parser.add_argument(
        '--dry-run',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Preview only (default). Use --no-dry-run or --execute to write.',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Shorthand for --no-dry-run: apply the moves via create_commit.',
    )
    parser.add_argument(
        '--subdir',
        default=EXPERIMENTAL_SUBDIR,
        help=f'Target subfolder for experimental GGUFs (default: {EXPERIMENTAL_SUBDIR}).',
    )
    parser.add_argument('--json', action='store_true', help='Also emit the reports as JSON.')
    return parser


def _print_report_summary(report: dict[str, Any]) -> None:
    subdir = report['subdir']
    moves = report['moves']
    kept = report['kept_root_ggufs']
    already = report['already_in_subdir']
    skipped = report['skipped']
    if report['executed']:
        status = f"EXECUTED — commit: {report.get('commit_url') or 'created'}"
    elif report['dry_run']:
        status = 'DRY RUN — no changes written (re-run with --execute to apply)'
    else:
        status = 'NOTHING TO DO — root already clean'
    print(f'    {status}')
    print(
        f'    files: {len(moves)} to move, {len(kept)} root GGUF(s) kept, '
        f'{len(already)} already in {subdir}/, {len(skipped)} skipped (non-GGUF/other)'
    )
    for move in moves:
        print(f"    MOVE  {move['src']} -> {move['dst']}")
    for name in kept:
        tag = '' if is_recommended_root_gguf(name) else '  [WARN: not a recommended root artifact]'
        print(f'    KEEP  {name}{tag}')
    for name in already:
        print(f'    OK    {name}')


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dry_run = args.dry_run and not args.execute
    mode = 'dry-run' if dry_run else 'execute'
    token = load_hf_token_from_env()
    if token is None and not dry_run:
        print('WARNING: no HF_TOKEN found in environment or .env; create_commit may fail.')

    reports: list[dict[str, Any]] = []
    failures = 0
    for repo_id in args.repos:
        print(f'==> {repo_id} [{mode}]')
        try:
            report = sanitize_repo_gguf_root(
                repo_id, dry_run=dry_run, token=token, subdir=args.subdir
            )
        except Exception as exc:  # CLI boundary: report and continue with remaining repos.
            print(f'    ERROR: {exc}')
            failures += 1
            continue
        reports.append(report)
        _print_report_summary(report)

    total_moves = sum(len(r['moves']) for r in reports)
    total_kept = sum(len(r['kept_root_ggufs']) for r in reports)
    total_already = sum(len(r['already_in_subdir']) for r in reports)
    print(
        f'\nSummary [{mode}]: {len(reports)} repo(s) scanned — '
        f'{total_moves} file(s) to move, {total_kept} root GGUF(s) kept, '
        f'{total_already} already in {args.subdir}/, {failures} error(s).'
    )
    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
