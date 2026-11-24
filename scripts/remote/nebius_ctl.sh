#!/usr/bin/env bash
# Nebius AI Cloud Compute GPU VM & Object Storage lifecycle management.
# Global flags: --dry-run (or env DRY_RUN=1) plans actions without side effects; --help prints usage.
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "${1:-}" == "" ]]; then
  cat <<'EOF'
Usage: nebius_ctl.sh [--dry-run] <subcommand>

Global flags:
  --dry-run        Print planned actions only (also enabled via DRY_RUN=1)
  --help           Show this help

Subcommands:
  preflight          Check auth, GPU instance, SSH, NVMe scratch, S3 bucket, MLflow URI
  start              Start the GPU instance (Nebius Compute API)
  stop               Stop the GPU instance (SSH shutdown fallback; $0/hr GPU billing stops)
  s3-sync-up         Sync results/ from VM to s3://<NEBIUS_S3_BUCKET>/results
  s3-sync-down       Sync results/ from s3://<NEBIUS_S3_BUCKET>/results to VM
  telemetry-snapshot [--run-id <id>] [--output <path>]
                     Write an infrastructure telemetry passport JSON

Environment:
  NEBIUS_INSTANCE_ID   Nebius Compute instance id (required for preflight/start/stop)
  NEBIUS_IAM_TOKEN     IAM token alternative to `nebius iam whoami`
  NEBIUS_SSH_HOST      SSH host of the VM (fallback stop, passport checks)
  NEBIUS_S3_BUCKET     Object Storage bucket (default: tenet-v3-artifacts)
  NEBIUS_RESULTS_DIR   Remote results dir (default: /mnt/nvme/tenet-lab/results)
  MLFLOW_TRACKING_URI  Nebius Managed MLflow endpoint
EOF
  exit 0
fi

SUBCOMMAND="${1:-}"
shift || true

S3_ENDPOINT="https://storage.eu-north1.nebius.cloud"
S3_BUCKET="${NEBIUS_S3_BUCKET:-tenet-v3-artifacts}"
RESULTS_DIR="${NEBIUS_RESULTS_DIR:-/mnt/nvme/tenet-lab/results}"
INSTANCE_ID="${NEBIUS_INSTANCE_ID:-}"
SSH_HOST="${NEBIUS_SSH_HOST:-}"

log() { echo "[nebius_ctl] $*"; }
fail() { echo "[nebius_ctl] ERROR: $*" >&2; exit 1; }

run_ssh() {
  # run_ssh <command...> — execute a command inside the VM over SSH (or plan it in dry-run).
  if [[ -z "$SSH_HOST" ]]; then
    fail "NEBIUS_SSH_HOST is not set; cannot run remote command: $*"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN ssh '$SSH_HOST' $*"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" "$*"
  fi
}

s3_cmd() {
  if command -v s5cmd >/dev/null 2>&1; then
    s5cmd --endpoint-url "$S3_ENDPOINT" "$@"
  else
    aws s3 sync --endpoint-url "$S3_ENDPOINT" "$@" | sed 's/^/[nebius_ctl][aws] /'
  fi
}

cmd_preflight() {
  log "preflight: Nebius GPU VM & storage readiness"

  # 1. CLI auth
  if [[ "$DRY_RUN" != "1" ]]; then
    if [[ -n "${NEBIUS_IAM_TOKEN:-}" ]]; then
      log "auth: NEBIUS_IAM_TOKEN is set"
    elif command -v nebius >/dev/null 2>&1 && nebius iam whoami >/dev/null 2>&1; then
      log "auth: 'nebius iam whoami' OK"
    else
      fail "Nebius auth failed: set NEBIUS_IAM_TOKEN or run 'nebius auth'"
    fi
  else
    log "DRY-RUN auth check (nebius iam whoami / NEBIUS_IAM_TOKEN)"
  fi

  # 2. GPU instance status
  if [[ -z "$INSTANCE_ID" ]]; then
    fail "NEBIUS_INSTANCE_ID is required for preflight"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN instance status: nebius compute instance get --id '$INSTANCE_ID'"
  else
    nebius compute instance get --id "$INSTANCE_ID" >/dev/null \
      || fail "instance '$INSTANCE_ID' not found or not accessible"
    log "instance: $INSTANCE_ID is reachable"
  fi

  # 3. GPU passport over SSH
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN GPU check: ssh '$SSH_HOST' nvidia-smi"
  else
    run_ssh nvidia-smi >/dev/null \
      || fail "nvidia-smi is not reachable on the VM"
    log "gpu: nvidia-smi OK"
  fi

  # 4. Fast local NVMe scratch mount (>= 500 GB free)
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN scratch check: /mnt/nvme or /workspace with >= 500 GB free"
  else
    SCRATCH_LINE=$(run_ssh "df -BG /mnt/nvme 2>/dev/null || df -BG /workspace" | tail -1)
    SCRATCH_FREE_GB=$(echo "$SCRATCH_LINE" | awk '{print $4}' | tr -d 'G')
    [[ "$SCRATCH_FREE_GB" =~ ^[0-9]+$ ]] || fail "cannot determine scratch free space"
    if (( SCRATCH_FREE_GB < 500 )); then
      fail "scratch has only ${SCRATCH_FREE_GB} GB free; >= 500 GB required"
    fi
    log "scratch: ${SCRATCH_FREE_GB} GB free (>= 500 GB required)"
  fi

  # 5. Object Storage bucket accessibility
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN S3 check: $S3_ENDPOINT bucket s3://$S3_BUCKET"
  else
    curl -sS -o /dev/null -w '%{http_code}' "$S3_ENDPOINT" >/dev/null \
      || fail "Object Storage endpoint $S3_ENDPOINT is unreachable"
    s3_cmd "ls" "s3://$S3_BUCKET" >/dev/null \
      || fail "bucket s3://$S3_BUCKET is not accessible"
    log "s3: bucket s3://$S3_BUCKET OK via $S3_ENDPOINT"
  fi

  # 6. Managed MLflow
  if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    log "mlflow: MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
  else
    log "mlflow: MLFLOW_TRACKING_URI is not set (Managed MLflow check skipped)"
  fi

  # 7. Skip uv sync in remote harness contexts
  log "env: exporting UV_NO_SYNC=1 SKIP_UV_SYNC=1"
  export UV_NO_SYNC=1
  export SKIP_UV_SYNC=1

  log "preflight: OK"
}

cmd_start() {
  [[ -n "$INSTANCE_ID" ]] || fail "NEBIUS_INSTANCE_ID is required for start"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN start: nebius compute instance start --id '$INSTANCE_ID'"
  else
    nebius compute instance start --id "$INSTANCE_ID"
    log "start: instance $INSTANCE_ID started"
  fi
}

cmd_stop() {
  [[ -n "$INSTANCE_ID" ]] || fail "NEBIUS_INSTANCE_ID is required for stop"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN stop: nebius compute instance stop --id '$INSTANCE_ID'"
    if [[ -n "$SSH_HOST" ]]; then
      log "DRY-RUN fallback: ssh '$SSH_HOST' sudo shutdown -h now"
    fi
    log "stop: GPU billing drops to \$0/hr; Persistent Boot SSD (incl. .venv) is preserved"
    return
  fi
  if nebius compute instance stop --id "$INSTANCE_ID"; then
    log "stop: instance $INSTANCE_ID stopped via Nebius API"
  else
    log "stop: 'nebius compute instance stop' failed; falling back to SSH shutdown"
    run_ssh "sudo shutdown -h now"
    log "stop: shutdown issued over SSH"
  fi
  log "stop: GPU billing drops to \$0/hr; Persistent Boot SSD (incl. .venv) is preserved"
}

s3_sync() {
  local direction="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$direction" == "up" ]]; then
      log "DRY-RUN s3-sync-up: '$RESULTS_DIR' -> 's3://$S3_BUCKET/results' (endpoint $S3_ENDPOINT)"
    else
      log "DRY-RUN s3-sync-down: 's3://$S3_BUCKET/results' -> '$RESULTS_DIR' (endpoint $S3_ENDPOINT)"
    fi
    return
  fi
  if [[ "$direction" == "up" ]]; then
    s3_cmd "$RESULTS_DIR" "s3://$S3_BUCKET/results"
  else
    s3_cmd "s3://$S3_BUCKET/results" "$RESULTS_DIR"
  fi
}

cmd_telemetry_snapshot() {
  local run_id="" output=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-id) run_id="$2"; shift 2 ;;
      --output) output="$2"; shift 2 ;;
      *) fail "unknown telemetry-snapshot option: $1" ;;
    esac
  done
  run_id="${run_id:-$(date -u +%Y%m%dT%H%M%SZ)}"
  output="${output:-results/infra_telemetry/${run_id}.json}"

  log "telemetry-snapshot: run_id=$run_id output=$output"
  mkdir -p "$(dirname "$output")"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN telemetry collection (no live nebius/SSH/S3 calls)"
    python3 - "$output" "$run_id" "$DRY_RUN" <<'PYEOF'
import json, sys
from datetime import datetime, timezone

output, run_id, dry_run = sys.argv[1], sys.argv[2], sys.argv[3]
passport = {
    "run_id": run_id,
    "provider": "nebius",
    "gpu_model": None,
    "nvme_read_mb_s": None,
    "s3_upload_gb_s": None,
    "instance_uptime_min": None,
    "estimated_cost_usd": None,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "dry_run": dry_run == "1",
}
with open(output, "w") as fh:
    json.dump(passport, fh, indent=2)
    fh.write("\n")
PYEOF
  else
    run_ssh "mkdir -p '$RESULTS_DIR'" >/dev/null
    run_ssh "bash -c 'nvidia-smi --query-gpu=name --format=csv,noheader | head -1'" >/dev/null 2>&1 || true
    run_ssh "bash -c 'uptime -s'" >/dev/null 2>&1 || true
    fail "live telemetry collection requires the full probe suite (nvidia-smi/dd/uptime); use --dry-run for a JSON stub"
  fi
  log "telemetry-snapshot: wrote $output"
}

case "$SUBCOMMAND" in
  preflight)          cmd_preflight ;;
  start)              cmd_start ;;
  stop)               cmd_stop ;;
  s3-sync-up)         s3_sync up ;;
  s3-sync-down)       s3_sync down ;;
  telemetry-snapshot) cmd_telemetry_snapshot "$@" ;;
  *)                  fail "unknown subcommand: $SUBCOMMAND (use --help)" ;;
esac
