#!/usr/bin/env bash
# Nebius AI Cloud Compute GPU VM & Object Storage lifecycle management.
# Global flags: --dry-run (or env DRY_RUN=1) plans actions without side effects; --help prints usage.
set -euo pipefail
export PATH="$HOME/.nebius/bin:/home/ubuntu/.nebius/bin:/usr/local/bin:$PATH"

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
  setup-sa-auth      Generate service-account credentials and create the tenet-sa CLI profile
  start              Start the GPU instance (Nebius Compute API)
  stop               Stop the GPU instance (SSH shutdown fallback; $0/hr GPU billing stops)
  s3-sync-up         Sync results/ from VM to s3://<NEBIUS_S3_BUCKET>/results
  s3-sync-down       Sync results/ from s3://<NEBIUS_S3_BUCKET>/results to VM
  reconfigure        Reconfigure the VM shell without terraform.tfstate:
                     reconfigure --preset <wave1|wave2> [--platform <platform>]
                       [--preemptible|--on-demand] [--nvme-gib <gib>]
                       [--running|--stopped]
                     Switches 1xH200 <-> 8xH200, billing mode, and grows the
                     NVMe scratch disk; persistent disks are always preserved.
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
# Local destination for s3-sync-down on the workstation (NEBIUS_RESULTS_DIR is a VM path).
LOCAL_RESULTS_DIR="${NEBIUS_LOCAL_RESULTS_DIR:-$(pwd)/results}"
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

SA_CREDENTIALS_FILE="${NEBIUS_SA_CREDENTIALS_FILE:-$HOME/.nebius/tenet-sa-credentials.json}"
SA_PROFILE="tenet-sa"
declare -a NEBIUS_PROFILE_ARGS=()

ensure_nebius_auth() {
  # Use the tenet-sa profile when available so instance commands authenticate
  # without an interactive login or a hand-managed IAM token.
  local has_profile=0
  if [[ -f "$SA_CREDENTIALS_FILE" ]] || nebius profile list 2>/dev/null | grep -q "$SA_PROFILE"; then
    has_profile=1
  fi
  if [[ "$has_profile" == "1" ]]; then
    if [[ -z "${NEBIUS_IAM_TOKEN:-}" ]]; then
      NEBIUS_IAM_TOKEN="$(nebius --profile "$SA_PROFILE" iam get-access-token 2>/dev/null || true)"
      export NEBIUS_IAM_TOKEN
    fi
    NEBIUS_PROFILE_ARGS=(--profile "$SA_PROFILE")
  else
    NEBIUS_PROFILE_ARGS=()
  fi
}

cmd_setup_sa_auth() {
  local sa_id="${NEBIUS_SA_ID:-}"
  if [[ -z "$sa_id" ]]; then
    local tf_dir
    tf_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../infra/terraform/nebius" 2>/dev/null && pwd)"
    if [[ -n "$tf_dir" ]] && command -v terraform >/dev/null 2>&1 && [[ -d "$tf_dir" ]]; then
      sa_id="$( (cd "$tf_dir" && terraform output -raw service_account_id 2>/dev/null) || true)"
    fi
  fi
  if [[ -z "$sa_id" ]] && [[ -n "${NEBIUS_PROJECT_ID:-}" ]]; then
    sa_id="$(nebius iam service-account get-by-name --name tenet-v3-sa --parent-id "$NEBIUS_PROJECT_ID" --format json 2>/dev/null | jq -r .metadata.id || true)"
  fi
  [[ -n "$sa_id" ]] || fail "cannot resolve SA_ID: set NEBIUS_SA_ID or NEBIUS_PROJECT_ID"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN setup-sa-auth: sa_id=$sa_id credentials=$SA_CREDENTIALS_FILE profile=$SA_PROFILE"
    return
  fi

  log "setup-sa-auth: generating service-account credentials for $sa_id"
  nebius iam auth-public-key generate --service-account-id "$sa_id" \
    --output "$SA_CREDENTIALS_FILE" --output-format service-account-json
  chmod 600 "$SA_CREDENTIALS_FILE"

  if nebius profile list 2>/dev/null | grep -q "$SA_PROFILE"; then
    log "setup-sa-auth: profile '$SA_PROFILE' already exists; updating"
    nebius profile update "$SA_PROFILE" --service-account-file "$SA_CREDENTIALS_FILE" --parent-id "$NEBIUS_PROJECT_ID"
  else
    log "setup-sa-auth: creating profile '$SA_PROFILE'"
    nebius profile create "$SA_PROFILE" --endpoint api.nebius.cloud --service-account-file "$SA_CREDENTIALS_FILE" --parent-id "$NEBIUS_PROJECT_ID" </dev/null
  fi
  log "setup-sa-auth: OK (credentials=$SA_CREDENTIALS_FILE)"
}

s3_cmd() {
  if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && [[ ! -f "$HOME/.aws/credentials" ]]; then
    log "s3-sync-${direction:-cmd}: static S3 credentials not set; skipping (persisted via HF Hub & rsync)"
    return 0
  fi
  if command -v s5cmd >/dev/null 2>&1; then
    s5cmd --endpoint-url "$S3_ENDPOINT" "$@"
  else
    aws s3 sync --endpoint-url "$S3_ENDPOINT" "$@" | sed 's/^/[nebius_ctl][aws] /'
  fi
}

cmd_preflight() {
  log "preflight: Nebius GPU VM & storage readiness"
  ensure_nebius_auth

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
    nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute instance get --id "$INSTANCE_ID" >/dev/null \
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
    if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || [[ -f "$HOME/.aws/credentials" ]]; then
      s3_cmd "ls" "s3://$S3_BUCKET" >/dev/null \
        || fail "bucket s3://$S3_BUCKET is not accessible"
      log "s3: bucket s3://$S3_BUCKET OK via $S3_ENDPOINT"
    else
      log "s3: endpoint $S3_ENDPOINT reachable (no static AWS_ACCESS_KEY_ID set; bucket ls skipped)"
    fi
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
  ensure_nebius_auth
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN start: nebius compute instance start --id '$INSTANCE_ID'"
  else
    nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute instance start --id "$INSTANCE_ID"
    log "start: instance $INSTANCE_ID started"
  fi
}

cmd_stop() {
  [[ -n "$INSTANCE_ID" ]] || fail "NEBIUS_INSTANCE_ID is required for stop"
  ensure_nebius_auth
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN stop: nebius compute instance stop --id '$INSTANCE_ID'"
    if [[ -n "$SSH_HOST" ]]; then
      log "DRY-RUN fallback: ssh '$SSH_HOST' sudo shutdown -h now"
    fi
    log "stop: GPU billing drops to \$0/hr; Persistent Boot SSD (incl. .venv) is preserved"
    return
  fi
  if nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute instance stop --id "$INSTANCE_ID"; then
    log "stop: instance $INSTANCE_ID stopped via Nebius API"
  else
    log "stop: local 'nebius compute instance stop' failed; trying service-account profile on the VM"
    if [[ -n "$SSH_HOST" ]] && run_ssh "command -v nebius >/dev/null 2>&1 && nebius --profile tenet-sa compute instance stop --id '$INSTANCE_ID'"; then
      log "stop: instance $INSTANCE_ID stopped via VM-side Nebius CLI (tenet-sa profile)"
    else
      log "stop: Nebius API unavailable; falling back to SSH shutdown"
      run_ssh "sudo shutdown -h now"
      log "stop: shutdown issued over SSH"
    fi
  fi
  log "stop: GPU billing drops to \$0/hr; Persistent Boot SSD (incl. .venv) is preserved"
}

reconfigure_extract_json() {
  # Reconfigure field extraction: fails hard if managed disks are missing so
  # boot/NVMe disks are NEVER orphaned by a shell recreation.
  python3 -c '
import json, shlex, sys
inst = json.loads(sys.argv[1])
spec = inst.get("spec") or {}
meta = inst.get("metadata") or {}

def need(label, value):
    if value is None or value == "":
        sys.exit("reconfigure: missing %s in instance JSON; refusing to proceed" % label)
    return value

parent_id = need("metadata.parent_id", meta.get("parent_id"))
sa_id = need("spec.service_account_id", spec.get("service_account_id"))
boot = spec.get("boot_disk") or {}
boot_disk_id = need("spec.boot_disk.existing_disk.id", (boot.get("existing_disk") or {}).get("id"))
secondary = spec.get("secondary_disks") or [{}]
nvme_disk_id = need("spec.secondary_disks[0].existing_disk.id", (secondary[0].get("existing_disk") or {}).get("id"))
nvme_dev_id = secondary[0].get("device_id") or "nvme-scratch"
ni = (spec.get("network_interfaces") or [{}])[0]
subnet_id = need("spec.network_interfaces[0].subnet_id", ni.get("subnet_id"))
alloc_id = (ni.get("public_ip_address") or {}).get("allocation_id") or ""
cur_state = (inst.get("status") or {}).get("state") or ""
res = spec.get("resources") or {}
cur_platform = res.get("platform") or ""
cur_preset = res.get("preset") or ""
cur_preempt = "1" if spec.get("preemptible") else "0"
for name, value in (
    ("parent_id", parent_id), ("sa_id", sa_id), ("boot_disk_id", boot_disk_id),
    ("nvme_disk_id", nvme_disk_id), ("nvme_dev_id", nvme_dev_id),
    ("subnet_id", subnet_id), ("alloc_id", alloc_id), ("cur_state", cur_state),
    ("cur_platform", cur_platform), ("cur_preset", cur_preset),
    ("cur_preempt", cur_preempt),
):
    print(f"{name}={shlex.quote(str(value))}")
' "$1"
}

cmd_reconfigure() {
  # Reconfigure an existing VM shell (and optionally grow its NVMe disk)
  # WITHOUT terraform.tfstate: stop -> resize disk -> delete shell -> create
  # shell re-attaching the same persistent disks -> update env/hosts files.
  local preset="" platform="gpu-h200-sxm" want_preempt=1 want_stopped=true nvme_gib=""
  local aligned_gib="" target_preset="" preset_name=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --preset) preset="$2"; shift 2 ;;
      --platform) platform="$2"; shift 2 ;;
      --preemptible) want_preempt=1; shift ;;
      --on-demand) want_preempt=0; shift ;;
      --nvme-gib) nvme_gib="$2"; shift 2 ;;
      --running) want_stopped=false; shift ;;
      --stopped) want_stopped=true; shift ;;
      *) fail "unknown reconfigure option: $1" ;;
    esac
  done
  case "$preset" in
    wave1) target_preset="1gpu-16vcpu-200gb"; preset_name="wave1" ;;
    wave2) target_preset="8gpu-128vcpu-1600gb"; preset_name="wave2" ;;
    *) fail "reconfigure: --preset must be wave1 or wave2 (got '${preset:-}')" ;;
  esac
  if [[ -n "$nvme_gib" ]]; then
    [[ "$nvme_gib" =~ ^[0-9]+$ ]] || fail "reconfigure: --nvme-gib must be an integer (GiB)"
    aligned_gib=$(( (nvme_gib / 93) * 93 ))
    if (( aligned_gib < 93 )); then
      aligned_gib=93
    fi
  fi
  [[ -n "$INSTANCE_ID" ]] || fail "NEBIUS_INSTANCE_ID is required for reconfigure"
  ensure_nebius_auth
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN reconfigure: instance=$INSTANCE_ID platform=$platform preset=$target_preset preemptible=$want_preempt nvme_gib=${aligned_gib:-unchanged} stopped=$want_stopped"
    return 0
  fi

  # a) Fetch current instance JSON
  inst_json="$(nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute instance get --id "$INSTANCE_ID" --format json)"

  # b) Validate & extract fields (refuses to continue without managed disk ids)
  eval "$(reconfigure_extract_json "$inst_json")"

  # c) Stop the instance before touching its disks / shell
  if [[ "$cur_state" != "STOPPED" ]]; then
    log "reconfigure: stopping instance $INSTANCE_ID (state=$cur_state)"
    nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute instance stop --id "$INSTANCE_ID"
  fi

  # d) Grow the NVMe scratch disk if requested and needed (never shrink)
  if [[ -n "$aligned_gib" ]]; then
    disk_json="$(nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute disk get --id "$nvme_disk_id" --format json)"
    cur_gib="$(python3 -c '
import json, sys
d = json.loads(sys.argv[1])
size = (d.get("spec") or {}).get("size_gibibytes")
if size is None:
    size = int((d.get("spec") or {}).get("size_bytes") or 0) // 1073741824
print(int(size))
' "$disk_json")"
    if (( aligned_gib > cur_gib )); then
      log "reconfigure: resizing NVMe disk $nvme_disk_id ${cur_gib} -> ${aligned_gib} GiB"
      nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute disk update --id "$nvme_disk_id" --size-gibibytes "$aligned_gib"
    else
      log "reconfigure: NVMe disk already ${cur_gib} GiB (target $aligned_gib); no resize"
    fi
  fi

  # e) Recreate the VM shell only when platform/preset/billing mode changed
  if [[ "$cur_preempt" != "$want_preempt" || "$cur_platform" != "$platform" || "$cur_preset" != "$target_preset" ]]; then
    create_payload="$(python3 -c '
import json, sys
inst = json.loads(sys.argv[1])
(preset_name, platform, target_preset, want_preempt, want_stopped,
 boot_disk_id, nvme_disk_id, nvme_dev_id, subnet_id, alloc_id,
 sa_id, parent_id) = sys.argv[2:14]
spec = inst.get("spec") or {}
payload = {
    "metadata": {"parent_id": parent_id, "name": f"tenet-v3-gpu-{preset_name}"},
    "spec": {
        "service_account_id": sa_id,
        "stopped": want_stopped == "true",
        "recovery_policy": "FAIL" if want_preempt == "1" else "RECOVER",
        "resources": {"platform": platform, "preset": target_preset},
        "boot_disk": {"attach_mode": "READ_WRITE", "existing_disk": {"id": boot_disk_id}},
        "secondary_disks": [
            {"attach_mode": "READ_WRITE", "device_id": nvme_dev_id,
             "existing_disk": {"id": nvme_disk_id}}
        ],
        "network_interfaces": [
            {"name": "eth0", "subnet_id": subnet_id, "ip_address": {},
             "public_ip_address": {"allocation_id": alloc_id, "static": True}}
        ],
        "cloud_init_user_data": spec.get("cloud_init_user_data", ""),
    },
}
if want_preempt == "1":
    payload["spec"]["preemptible"] = {"on_preemption": "STOP"}
print(json.dumps(payload))
' "$inst_json" "$preset_name" "$platform" "$target_preset" "$want_preempt" "$want_stopped" \
  "$boot_disk_id" "$nvme_disk_id" "$nvme_dev_id" "$subnet_id" "$alloc_id" "$sa_id" "$parent_id")"

    log "reconfigure: deleting stopped VM shell $INSTANCE_ID (persistent disks kept)"
    nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute instance delete --id "$INSTANCE_ID"

    log "reconfigure: creating reconfigured VM shell (preset=$target_preset preemptible=$want_preempt stopped=$want_stopped)"
    create_out="$(nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute instance create "$create_payload" --format json)"
    new_id="$(python3 -c '
import json, sys
out = json.loads(sys.argv[1])
print((out.get("metadata") or {}).get("id") or out.get("resource_id") or "")
' "$create_out")"
    if [[ -z "$new_id" ]]; then
      new_id="$(nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute instance get-by-name \
        --parent-id "$parent_id" --name "tenet-v3-gpu-${preset_name}" --format json \
        | python3 -c 'import json, sys; print((json.load(sys.stdin).get("metadata") or {}).get("id") or "")')"
    fi
    [[ -n "$new_id" ]] || fail "reconfigure: could not determine the new instance id"

    python3 - "$new_id" "$parent_id" <<'PYEOF'
import os, re, sys

new_id, parent_id = sys.argv[1], sys.argv[2]


def rewrite(path, replacements, appends):
    """Replace matching lines preserving leading indent; append missing keys."""
    if not os.path.isfile(path):
        return
    st = os.stat(path)
    with open(path) as fh:
        lines = fh.readlines()
    out, anchor_idx, anchor_indent = [], None, ""
    for line in lines:
        new_line = line
        for pat, repl in replacements:
            m = re.match(pat, line)
            if m:
                indent = m.group(1) if m.lastindex else ""
                new_line = f"{indent}{repl}"
                if anchor_idx is None:
                    anchor_idx = len(out) + 1
                    anchor_indent = indent
                break
        out.append(new_line)
    for pat, extra in appends:
        if not any(re.search(pat, l) for l in out):
            if anchor_idx is not None:
                out.insert(anchor_idx, f"{anchor_indent}{extra}")
                anchor_idx += 1
            else:
                out.append(extra)
    with open(path, "w") as fh:
        fh.writelines(out)
    os.chmod(path, st.st_mode)


rewrite(
    os.environ.get("NEBIUS_ENV_FILE", ".env.nebius"),
    [(r"^(\s*)NEBIUS_INSTANCE_ID=.*", f"NEBIUS_INSTANCE_ID={new_id}\n")],
    [(r"^\s*NEBIUS_PROJECT_ID=", f"NEBIUS_PROJECT_ID={parent_id}\n")],
)
rewrite(
    os.environ.get("NEBIUS_ANSIBLE_HOSTS_FILE", "infra/ansible/inventory/hosts.yml"),
    [(r"^(\s*)nebius_instance_id:.*", f"nebius_instance_id: {new_id}\n")],
    [(r"^\s*nebius_project_id:", f"nebius_project_id: {parent_id}\n")],
)
PYEOF

    log "reconfigure: updated env/hosts files with new instance id $new_id"
    INSTANCE_ID="$new_id"
  else
    # f) No shell recreation needed: just start when a running VM is wanted
    if [[ "$want_stopped" == "false" && "$cur_state" == "STOPPED" ]]; then
      log "reconfigure: starting instance $INSTANCE_ID"
      nebius ${NEBIUS_PROFILE_ARGS[@]+"${NEBIUS_PROFILE_ARGS[@]}"} compute instance start --id "$INSTANCE_ID"
    fi
  fi
  log "reconfigure: OK"
}

s3_sync() {
  local direction="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$direction" == "up" ]]; then
      log "DRY-RUN s3-sync-up: '$RESULTS_DIR' -> 's3://$S3_BUCKET/results' (endpoint $S3_ENDPOINT)"
    else
      log "DRY-RUN s3-sync-down: 's3://$S3_BUCKET/results' -> '$LOCAL_RESULTS_DIR' (endpoint $S3_ENDPOINT)"
    fi
    return
  fi
  # Ensure the local destination exists even when the sync itself is skipped.
  if [[ "$direction" == "down" ]]; then
    mkdir -p "$LOCAL_RESULTS_DIR"
  fi
  if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && [[ ! -f "$HOME/.aws/credentials" ]]; then
    log "s3-sync-${direction:-cmd}: static S3 credentials not set; skipping (persisted via HF Hub & rsync)"
    return 0
  fi
  if command -v s5cmd >/dev/null 2>&1; then
    if [[ "$direction" == "up" ]]; then
      s5cmd --endpoint-url "$S3_ENDPOINT" sync "${RESULTS_DIR%/}/*" "s3://$S3_BUCKET/results/"
    else
      mkdir -p "$LOCAL_RESULTS_DIR"
      s5cmd --endpoint-url "$S3_ENDPOINT" sync "s3://$S3_BUCKET/results/*" "${LOCAL_RESULTS_DIR%/}/"
    fi
  else
    if [[ "$direction" == "up" ]]; then
      aws s3 sync --endpoint-url "$S3_ENDPOINT" "$RESULTS_DIR" "s3://$S3_BUCKET/results" | sed 's/^/[nebius_ctl][aws] /'
    else
      mkdir -p "$LOCAL_RESULTS_DIR"
      aws s3 sync --endpoint-url "$S3_ENDPOINT" "s3://$S3_BUCKET/results" "$LOCAL_RESULTS_DIR" | sed 's/^/[nebius_ctl][aws] /'
    fi
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
    local gpu_model="" nvme_mb_s="" uptime_sec=""
    if [[ -n "$SSH_HOST" ]]; then
      gpu_model="$(run_ssh "nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',' -" 2>/dev/null || true)"
      nvme_mb_s="$(run_ssh "python3 -c 'import time; p=\"/mnt/nvme/.bench_tmp\"; open(p,\"wb\").write(b\"0\"*67108864) if True else None; t0=time.perf_counter(); open(p,\"rb\").read(); dt=max(time.perf_counter()-t0,1e-6); import os; os.remove(p); print(round(64.0/dt,1))'" 2>/dev/null || true)"
      uptime_sec="$(run_ssh "awk '{print \$1}' /proc/uptime" 2>/dev/null || true)"
    elif command -v nvidia-smi >/dev/null 2>&1; then
      gpu_model="$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',' - 2>/dev/null || true)"
      uptime_sec="$(awk '{print $1}' /proc/uptime 2>/dev/null || true)"
    fi
    python3 - "$output" "$run_id" "$gpu_model" "$nvme_mb_s" "$uptime_sec" <<'PYEOF'
import json, sys
from datetime import datetime, timezone

output, run_id, gpu_model, nvme_mb_s, uptime_sec = sys.argv[1:6]
uptime_min = round(float(uptime_sec) / 60.0, 2) if uptime_sec else None
nvme_val = float(nvme_mb_s) if nvme_mb_s else None
passport = {
    "run_id": run_id,
    "provider": "nebius",
    "gpu_model": gpu_model or None,
    "nvme_read_mb_s": nvme_val,
    "s3_upload_gb_s": None,
    "instance_uptime_min": uptime_min,
    "estimated_cost_usd": round((uptime_min / 60.0) * 3.50, 2) if uptime_min is not None else None,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "dry_run": False,
}
with open(output, "w") as fh:
    json.dump(passport, fh, indent=2)
    fh.write("\n")
PYEOF
  fi
  log "telemetry-snapshot: wrote $output"
}

case "$SUBCOMMAND" in
  preflight)          cmd_preflight ;;
  start)              cmd_start ;;
  stop)               cmd_stop ;;
  setup-sa-auth)      cmd_setup_sa_auth ;;
  s3-sync-up)         s3_sync up ;;
  s3-sync-down)       s3_sync down ;;
  reconfigure)        cmd_reconfigure "$@" ;;
  telemetry-snapshot) cmd_telemetry_snapshot "$@" ;;
  *)                  fail "unknown subcommand: $SUBCOMMAND (use --help)" ;;
esac
