#!/usr/bin/env bash
# H100 Tenet v3 suite driver: eval_v2 -> h18_audit -> h18_h19_bench -> tenet_v3.
#
# Usage:
#   bash scripts/remote/run_h100_v3_suite.sh [options]
#
# Options:
#   --steps <csv> | --steps=<csv>   Comma-separated subset of
#                                   eval_v2,h18_audit,h18_h19_bench,tenet_v3
#                                   (default: all four, in that order).
#   --dry-run                       Print commands without executing any
#                                   GPU / network / uv sync steps.
#   --auto-stop-pod                 Stop the RunPod pod (runpodctl stop pod)
#                                   after the suite finishes.
#   -h | --help                     Print this usage text and exit 0.
#
# Environment knobs (see defaults below): HF_NAMESPACE, BATCH_SIZE,
# MAX_GEN_TOKS, THINK_END_TOKEN, DENSE_THINK_END_TOKEN, MMLU_PRO_LIMIT,
# NUM_GPUS, FORCE_RERUN, KEEP_HF_CACHE, UPLOAD_TO_HF,
# NAIVE_4B_MODEL, NAIVE_8B_MODEL, NAIVE_32B_MODEL, TENET_V3_MOE_MODEL,
# TENET_V3_MOE_RELEASE, TENET_V3_DENSE_MODEL, TENET_V3_DENSE_RELEASE,
# V3_EVAL_SUITE.
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment knobs
# ---------------------------------------------------------------------------
HF_NAMESPACE="${HF_NAMESPACE:-mr-andrey-isachenko}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-8192}"
THINK_END_TOKEN="${THINK_END_TOKEN:-</think>}"
FORCE_RERUN="${FORCE_RERUN:-0}"
KEEP_HF_CACHE="${KEEP_HF_CACHE:-0}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-0}"
NAIVE_4B_MODEL="${NAIVE_4B_MODEL:-huihui-ai/Qwen3-4B-abliterated}"
NAIVE_8B_MODEL="${NAIVE_8B_MODEL:-huihui-ai/Qwen3-8B-abliterated}"
NAIVE_32B_MODEL="${NAIVE_32B_MODEL:-huihui-ai/Qwen3-32B-abliterated}"
TENET_V3_MOE_MODEL="${TENET_V3_MOE_MODEL:-google/gemma-4-26B-A4B-it}"
TENET_V3_MOE_RELEASE="${TENET_V3_MOE_RELEASE:-Tenet-26B-A4B-Gemma4-Unslop-v2}"
TENET_V3_DENSE_MODEL="${TENET_V3_DENSE_MODEL:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
TENET_V3_DENSE_RELEASE="${TENET_V3_DENSE_RELEASE:-Tenet-24B-Mistral3.1-Unslop-v2}"
MMLU_PRO_LIMIT="${MMLU_PRO_LIMIT:-0}"
DENSE_THINK_END_TOKEN="${DENSE_THINK_END_TOKEN:-}"
NUM_GPUS="${NUM_GPUS:-1}"
V3_EVAL_SUITE="${V3_EVAL_SUITE:-full}"
SKIP_UV_SYNC="${SKIP_UV_SYNC:-0}"
AUTO_STOP_POD="${AUTO_STOP_POD:-0}"

export HF_NAMESPACE BATCH_SIZE MAX_GEN_TOKS THINK_END_TOKEN \
  DENSE_THINK_END_TOKEN MMLU_PRO_LIMIT NUM_GPUS FORCE_RERUN \
  KEEP_HF_CACHE UPLOAD_TO_HF NAIVE_4B_MODEL NAIVE_8B_MODEL NAIVE_32B_MODEL \
  TENET_V3_MOE_MODEL TENET_V3_MOE_RELEASE TENET_V3_DENSE_MODEL \
  TENET_V3_DENSE_RELEASE V3_EVAL_SUITE SKIP_UV_SYNC AUTO_STOP_POD

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
DRY_RUN=0
STEPS="h18_audit,h18_h19_bench,eval_v2,tenet_v3"
ALL_STEPS=(eval_v2 h18_audit h18_h19_bench tenet_v3)

usage() {
  cat <<'EOF'
Usage: scripts/remote/run_h100_v3_suite.sh [options]

Run the H100 Tenet v3 suite end to end:
  1. eval_v2         - full vLLM open evals for Tenet-*-Qwen3-Unslop-v2 (4B/8B/32B)
  2. h18_audit       - H18 NVFP4/INT4/FP8 shard-streaming audit on Qwen/Qwen3-4B
  3. h18_h19_bench   - H18 + H19 standard benchmark runs via `sml run`
  4. tenet_v3        - H20 pilot releases (MoE Gemma4-26B + dense Mistral-Small-3.1-24B)
  final digest       - results/h100_v3_suite_summary.md via summarize_h100_v3_suite.py

Options:
  --steps <csv>       Comma-separated subset of
                      eval_v2,h18_audit,h18_h19_bench,tenet_v3
                      (default: all four, in the order above).
  --steps=<csv>       Same as above, `=`-separated form.
  --dry-run           Print every command prefixed with "[dry-run]" without
                      executing any GPU / network / uv sync steps.
  --auto-stop-pod     After the suite finishes, stop the RunPod pod via
                      `runpodctl stop pod` (only when RUNPOD_POD_ID is set).
  -h, --help          Print this help and exit 0.

Environment knobs (defaults):
  HF_NAMESPACE            mr-andrey-isachenko
  BATCH_SIZE              auto
  MAX_GEN_TOKS            8192
  THINK_END_TOKEN         </think>
  DENSE_THINK_END_TOKEN   (empty; --think-end-token for the dense, non-thinking
                          Mistral v3 pilot)
  MMLU_PRO_LIMIT          0     (0 = run mmlu_pro at the suite limit; >0 adds
                                 --mmlu-pro-limit to eval_v2 runs)
  NUM_GPUS                1     (2 = run eval_v2 / tenet_v3 workers in
                                 parallel across GPU 0 and GPU 1)
  FORCE_RERUN             0
  KEEP_HF_CACHE           0     (1 = skip `cleanup_hf_model_cache` calls)
  UPLOAD_TO_HF            0     (1 = add --upload-to-hf to the tenet_v3 pilot)
  NAIVE_4B_MODEL          huihui-ai/Qwen3-4B-abliterated
  NAIVE_8B_MODEL          huihui-ai/Qwen3-8B-abliterated
  NAIVE_32B_MODEL         huihui-ai/Qwen3-32B-abliterated
  TENET_V3_MOE_MODEL      google/gemma-4-26B-A4B-it
  TENET_V3_MOE_RELEASE    Tenet-26B-A4B-Gemma4-Unslop-v2
  TENET_V3_DENSE_MODEL    mistralai/Mistral-Small-3.1-24B-Instruct-2503
  TENET_V3_DENSE_RELEASE  Tenet-24B-Mistral3.1-Unslop-v2
  V3_EVAL_SUITE           full
  AUTO_STOP_POD           0     (1 or --auto-stop-pod = stop the RunPod pod
                                 via `runpodctl stop pod` after the suite)

Examples:
  bash scripts/remote/run_h100_v3_suite.sh --dry-run
  bash scripts/remote/run_h100_v3_suite.sh --steps eval_v2,h18_audit
  UPLOAD_TO_HF=1 bash scripts/remote/run_h100_v3_suite.sh --steps tenet_v3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      if [[ $# -lt 2 ]]; then
        echo "error: --steps requires a comma-separated value" >&2
        exit 2
      fi
      STEPS="$2"
      shift 2
      ;;
    --steps=*)
      STEPS="${1#--steps=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --auto-stop-pod)
      AUTO_STOP_POD=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1 (see --help)" >&2
      exit 2
      ;;
  esac
done

# Validate and (idempotently) order the requested steps.
declare -a SELECTED_STEPS=()
for step in $(echo "${STEPS}" | tr ',' ' '); do
  valid=0
  for known_step in "${ALL_STEPS[@]}"; do
    if [[ "${step}" == "${known_step}" ]]; then
      valid=1
      break
    fi
  done
  if [[ "${valid}" -ne 1 ]]; then
    echo "error: unknown step '${step}' (valid steps: ${ALL_STEPS[*]})" >&2
    exit 2
  fi
  already_selected=0
  for selected in ${SELECTED_STEPS[@]+"${SELECTED_STEPS[@]}"}; do
    if [[ "${selected}" == "${step}" ]]; then
      already_selected=1
      break
    fi
  done
  if [[ "${already_selected}" -ne 1 ]]; then
    SELECTED_STEPS+=("${step}")
  fi
done
if [[ ${#SELECTED_STEPS[@]} -eq 0 ]]; then
  echo "error: no steps selected (valid steps: ${ALL_STEPS[*]})" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
WATCHDOG_PID=""

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

# Run a command pinned to one GPU via CUDA_VISIBLE_DEVICES.
run_cmd_on_gpu() {
  local gpu_id="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] CUDA_VISIBLE_DEVICES=${gpu_id} $*"
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" "$@"
  fi
}

cleanup_hf_models() {
  if [[ "${KEEP_HF_CACHE}" == "1" ]]; then
    echo "[cleanup] KEEP_HF_CACHE=1 -> skipping HF cache cleanup for: $*"
    return 0
  fi
  local python_list="" model_id
  for model_id in "$@"; do
    python_list+="\"${model_id}\", "
  done
  python_list="[${python_list%, }]"
  run_cmd uv run python -c \
    "from sml.hub.release import cleanup_hf_model_cache; [cleanup_hf_model_cache(m) for m in ${python_list}]"
}

cleanup() {
  if [[ -n "${WATCHDOG_PID}" ]] && kill -0 "${WATCHDOG_PID}" 2>/dev/null; then
    echo "[cleanup] stopping idle watchdog (pid ${WATCHDOG_PID})"
    kill "${WATCHDOG_PID}" 2>/dev/null || true
    wait "${WATCHDOG_PID}" 2>/dev/null || true
  fi
}

# Optionally stop the RunPod pod once the suite is done (AUTO_STOP_POD=1).
stop_pod_if_requested() {
  if [[ "${AUTO_STOP_POD}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${NEBIUS_INSTANCE_ID:-}" ]]; then
    run_cmd nebius compute instance stop --id "${NEBIUS_INSTANCE_ID}"
    return 0
  fi
  if [[ -f /etc/rp_environment && "${DRY_RUN}" != "1" ]]; then
    # shellcheck disable=SC1091
    source /etc/rp_environment
  fi
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "[auto-stop] AUTO_STOP_POD=1 but neither NEBIUS_INSTANCE_ID nor RUNPOD_POD_ID is set; skipping pod stop"
    return 0
  fi
  run_cmd runpodctl stop pod "${RUNPOD_POD_ID}"
}

trap cleanup EXIT

preflight() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi not found; this suite must run on an H100 GPU host" >&2
    exit 1
  fi
  echo "[preflight] GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
  if [[ -f .env ]]; then
    echo "[preflight] sourcing .env"
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "error: HF_TOKEN is not set (export it or put it in .env)" >&2
    exit 1
  fi
  if [[ "${ENABLE_IDLE_WATCHDOG:-0}" == "1" ]]; then
    echo "[preflight] starting idle watchdog: scripts/remote/idle_watchdog.sh"
    bash scripts/remote/idle_watchdog.sh &
    WATCHDOG_PID=$!
  else
    echo "[preflight] idle_watchdog disabled by default (relying on --auto-stop-pod on exit)"
  fi
  export PATH="${PWD}/.venv/bin:${PATH}"
  if [[ "${SKIP_UV_SYNC}" == "1" ]]; then
    export UV_NO_SYNC=1
    echo "[preflight] SKIP_UV_SYNC=1: keeping existing .venv (UV_NO_SYNC=1)"
  else
    echo "[preflight] uv sync --frozen --extra data --extra parity --extra telemetry --extra eval"
    uv sync --frozen --extra data --extra parity --extra telemetry --extra eval
  fi
}

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
step_eval_v2() {
  local -a model_matrix=(
    "Tenet-4B-Qwen3-Unslop-v2|Qwen/Qwen3-4B|${NAIVE_4B_MODEL}"
    "Tenet-8B-Qwen3-Unslop-v2|Qwen/Qwen3-8B|${NAIVE_8B_MODEL}"
    "Tenet-32B-Qwen3-Unslop-v2|Qwen/Qwen3-32B|${NAIVE_32B_MODEL}"
  )
  local -a mmlu_pro_limit_flags=()
  if [[ "${MMLU_PRO_LIMIT}" -gt 0 ]]; then
    mmlu_pro_limit_flags+=(--mmlu-pro-limit "${MMLU_PRO_LIMIT}")
  fi

  _eval_v2_one() {
    local entry="$1" gpu_id="$2"
    local release_id base_model naive_model
    IFS='|' read -r release_id base_model naive_model <<<"${entry}"
    local -a eval_cmd=(
      uv run python scripts/eval/run_vllm_open_evals.py
      --release-id "${release_id}"
      --base-model "${base_model}"
      --naive-model "${naive_model}"
      --tenet-repo "${HF_NAMESPACE}/${release_id}"
      --suite full
      --backend vllm
      --precision bf16
      --batch-size "${BATCH_SIZE}"
      --max-gen-toks "${MAX_GEN_TOKS}"
      --think-end-token "${THINK_END_TOKEN}"
      ${mmlu_pro_limit_flags[@]+"${mmlu_pro_limit_flags[@]}"}
    )
    if [[ "${NUM_GPUS}" -ge 2 ]]; then
      run_cmd_on_gpu "${gpu_id}" "${eval_cmd[@]}"
    else
      run_cmd "${eval_cmd[@]}"
    fi
    run_cmd uv run python scripts/eval/export_model_index.py \
      --eval-json "results/external_evals/${release_id}.json" \
      --also-release-dir
    cleanup_hf_models "${base_model}" "${naive_model}" "${HF_NAMESPACE}/${release_id}"
  }

  local idx=0 entry
  if [[ "${NUM_GPUS}" -ge 2 ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      # Sequential dry-run: alternate GPUs so output is deterministic.
      for entry in "${model_matrix[@]}"; do
        _eval_v2_one "${entry}" $((idx % 2))
        idx=$((idx + 1))
      done
    else
      # Real run: execute entries in pairs, one per GPU, in parallel.
      local -a batch_pids=()
      local pid failed=0
      for entry in "${model_matrix[@]}"; do
        _eval_v2_one "${entry}" $((idx % 2)) &
        batch_pids+=("$!")
        idx=$((idx + 1))
        if [[ "${#batch_pids[@]}" -eq 2 ]]; then
          for pid in "${batch_pids[@]}"; do
            wait "${pid}" || failed=1
          done
          batch_pids=()
        fi
      done
      for pid in "${batch_pids[@]+"${batch_pids[@]}"}"; do
        wait "${pid}" || failed=1
      done
      if [[ "${failed}" -ne 0 ]]; then
        echo "error: one or more eval_v2 workers failed" >&2
        return 1
      fi
    fi
  else
    for entry in "${model_matrix[@]}"; do
      _eval_v2_one "${entry}" 0
    done
  fi
}

step_h18_audit() {
  run_cmd uv run python scripts/eval/audit_h18_qwen3_4b.py \
    --model-id Qwen/Qwen3-4B \
    --basis-path results/tenet_releases/Tenet-4B-Qwen3-Unslop-v2/lancet_basis.safetensors \
    --output results/h18_nvfp4_quant_snap/qwen3_4b_audit.json \
    --cleanup-cache
}

step_h18_h19_bench() {
  run_cmd uv run sml run \
    --config experiments/h18_nvfp4_quant_snap.yaml \
    --output-dir results \
    --run-id h18_nvfp4_quant_snap
  run_cmd uv run sml run \
    --config experiments/h19_loop_unrolled_lancet.yaml \
    --output-dir results \
    --run-id h19_loop_unrolled_lancet \
    --quant-snap
}

step_tenet_v3() {
  local upload_flags=()
  if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    upload_flags+=(--upload-to-hf)
  fi

  _tenet_v3_pilot() {
    local gpu_id="$1" target="$2" model_id="$3" release_id="$4" think_end_token="$5"
    local -a pilot_cmd=(
      uv run python scripts/run_tenet_v3_pilot.py
      --target "${target}"
      --model-id "${model_id}"
      --release-id "${release_id}"
      --quant-snap
      --eval-suite "${V3_EVAL_SUITE}"
      --think-end-token "${think_end_token}"
      --max-gen-toks "${MAX_GEN_TOKS}"
      --batch-size "${BATCH_SIZE}"
      --hf-namespace "${HF_NAMESPACE}"
      --drop-local-weights
      --cleanup-cache
      ${upload_flags[@]+"${upload_flags[@]}"}
    )
    if [[ "${NUM_GPUS}" -ge 2 ]]; then
      run_cmd_on_gpu "${gpu_id}" "${pilot_cmd[@]}"
    else
      run_cmd "${pilot_cmd[@]}"
    fi
  }

  if [[ "${NUM_GPUS}" -ge 2 ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      # Sequential dry-run: GPU 0 (moe) then GPU 1 (dense), deterministic.
      _tenet_v3_pilot 0 moe "${TENET_V3_MOE_MODEL}" "${TENET_V3_MOE_RELEASE}" "${THINK_END_TOKEN}"
      _tenet_v3_pilot 1 dense "${TENET_V3_DENSE_MODEL}" "${TENET_V3_DENSE_RELEASE}" "${DENSE_THINK_END_TOKEN}"
    else
      # Real run: moe on GPU 0 and dense on GPU 1 in parallel.
      _tenet_v3_pilot 0 moe "${TENET_V3_MOE_MODEL}" "${TENET_V3_MOE_RELEASE}" "${THINK_END_TOKEN}" &
      local moe_pid=$!
      _tenet_v3_pilot 1 dense "${TENET_V3_DENSE_MODEL}" "${TENET_V3_DENSE_RELEASE}" "${DENSE_THINK_END_TOKEN}" &
      local dense_pid=$!
      local failed=0
      wait "${moe_pid}" || failed=1
      wait "${dense_pid}" || failed=1
      if [[ "${failed}" -ne 0 ]]; then
        echo "error: one or more tenet_v3 pilots failed" >&2
        return 1
      fi
    fi
  else
    _tenet_v3_pilot 0 moe "${TENET_V3_MOE_MODEL}" "${TENET_V3_MOE_RELEASE}" "${THINK_END_TOKEN}"
    _tenet_v3_pilot 1 dense "${TENET_V3_DENSE_MODEL}" "${TENET_V3_DENSE_RELEASE}" "${DENSE_THINK_END_TOKEN}"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] selected steps: ${SELECTED_STEPS[*]}"
else
  preflight
fi

for step in ${SELECTED_STEPS[@]+"${SELECTED_STEPS[@]}"}; do
  echo "=== [step: ${step}] ==="
  case "${step}" in
    eval_v2) step_eval_v2 ;;
    h18_audit) step_h18_audit ;;
    h18_h19_bench) step_h18_h19_bench ;;
    tenet_v3) step_tenet_v3 ;;
  esac
done

echo "=== [digest] ==="
run_cmd uv run python scripts/eval/summarize_h100_v3_suite.py \
  --results-dir results \
  --output results/h100_v3_suite_summary.md

echo "H100 v3 suite complete (steps: ${SELECTED_STEPS[*]})"
stop_pod_if_requested
