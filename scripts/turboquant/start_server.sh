#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
if [[ "$#" -gt 0 && "${1}" != --* ]]; then
  MODEL_PATH="${1}"
  shift
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: MODEL_PATH=/path/to/model $0 [extra vllm args]"
  echo "   or: $0 /path/to/model [extra vllm args]"
  exit 2
fi

WORK_DIR="${WORK_DIR:-/workspace}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-turboquant-smoke}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-turboquant_4bit_nc}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DRY_RUN="${DRY_RUN:-0}"

export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-0}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"

cmd=(
  vllm serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --disable-log-requests
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

if [[ -n "${LOAD_FORMAT}" && "${LOAD_FORMAT}" != "real" ]]; then
  cmd+=(--load-format "${LOAD_FORMAT}")
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "Work dir: ${WORK_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "KV cache dtype: ${KV_CACHE_DTYPE}"
echo "Load format: ${LOAD_FORMAT}"
echo "VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER}"
echo "Command:"
printf ' %q' "${cmd[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${WORK_DIR}"
exec "${cmd[@]}"
