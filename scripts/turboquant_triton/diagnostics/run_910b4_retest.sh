#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/turboquant}"
RUN_DIR="${LOG_ROOT}/910b4_retest_${TIMESTAMP}"
RESULTS_FILE="${RUN_DIR}/results.tsv"
SUMMARY_FILE="${RUN_DIR}/summary.txt"

MODEL="${MODEL:-/run/test_llm/Qwen3-0.6B-hf}"
PORT="${PORT:-18000}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
TQ_CACHE_DTYPE="${TQ_CACHE_DTYPE:-turboquant_4bit_nc}"
RUN_ACLGRAPH="${RUN_ACLGRAPH:-1}"
RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-1}"
DEBUG_SYNC="${DEBUG_SYNC:-1}"
MODEL_DEBUG_SYNC="${MODEL_DEBUG_SYNC:-0}"
NETWORK_IFNAME="${NETWORK_IFNAME:-eth0}"
FAILURES=0

mkdir -p "${RUN_DIR}"
printf 'stage\tstatus\texit_code\tlog\n' >"${RESULTS_FILE}"

# Keep all host-side collective transports on the server's configured NIC.
export GLOO_SOCKET_IFNAME="${NETWORK_IFNAME}"
export TP_SOCKET_IFNAME="${NETWORK_IFNAME}"
export HCCL_SOCKET_IFNAME="${NETWORK_IFNAME}"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
if [[ "${DEBUG_SYNC}" == "1" ]]; then
    export ASCEND_LAUNCH_BLOCKING=1
fi

run_stage() {
    local stage="$1"
    local description="$2"
    shift 2
    local log_file="${RUN_DIR}/${stage}.log"
    local status

    printf '\n===== %s: %s =====\n' "${stage}" "${description}"
    set +e
    "$@" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -eq 0 ]]; then
        printf '%s\tPASS\t0\t%s\n' "${stage}" "$(basename "${log_file}")" >>"${RESULTS_FILE}"
    else
        printf '%s\tFAIL\t%s\t%s\n' "${stage}" "${status}" "$(basename "${log_file}")" >>"${RESULTS_FILE}"
        FAILURES=$((FAILURES + 1))
    fi
}

skip_stage() {
    local stage="$1"
    local reason="$2"
    printf '%s\tSKIP\t0\t%s\n' "${stage}" "${reason}" >>"${RESULTS_FILE}"
}

collect_source() {
    git branch --show-current
    git rev-parse HEAD
    git status --short
    git log -3 --oneline --decorate
}

cd "${REPO_ROOT}"

if [[ ! -d "/sys/class/net/${NETWORK_IFNAME}" ]]; then
    printf 'Network interface does not exist: %s\n' "${NETWORK_IFNAME}"
    exit 2
fi

run_stage "00_source" "branch and commit used by the retest" collect_source
run_stage \
    "01_environment" \
    "vLLM core, vLLM Ascend, torch-npu, Triton-Ascend, and cache contract" \
    python3 "${TQ_COMMON_DIR}/check_environment.py"
run_stage \
    "02_backend" \
    "Python cache layout, metadata, prefill, and decode routing" \
    pytest -sv tests/ut/test_turboquant_kv_cache.py

# Every positive-slot store case uses a new pytest process. A device exception
# therefore cannot contaminate the next preset's NPU stream.
STORE_CASES=(
    "4bit-d64"
    "4bit-d128"
    "k3v4-d128"
    "3bit-d128"
)
for case_id in "${STORE_CASES[@]}"; do
    run_stage \
        "03_store_${case_id}" \
        "positive-slot store alignment regression (${case_id})" \
        pytest -sv \
        "tests/ut/ops/test_turboquant_triton.py::test_turboquant_store_writes_valid_slots[${case_id}]"
done

run_stage \
    "04_kernel_reference" \
    "store/dequant/packed-decode correctness and edge cases" \
    pytest -sv tests/ut/ops/test_turboquant_triton.py -k "not aclgraph"

if [[ "${RUN_ACLGRAPH}" == "1" ]]; then
    run_stage \
        "05_aclgraph" \
        "uniform multi-token graph capture and replay" \
        pytest -sv \
        tests/ut/ops/test_turboquant_triton.py::test_turboquant_store_and_decode_aclgraph_replay_matches_eager
else
    skip_stage "05_aclgraph" "RUN_ACLGRAPH=0"
fi

if [[ "${RUN_MODEL_SMOKE}" == "1" ]]; then
    MODEL_ENV=(env)
    if [[ "${MODEL_DEBUG_SYNC}" != "1" ]]; then
        MODEL_ENV+=(-u ASCEND_LAUNCH_BLOCKING)
    fi
    run_stage \
        "06_qwen3_0_6b_eager" \
        "Qwen3-0.6B native versus TurboQuant eager requests" \
        "${MODEL_ENV[@]}" \
        MODEL="${MODEL}" \
        PORT="${PORT}" \
        TP_SIZE="${TP_SIZE}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
        TQ_CACHE_DTYPE="${TQ_CACHE_DTYPE}" \
        OUTPUT_DIR="${RUN_DIR}/model_eager" \
        WARM_PREFIX=0 \
        bash "${TQ_CORRECTNESS_DIR}/run_accuracy_comparison.sh"
else
    skip_stage "06_qwen3_0_6b_eager" "RUN_MODEL_SMOKE=0"
fi

{
    printf 'TurboQuant 910B4 post-fix retest\n'
    printf 'Generated: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
    printf 'Model: %s\n' "${MODEL}"
    printf 'Port: %s\n' "${PORT}"
    printf 'TP size: %s\n' "${TP_SIZE}"
    printf 'Debug sync: %s\n' "${DEBUG_SYNC}"
    printf 'Model debug sync: %s\n' "${MODEL_DEBUG_SYNC}"
    printf 'Failed stages: %s\n\n' "${FAILURES}"
    column -t -s $'\t' "${RESULTS_FILE}" 2>/dev/null || cat "${RESULTS_FILE}"
} >"${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"

ARCHIVE_PATH="${RUN_DIR}.tar.gz"
if tar -czf "${ARCHIVE_PATH}" -C "${LOG_ROOT}" "$(basename "${RUN_DIR}")"; then
    printf 'Retest archive: %s\n' "${ARCHIVE_PATH}"
else
    printf 'Failed to create retest archive; logs remain at %s\n' "${RUN_DIR}"
    FAILURES=$((FAILURES + 1))
fi

if [[ ${FAILURES} -ne 0 ]]; then
    exit 1
fi
