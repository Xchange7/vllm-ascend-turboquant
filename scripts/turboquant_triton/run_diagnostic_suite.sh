#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/turboquant}"
RUN_DIR="${LOG_ROOT}/diagnostic_${TIMESTAMP}"
COMBINED_LOG="${RUN_DIR}/combined.log"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
RESULTS_FILE="${RUN_DIR}/results.tsv"

RUN_PROFILE="${RUN_PROFILE:-1}"
PROFILE_SEQUENCE_LENGTH="${PROFILE_SEQUENCE_LENGTH:-512}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-5}"
PROFILE_CACHE_DTYPE="${PROFILE_CACHE_DTYPE:-turboquant_4bit_nc}"
PROFILE_ACTIVATION_DTYPE="${PROFILE_ACTIVATION_DTYPE:-float16}"
PROFILE_NUM_QUERY_HEADS="${PROFILE_NUM_QUERY_HEADS:-32}"
PROFILE_NUM_KV_HEADS="${PROFILE_NUM_KV_HEADS:-4}"
PROFILE_HEAD_DIM="${PROFILE_HEAD_DIM:-128}"
PROFILE_NUM_KV_SPLITS="${PROFILE_NUM_KV_SPLITS:-8}"
COLLECT_PROFILE_TRACE="${COLLECT_PROFILE_TRACE:-0}"
DEBUG_SYNC="${DEBUG_SYNC:-0}"

mkdir -p "${RUN_DIR}"
: >"${COMBINED_LOG}"
printf 'stage\tstatus\texit_code\tlog\n' >"${RESULTS_FILE}"

export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
if [[ "${DEBUG_SYNC}" == "1" ]]; then
    export ASCEND_LAUNCH_BLOCKING=1
fi

FAILURES=0

run_stage() {
    local stage="$1"
    local description="$2"
    shift 2

    local stage_log="${RUN_DIR}/${stage}.log"
    local status
    printf '\n===== %s: %s =====\n' "${stage}" "${description}" | tee -a "${COMBINED_LOG}"

    set +e
    "$@" 2>&1 | tee "${stage_log}" | tee -a "${COMBINED_LOG}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -eq 0 ]]; then
        printf '%s\tPASS\t0\t%s\n' "${stage}" "$(basename "${stage_log}")" >>"${RESULTS_FILE}"
        printf '===== %s: PASS =====\n' "${stage}" | tee -a "${COMBINED_LOG}"
    else
        printf '%s\tFAIL\t%s\t%s\n' "${stage}" "${status}" "$(basename "${stage_log}")" >>"${RESULTS_FILE}"
        printf '===== %s: FAIL (exit %s) =====\n' "${stage}" "${status}" | tee -a "${COMBINED_LOG}"
        FAILURES=$((FAILURES + 1))
    fi
}

collect_system_info() {
    printf 'timestamp: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
    printf 'hostname: %s\n' "$(hostname)"
    printf 'kernel: %s\n' "$(uname -a)"
    printf 'working_directory: %s\n' "${REPO_ROOT}"
    printf 'python: %s\n' "$(command -v python3 || true)"
    python3 --version

    printf '\nInstalled package metadata:\n'
    python3 -m pip show \
        torch \
        torch-npu \
        triton \
        triton-ascend \
        vllm \
        vllm-ascend || true

    printf '\nResolved module paths:\n'
    for module in torch torch_npu triton vllm vllm_ascend; do
        python3 -c \
            "import importlib.util; spec = importlib.util.find_spec('${module}'); print('${module}:', None if spec is None else spec.origin)"
    done

    printf '\nSelected environment variables:\n'
    for variable in \
        ASCEND_RT_VISIBLE_DEVICES \
        ASCEND_DEVICE_ID \
        ASCEND_HOME_PATH \
        ASCEND_LAUNCH_BLOCKING \
        HCCL_CONNECT_TIMEOUT \
        PYTHONPATH \
        VLLM_USE_V1 \
        VLLM_USE_V2_MODEL_RUNNER \
        VLLM_ASCEND_ENABLE_NZ; do
        printf '%s=%s\n' "${variable}" "${!variable-<unset>}"
    done

    if command -v npu-smi >/dev/null 2>&1; then
        printf '\nnpu-smi info:\n'
        npu-smi info
    else
        printf '\nnpu-smi: command not found\n'
    fi
}

collect_source_info() {
    git rev-parse --show-toplevel
    git branch --show-current
    git rev-parse HEAD
    git status --short
    printf '\nRecent commits:\n'
    git log -5 --oneline --decorate
    printf '\nChanged files:\n'
    git diff --stat
}

cd "${REPO_ROOT}"

run_stage "00_system" "host, runtime variables, and NPU inventory" collect_system_info
run_stage "01_source" "branch, commit, and working-tree state" collect_source_info
run_stage \
    "02_environment" \
    "vLLM, vLLM-Ascend, torch-npu, Triton-Ascend, and TurboQuant API checks" \
    python3 scripts/turboquant_triton/check_environment.py
run_stage \
    "03_platform" \
    "TurboQuant backend selection and uniform ACLGraph capability" \
    pytest -sv \
    tests/ut/test_platform.py::TestNPUPlatform::test_get_attn_backend_cls_uses_turboquant \
    tests/ut/test_platform.py::TestNPUPlatform::test_turboquant_supports_uniform_batch_graph_capture
run_stage \
    "04_backend" \
    "cache layout, mixed batch, continuation prefill, prefix pages, ALiBi, and soft cap" \
    pytest -sv tests/ut/test_turboquant_kv_cache.py
run_stage \
    "05_triton_kernels" \
    "packed store, dequantization, decode, dtype/head-dim, cross-page, and spec query lengths" \
    pytest -sv tests/ut/ops/test_turboquant_triton.py -k "not aclgraph"
run_stage \
    "06_aclgraph" \
    "uniform multi-token ACLGraph capture and replay" \
    pytest -sv \
    tests/ut/ops/test_turboquant_triton.py::test_turboquant_store_and_decode_aclgraph_replay_matches_eager

if [[ "${RUN_PROFILE}" == "1" ]]; then
    PROFILE_ARGS=(
        --operation all
        --native-baseline
        --cache-dtype "${PROFILE_CACHE_DTYPE}"
        --activation-dtype "${PROFILE_ACTIVATION_DTYPE}"
        --batch-size 4
        --sequence-length "${PROFILE_SEQUENCE_LENGTH}"
        --num-query-heads "${PROFILE_NUM_QUERY_HEADS}"
        --num-kv-heads "${PROFILE_NUM_KV_HEADS}"
        --head-dim "${PROFILE_HEAD_DIM}"
        --num-kv-splits "${PROFILE_NUM_KV_SPLITS}"
        --warmup 3
        --iterations "${PROFILE_ITERATIONS}"
        --trace-dir "${RUN_DIR}/profile"
    )
    if [[ "${COLLECT_PROFILE_TRACE}" != "1" ]]; then
        PROFILE_ARGS+=(--no-trace)
    fi
    run_stage \
        "07_profile" \
        "short store/decode/dequant latency and memory profile" \
        python3 scripts/turboquant_triton/profile_kernels.py "${PROFILE_ARGS[@]}"
else
    printf '07_profile\tSKIP\t0\tRUN_PROFILE=0\n' >>"${RESULTS_FILE}"
fi

{
    printf 'TurboQuant diagnostic summary\n'
    printf 'Generated: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
    printf 'Repository: %s\n' "${REPO_ROOT}"
    printf 'Log directory: %s\n' "${RUN_DIR}"
    printf 'Failed stages: %s\n\n' "${FAILURES}"
    column -t -s $'\t' "${RESULTS_FILE}" 2>/dev/null || cat "${RESULTS_FILE}"
    printf '\nDebug options:\n'
    printf 'DEBUG_SYNC=%s\n' "${DEBUG_SYNC}"
    printf 'RUN_PROFILE=%s\n' "${RUN_PROFILE}"
    printf 'COLLECT_PROFILE_TRACE=%s\n' "${COLLECT_PROFILE_TRACE}"
} >"${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"

ARCHIVE_PATH="${RUN_DIR}.tar.gz"
if tar -czf "${ARCHIVE_PATH}" -C "${LOG_ROOT}" "$(basename "${RUN_DIR}")"; then
    printf '\nLog archive: %s\n' "${ARCHIVE_PATH}"
else
    printf '\nFailed to create archive; logs remain at: %s\n' "${RUN_DIR}"
    FAILURES=$((FAILURES + 1))
fi

if [[ ${FAILURES} -ne 0 ]]; then
    printf 'TurboQuant diagnostics completed with %s failed stage(s).\n' "${FAILURES}"
    exit 1
fi

printf 'TurboQuant diagnostics completed successfully.\n'
