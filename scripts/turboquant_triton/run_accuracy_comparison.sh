#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:?Set MODEL to the local model path or served model ID}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TQ_CACHE_DTYPE="${TQ_CACHE_DTYPE:-turboquant_4bit_nc}"
BASE_LABEL="${BASE_LABEL:-native}"
BASE_CACHE_DTYPE="${BASE_CACHE_DTYPE:-auto}"
BASE_ENFORCE_EAGER="${BASE_ENFORCE_EAGER:-1}"
BASE_SPECULATIVE_CONFIG="${BASE_SPECULATIVE_CONFIG:-}"
TEST_LABEL="${TEST_LABEL:-turboquant}"
TEST_CACHE_DTYPE="${TEST_CACHE_DTYPE:-${TQ_CACHE_DTYPE}}"
TEST_ENFORCE_EAGER="${TEST_ENFORCE_EAGER:-1}"
TEST_SPECULATIVE_CONFIG="${TEST_SPECULATIVE_CONFIG:-}"
PROMPTS="${PROMPTS:-${SCRIPT_DIR}/accuracy_prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/accuracy_${TIMESTAMP}}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
MIN_EXACT_MATCH_RATE="${MIN_EXACT_MATCH_RATE:-0.0}"
MIN_TOKEN_PREFIX_RATE="${MIN_TOKEN_PREFIX_RATE:-0.0}"
MAX_MEAN_LOGPROB_DIFF="${MAX_MEAN_LOGPROB_DIFF:-}"
MIN_TURBOQUANT_ACCURACY="${MIN_TURBOQUANT_ACCURACY:-}"
MAX_QUALITY_REGRESSIONS="${MAX_QUALITY_REGRESSIONS:-}"
WARM_PREFIX="${WARM_PREFIX:-1}"
REQUEST_MODE="${REQUEST_MODE:-completion}"

SERVER_PID=""
mkdir -p "${OUTPUT_DIR}"

port_is_listening() {
    python3 -c '
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(1)
    sys.exit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)
' "${PORT}"
}

stop_server() {
    if [[ -z "${SERVER_PID}" ]]; then
        return
    fi
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 60); do
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -KILL "${SERVER_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""

    for _ in $(seq 1 60); do
        if ! port_is_listening; then
            return
        fi
        sleep 1
    done
    printf 'Port %s is still listening after stopping the test server.\n' "${PORT}"
    return 1
}

finalize() {
    local status=$?
    trap - EXIT INT TERM
    if ! stop_server; then
        status=1
    fi
    if ! tar -czf "${OUTPUT_DIR}.tar.gz" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"; then
        status=1
    fi
    printf 'Accuracy output: %s\n' "${OUTPUT_DIR}"
    printf 'Accuracy archive: %s.tar.gz\n' "${OUTPUT_DIR}"
    exit "${status}"
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
    local log_file="$1"
    local deadline=$((SECONDS + SERVER_TIMEOUT))
    local next_report=$((SECONDS + 30))
    while ((SECONDS < deadline)); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            printf 'Server exited before becoming healthy. Last log lines:\n'
            tail -n 100 "${log_file}" || true
            return 1
        fi
        if curl --noproxy '*' --fail --silent \
            --connect-timeout 2 --max-time 5 \
            "http://127.0.0.1:${PORT}/health" >/dev/null; then
            printf 'Server is healthy on port %s.\n' "${PORT}"
            return 0
        fi
        if ((SECONDS >= next_report)); then
            printf 'Still waiting for port %s; latest server log: ' "${PORT}"
            tail -n 1 "${log_file}" || true
            next_report=$((SECONDS + 30))
        fi
        sleep 5
    done
    printf 'Timed out after %ss waiting for port %s. Last log lines:\n' "${SERVER_TIMEOUT}" "${PORT}"
    tail -n 100 "${log_file}" || true
    return 1
}

start_server() {
    local label="$1"
    local cache_dtype="$2"
    local enforce_eager="$3"
    local speculative_config="$4"
    local log_file="${OUTPUT_DIR}/${label}_server.log"

    printf 'Starting %s server with cache dtype %s; log: %s\n' \
        "${label}" "${cache_dtype}" "${log_file}"
    MODEL="${MODEL}" \
        PORT="${PORT}" \
        TP_SIZE="${TP_SIZE}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        KV_CACHE_DTYPE="${cache_dtype}" \
        ENFORCE_EAGER="${enforce_eager}" \
        SPECULATIVE_CONFIG="${speculative_config}" \
        bash "${SCRIPT_DIR}/serve_qwen3_32b.sh" >"${log_file}" 2>&1 &
    SERVER_PID=$!
    wait_for_server "${log_file}"
}

collect_results() {
    local label="$1"
    local collect_args=(
        collect
        --base-url "http://127.0.0.1:${PORT}"
        --model "${MODEL}"
        --prompts "${PROMPTS}"
        --output "${OUTPUT_DIR}/${label}.json"
        --label "${label}"
        --timeout "${REQUEST_TIMEOUT}"
        --max-model-len "${MAX_MODEL_LEN}"
        --request-mode "${REQUEST_MODE}"
    )
    if [[ "${WARM_PREFIX}" == "1" ]]; then
        collect_args+=(--warm-prefix)
    fi
    python3 "${SCRIPT_DIR}/accuracy_eval.py" "${collect_args[@]}"
}

cd "${REPO_ROOT}"
if port_is_listening; then
    printf 'Port %s is already in use; refusing to test the wrong process.\n' "${PORT}"
    exit 2
fi
python3 "${SCRIPT_DIR}/check_environment.py" | tee "${OUTPUT_DIR}/environment.log"

start_server \
    "${BASE_LABEL}" \
    "${BASE_CACHE_DTYPE}" \
    "${BASE_ENFORCE_EAGER}" \
    "${BASE_SPECULATIVE_CONFIG}"
collect_results base
stop_server

start_server \
    "${TEST_LABEL}" \
    "${TEST_CACHE_DTYPE}" \
    "${TEST_ENFORCE_EAGER}" \
    "${TEST_SPECULATIVE_CONFIG}"
collect_results test
stop_server

COMPARE_ARGS=(
    compare
    --native "${OUTPUT_DIR}/base.json"
    --turboquant "${OUTPUT_DIR}/test.json"
    --output "${OUTPUT_DIR}/comparison.json"
    --summary-markdown "${OUTPUT_DIR}/summary.md"
    --min-exact-match-rate "${MIN_EXACT_MATCH_RATE}"
    --min-token-prefix-rate "${MIN_TOKEN_PREFIX_RATE}"
)
if [[ -n "${MAX_MEAN_LOGPROB_DIFF}" ]]; then
    COMPARE_ARGS+=(--max-mean-logprob-diff "${MAX_MEAN_LOGPROB_DIFF}")
fi
if [[ -n "${MIN_TURBOQUANT_ACCURACY}" ]]; then
    COMPARE_ARGS+=(--min-turboquant-accuracy "${MIN_TURBOQUANT_ACCURACY}")
fi
if [[ -n "${MAX_QUALITY_REGRESSIONS}" ]]; then
    COMPARE_ARGS+=(--max-quality-regressions "${MAX_QUALITY_REGRESSIONS}")
fi
python3 "${SCRIPT_DIR}/accuracy_eval.py" "${COMPARE_ARGS[@]}" | tee "${OUTPUT_DIR}/comparison.log"
printf 'Accuracy report: %s\n' "${OUTPUT_DIR}/summary.md"
