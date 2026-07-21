#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL="${MODEL:-/run/test_llm/Qwen3-0.6B-hf}"
PORT="${PORT:-18003}"
DEVICE_IDS="${DEVICE_IDS:-0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
NETWORK_IFNAME="${NETWORK_IFNAME:-eth0}"
TQ_CACHE_DTYPE="${TQ_CACHE_DTYPE:-turboquant_4bit_nc}"
MODES="${MODES:-native native_repeat tq_reference tq_auto tq_ascend_fused}"
TEACHER_CASES="${TEACHER_CASES:-${SCRIPT_DIR}/teacher_forcing_cases.jsonl}"
QUALITY_CASES="${QUALITY_CASES:-${SCRIPT_DIR}/quality_cases.jsonl}"
LOGPROBS="${LOGPROBS:-20}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/e2e_correctness_${TIMESTAMP}}"

# These are screening thresholds, not paper-level quality claims. Override them
# after establishing a stable baseline for the target model and cache dtype.
MAX_PROMPT_MEAN_ABS_LOGPROB_DIFF="${MAX_PROMPT_MEAN_ABS_LOGPROB_DIFF:-0.30}"
MAX_PROMPT_P95_ABS_LOGPROB_DIFF="${MAX_PROMPT_P95_ABS_LOGPROB_DIFF:-0.75}"
MAX_PROMPT_ABS_MEAN_NLL_DELTA="${MAX_PROMPT_ABS_MEAN_NLL_DELTA:-0.10}"
MIN_FIRST_TOKEN_TOP1_MATCH_RATE="${MIN_FIRST_TOKEN_TOP1_MATCH_RATE:-0.85}"
MIN_FIRST_TOKEN_TOPK_OVERLAP="${MIN_FIRST_TOKEN_TOPK_OVERLAP:-0.50}"
MAX_QUALITY_REGRESSIONS="${MAX_QUALITY_REGRESSIONS:-0}"

SERVER_PID=""
FINAL_STATUS=0
mkdir -p "${OUTPUT_DIR}/runs" "${OUTPUT_DIR}/comparisons"

export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_IDS}"
export GLOO_SOCKET_IFNAME="${NETWORK_IFNAME}"
export TP_SOCKET_IFNAME="${NETWORK_IFNAME}"
export HCCL_SOCKET_IFNAME="${NETWORK_IFNAME}"

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
        return 0
    fi
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
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
            return 0
        fi
        sleep 1
    done
    printf 'Port %s is still listening after server shutdown.\n' "${PORT}" >&2
    return 1
}

finalize() {
    local status=$?
    trap - EXIT INT TERM
    if ! stop_server; then
        status=1
    fi
    if ((FINAL_STATUS != 0)); then
        status="${FINAL_STATUS}"
    fi
    python3 "${SCRIPT_DIR}/summarize_e2e.py" \
        --input-dir "${OUTPUT_DIR}/comparisons" \
        --output-json "${OUTPUT_DIR}/summary.json" \
        --output-markdown "${OUTPUT_DIR}/summary.md" || status=1
    tar -czf "${OUTPUT_DIR}.tar.gz" \
        -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")" || status=1
    printf 'E2E summary: %s\n' "${OUTPUT_DIR}/summary.md"
    printf 'E2E archive: %s.tar.gz\n' "${OUTPUT_DIR}"
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
            printf 'Server exited before becoming healthy. Last log lines:\n' >&2
            tail -n 120 "${log_file}" >&2 || true
            return 1
        fi
        if curl --noproxy '*' --fail --silent \
            --connect-timeout 2 --max-time 5 \
            "http://127.0.0.1:${PORT}/health" >/dev/null; then
            printf 'Server is healthy on port %s.\n' "${PORT}"
            return 0
        fi
        if ((SECONDS >= next_report)); then
            printf 'Waiting for %s; latest server log: ' "${PORT}"
            tail -n 1 "${log_file}" || true
            next_report=$((SECONDS + 30))
        fi
        sleep 5
    done
    printf 'Server health check timed out after %ss.\n' "${SERVER_TIMEOUT}" >&2
    tail -n 120 "${log_file}" >&2 || true
    return 1
}

mode_configuration() {
    case "$1" in
        native | native_repeat)
            MODE_CACHE_DTYPE="auto"
            MODE_DECODE_IMPLEMENTATION="auto"
            ;;
        tq_reference)
            MODE_CACHE_DTYPE="${TQ_CACHE_DTYPE}"
            MODE_DECODE_IMPLEMENTATION="reference"
            ;;
        tq_auto)
            MODE_CACHE_DTYPE="${TQ_CACHE_DTYPE}"
            MODE_DECODE_IMPLEMENTATION="auto"
            ;;
        tq_ascend_fused)
            MODE_CACHE_DTYPE="${TQ_CACHE_DTYPE}"
            MODE_DECODE_IMPLEMENTATION="ascend_fused"
            ;;
        *)
            printf 'Unknown mode: %s\n' "$1" >&2
            return 2
            ;;
    esac
}

start_server() {
    local mode="$1"
    local log_file="${OUTPUT_DIR}/runs/${mode}/server.log"
    mode_configuration "${mode}" || return
    mkdir -p "${OUTPUT_DIR}/runs/${mode}"
    printf 'Starting %s: cache=%s decode=%s devices=%s tp=%s\n' \
        "${mode}" "${MODE_CACHE_DTYPE}" "${MODE_DECODE_IMPLEMENTATION}" \
        "${DEVICE_IDS}" "${TP_SIZE}"
    MODEL="${MODEL}" \
        PORT="${PORT}" \
        TP_SIZE="${TP_SIZE}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        NETWORK_IFNAME="${NETWORK_IFNAME}" \
        KV_CACHE_DTYPE="${MODE_CACHE_DTYPE}" \
        ENFORCE_EAGER=1 \
        VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION="${MODE_DECODE_IMPLEMENTATION}" \
        bash "${TQ_COMMON_DIR}/serve_qwen3_32b.sh" \
        --no-enable-prefix-caching \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --max-logprobs "${LOGPROBS}" >"${log_file}" 2>&1 &
    SERVER_PID=$!
    wait_for_server "${log_file}"
}

collect_suite() {
    local mode="$1"
    local suite="$2"
    local request_mode="$3"
    local prompts="$4"
    local prompt_logprobs="$5"
    local run_dir="${OUTPUT_DIR}/runs/${mode}"
    local args=(
        collect
        --base-url "http://127.0.0.1:${PORT}"
        --model "${MODEL}"
        --prompts "${prompts}"
        --output "${run_dir}/${suite}.json"
        --text-output "${run_dir}/${suite}_answers.txt"
        --label "${mode}"
        --seed 0
        --logprobs "${LOGPROBS}"
        --prompt-logprobs "${prompt_logprobs}"
        --timeout "${REQUEST_TIMEOUT}"
        --request-mode "${request_mode}"
        --max-model-len "${MAX_MODEL_LEN}"
    )
    python3 "${SCRIPT_DIR}/accuracy_eval.py" "${args[@]}" 2>&1 \
        | tee "${run_dir}/${suite}.log"
    return "${PIPESTATUS[0]}"
}

run_mode() {
    local mode="$1"
    if ! start_server "${mode}"; then
        FINAL_STATUS=1
        stop_server || true
        return
    fi
    collect_suite "${mode}" teacher_forcing completion \
        "${TEACHER_CASES}" "${PROMPT_LOGPROBS}" || FINAL_STATUS=1
    collect_suite "${mode}" quality chat "${QUALITY_CASES}" 0 || FINAL_STATUS=1
    stop_server || FINAL_STATUS=1
}

compare_pair() {
    local base="$1"
    local candidate="$2"
    local suite="$3"
    local pair_dir="${OUTPUT_DIR}/comparisons/${base}_vs_${candidate}"
    mkdir -p "${pair_dir}"
    local args=(
        compare
        --native "${OUTPUT_DIR}/runs/${base}/${suite}.json"
        --turboquant "${OUTPUT_DIR}/runs/${candidate}/${suite}.json"
        --output "${pair_dir}/${suite}.json"
        --summary-markdown "${pair_dir}/${suite}.md"
        --min-exact-match-rate 0
        --min-token-prefix-rate 0
    )
    if [[ "${suite}" == "teacher_forcing" ]]; then
        args+=(
            --max-prompt-mean-abs-logprob-diff "${MAX_PROMPT_MEAN_ABS_LOGPROB_DIFF}"
            --max-prompt-p95-abs-logprob-diff "${MAX_PROMPT_P95_ABS_LOGPROB_DIFF}"
            --max-prompt-abs-mean-nll-delta "${MAX_PROMPT_ABS_MEAN_NLL_DELTA}"
            --min-first-token-top1-match-rate "${MIN_FIRST_TOKEN_TOP1_MATCH_RATE}"
            --min-first-token-topk-overlap "${MIN_FIRST_TOKEN_TOPK_OVERLAP}"
        )
    else
        args+=(--max-quality-regressions "${MAX_QUALITY_REGRESSIONS}")
    fi

    python3 "${SCRIPT_DIR}/accuracy_eval.py" "${args[@]}" 2>&1 \
        | tee "${pair_dir}/${suite}.log"
    local status="${PIPESTATUS[0]}"
    printf '%s\n' "${status}" >"${pair_dir}/${suite}.status"
    if ((status != 0)); then
        FINAL_STATUS=1
    fi
}

contains_mode() {
    local expected="$1"
    local mode
    for mode in ${MODES}; do
        if [[ "${mode}" == "${expected}" ]]; then
            return 0
        fi
    done
    return 1
}

cd "${REPO_ROOT}"
if port_is_listening; then
    printf 'Port %s is already in use; refusing to test the wrong server.\n' "${PORT}" >&2
    exit 2
fi
if ! contains_mode native; then
    printf 'MODES must include native.\n' >&2
    exit 2
fi

{
    printf 'model=%s\n' "${MODEL}"
    printf 'modes=%s\n' "${MODES}"
    printf 'cache_dtype=%s\n' "${TQ_CACHE_DTYPE}"
    printf 'devices=%s\n' "${DEVICE_IDS}"
    printf 'tp_size=%s\n' "${TP_SIZE}"
    printf 'max_model_len=%s\n' "${MAX_MODEL_LEN}"
    printf 'max_num_batched_tokens=%s\n' "${MAX_NUM_BATCHED_TOKENS}"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
} | tee "${OUTPUT_DIR}/configuration.txt"

python3 "${TQ_COMMON_DIR}/check_environment.py" 2>&1 \
    | tee "${OUTPUT_DIR}/environment.log"
CHECK_STATUS="${PIPESTATUS[0]}"
if ((CHECK_STATUS != 0)); then
    exit 1
fi

for mode in ${MODES}; do
    run_mode "${mode}"
done

for candidate in ${MODES}; do
    if [[ "${candidate}" == "native" ]]; then
        continue
    fi
    compare_pair native "${candidate}" teacher_forcing
    compare_pair native "${candidate}" quality
done

if contains_mode tq_reference; then
    for candidate in ${MODES}; do
        if [[ "${candidate}" == "native" \
            || "${candidate}" == "native_repeat" \
            || "${candidate}" == "tq_reference" ]]; then
            continue
        fi
        compare_pair tq_reference "${candidate}" teacher_forcing
        compare_pair tq_reference "${candidate}" quality
    done
fi

exit "${FINAL_STATUS}"
