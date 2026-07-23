#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"

MODEL="${MODEL:?Set MODEL to the local Qwen3 model directory}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/logs/turboquant/full_910b4_${TIMESTAMP}}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
NETWORK_IFNAME="${NETWORK_IFNAME:-eth0}"
PORT="${PORT:-18100}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TQ_CACHE_DTYPE="${TQ_CACHE_DTYPE:-turboquant_4bit_nc}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bfloat16}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
NPU_POLL_INTERVAL="${NPU_POLL_INTERVAL:-30}"
FAIL_FAST="${FAIL_FAST:-0}"
QUICK="${QUICK:-0}"

if [[ "${QUICK}" == "1" ]]; then
    DEFAULT_RUN_QUALITY=0
    DEFAULT_RUN_GRAPH=0
    DEFAULT_RUN_SERVING_GRAPH=0
    DEFAULT_PROFILE_ITERATIONS=5
    DEFAULT_MEASURE_REQUESTS=16
else
    DEFAULT_RUN_QUALITY=1
    DEFAULT_RUN_GRAPH=1
    DEFAULT_RUN_SERVING_GRAPH=1
    DEFAULT_PROFILE_ITERATIONS=20
    DEFAULT_MEASURE_REQUESTS=32
fi

RUN_KERNELS="${RUN_KERNELS:-1}"
RUN_RUNTIME_PROBE="${RUN_RUNTIME_PROBE:-1}"
RUN_OPERATOR_CORRECTNESS="${RUN_OPERATOR_CORRECTNESS:-1}"
RUN_OPERATOR_PROFILE="${RUN_OPERATOR_PROFILE:-1}"
RUN_ACCURACY="${RUN_ACCURACY:-1}"
RUN_QUALITY="${RUN_QUALITY:-${DEFAULT_RUN_QUALITY}}"
RUN_GRAPH="${RUN_GRAPH:-${DEFAULT_RUN_GRAPH}}"
RUN_SERVING_EAGER="${RUN_SERVING_EAGER:-1}"
RUN_SERVING_GRAPH="${RUN_SERVING_GRAPH:-${DEFAULT_RUN_SERVING_GRAPH}}"
COLLECT_PROFILE_TRACE="${COLLECT_PROFILE_TRACE:-0}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-${DEFAULT_PROFILE_ITERATIONS}}"

ACCURACY_MAX_MODEL_LEN="${ACCURACY_MAX_MODEL_LEN:-4096}"
ACCURACY_MAX_NUM_SEQS="${ACCURACY_MAX_NUM_SEQS:-8}"
ACCURACY_MIN_PREFIX_RATE="${ACCURACY_MIN_PREFIX_RATE:-0.25}"
QUALITY_MAX_MODEL_LEN="${QUALITY_MAX_MODEL_LEN:-16384}"
QUALITY_MIN_ACCURACY="${QUALITY_MIN_ACCURACY:-0.70}"
QUALITY_MAX_REGRESSIONS="${QUALITY_MAX_REGRESSIONS:-0}"
GRAPH_MAX_MODEL_LEN="${GRAPH_MAX_MODEL_LEN:-4096}"
GRAPH_MIN_PREFIX_RATE="${GRAPH_MIN_PREFIX_RATE:-0.95}"

PERF_MAX_MODEL_LEN="${PERF_MAX_MODEL_LEN:-16384}"
PERF_INPUT_TOKENS="${PERF_INPUT_TOKENS:-12288}"
PERF_OUTPUT_TOKENS="${PERF_OUTPUT_TOKENS:-256}"
PERF_CONCURRENCY="${PERF_CONCURRENCY:-16}"
PERF_CONCURRENCY_LEVELS="${PERF_CONCURRENCY_LEVELS:-1 ${PERF_CONCURRENCY}}"
PERF_WARMUP_REQUESTS="${PERF_WARMUP_REQUESTS:-16}"
PERF_MEASURE_REQUESTS="${PERF_MEASURE_REQUESTS:-${DEFAULT_MEASURE_REQUESTS}}"
PERF_DECODE_IMPLEMENTATION="${PERF_DECODE_IMPLEMENTATION:-ascend_fused}"
PERF_MAX_TTFT_RATIO="${PERF_MAX_TTFT_RATIO:-2.0}"
PERF_MAX_TPOT_RATIO="${PERF_MAX_TPOT_RATIO:-2.0}"
PERF_MIN_THROUGHPUT_RATIO="${PERF_MIN_THROUGHPUT_RATIO:-0.5}"
PERF_MIN_KV_CAPACITY_RATIO="${PERF_MIN_KV_CAPACITY_RATIO:-2.0}"

RESULTS_FILE="${RUN_ROOT}/results.tsv"
COMBINED_LOG="${RUN_ROOT}/combined.log"
SUMMARY_FILE="${RUN_ROOT}/summary.txt"
CONFIG_FILE="${RUN_ROOT}/configuration.env"
FAILURES=0
LAST_STAGE_STATUS=0
NPU_MONITOR_PID=""
FINALIZED=0

mkdir -p "${RUN_ROOT}"
printf 'stage\tstatus\texit_code\tlog\tnpu_log\n' >"${RESULTS_FILE}"
: >"${COMBINED_LOG}"

export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export GLOO_SOCKET_IFNAME="${NETWORK_IFNAME}"
export TP_SOCKET_IFNAME="${NETWORK_IFNAME}"
export HCCL_SOCKET_IFNAME="${NETWORK_IFNAME}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONUNBUFFERED=1

write_configuration() {
    {
        printf 'MODEL=%q\n' "${MODEL}"
        printf 'SOC_VERSION=%q\n' "${SOC_VERSION:-<unset>}"
        printf 'VISIBLE_DEVICES=%q\n' "${VISIBLE_DEVICES}"
        printf 'TP_SIZE=%q\n' "${TP_SIZE}"
        printf 'NETWORK_IFNAME=%q\n' "${NETWORK_IFNAME}"
        printf 'PORT=%q\n' "${PORT}"
        printf 'TQ_CACHE_DTYPE=%q\n' "${TQ_CACHE_DTYPE}"
        printf 'ACTIVATION_DTYPE=%q\n' "${ACTIVATION_DTYPE}"
        printf 'GPU_MEMORY_UTILIZATION=%q\n' "${GPU_MEMORY_UTILIZATION}"
        printf 'QUICK=%q\n' "${QUICK}"
        printf 'RUN_KERNELS=%q\n' "${RUN_KERNELS}"
        printf 'RUN_RUNTIME_PROBE=%q\n' "${RUN_RUNTIME_PROBE}"
        printf 'RUN_OPERATOR_CORRECTNESS=%q\n' "${RUN_OPERATOR_CORRECTNESS}"
        printf 'RUN_OPERATOR_PROFILE=%q\n' "${RUN_OPERATOR_PROFILE}"
        printf 'RUN_ACCURACY=%q\n' "${RUN_ACCURACY}"
        printf 'RUN_QUALITY=%q\n' "${RUN_QUALITY}"
        printf 'RUN_GRAPH=%q\n' "${RUN_GRAPH}"
        printf 'RUN_SERVING_EAGER=%q\n' "${RUN_SERVING_EAGER}"
        printf 'RUN_SERVING_GRAPH=%q\n' "${RUN_SERVING_GRAPH}"
        printf 'PERF_MAX_MODEL_LEN=%q\n' "${PERF_MAX_MODEL_LEN}"
        printf 'PERF_INPUT_TOKENS=%q\n' "${PERF_INPUT_TOKENS}"
        printf 'PERF_OUTPUT_TOKENS=%q\n' "${PERF_OUTPUT_TOKENS}"
        printf 'PERF_CONCURRENCY=%q\n' "${PERF_CONCURRENCY}"
        printf 'PERF_CONCURRENCY_LEVELS=%q\n' "${PERF_CONCURRENCY_LEVELS}"
        printf 'PERF_WARMUP_REQUESTS=%q\n' "${PERF_WARMUP_REQUESTS}"
        printf 'PERF_MEASURE_REQUESTS=%q\n' "${PERF_MEASURE_REQUESTS}"
        printf 'PERF_DECODE_IMPLEMENTATION=%q\n' "${PERF_DECODE_IMPLEMENTATION}"
        printf 'PERF_MAX_TTFT_RATIO=%q\n' "${PERF_MAX_TTFT_RATIO}"
        printf 'PERF_MAX_TPOT_RATIO=%q\n' "${PERF_MAX_TPOT_RATIO}"
        printf 'PERF_MIN_THROUGHPUT_RATIO=%q\n' "${PERF_MIN_THROUGHPUT_RATIO}"
        printf 'PERF_MIN_KV_CAPACITY_RATIO=%q\n' "${PERF_MIN_KV_CAPACITY_RATIO}"
    } >"${CONFIG_FILE}"
}

start_npu_monitor() {
    local stage="$1"
    local output="$2"
    if ! command -v npu-smi >/dev/null 2>&1; then
        printf 'npu-smi is unavailable; no periodic NPU snapshots.\n' >"${output}"
        return
    fi
    (
        while true; do
            printf '\n===== %s stage=%s =====\n' \
                "$(date --iso-8601=seconds 2>/dev/null || date)" "${stage}"
            if command -v timeout >/dev/null 2>&1; then
                timeout 20s npu-smi info || true
            else
                npu-smi info || true
            fi
            sleep "${NPU_POLL_INTERVAL}"
        done
    ) >"${output}" 2>&1 &
    NPU_MONITOR_PID=$!
}

stop_npu_monitor() {
    if [[ -n "${NPU_MONITOR_PID}" ]] && kill -0 "${NPU_MONITOR_PID}" 2>/dev/null; then
        kill "${NPU_MONITOR_PID}" 2>/dev/null || true
        wait "${NPU_MONITOR_PID}" 2>/dev/null || true
    fi
    NPU_MONITOR_PID=""
}

run_stage() {
    local stage="$1"
    local description="$2"
    shift 2

    local stage_log="${RUN_ROOT}/${stage}.log"
    local npu_log="${RUN_ROOT}/${stage}_npu.log"
    local status

    printf '\n===== %s: %s =====\n' "${stage}" "${description}" | tee -a "${COMBINED_LOG}"
    start_npu_monitor "${stage}" "${npu_log}"
    set +e
    "$@" 2>&1 | tee "${stage_log}" | tee -a "${COMBINED_LOG}"
    status=${PIPESTATUS[0]}
    set -e
    stop_npu_monitor
    LAST_STAGE_STATUS=${status}

    if [[ ${status} -eq 0 ]]; then
        printf '%s\tPASS\t0\t%s\t%s\n' \
            "${stage}" "$(basename "${stage_log}")" "$(basename "${npu_log}")" >>"${RESULTS_FILE}"
        printf '===== %s: PASS =====\n' "${stage}" | tee -a "${COMBINED_LOG}"
    else
        printf '%s\tFAIL\t%s\t%s\t%s\n' \
            "${stage}" "${status}" "$(basename "${stage_log}")" "$(basename "${npu_log}")" >>"${RESULTS_FILE}"
        printf '===== %s: FAIL (exit %s) =====\n' "${stage}" "${status}" | tee -a "${COMBINED_LOG}"
        FAILURES=$((FAILURES + 1))
        if [[ "${FAIL_FAST}" == "1" ]]; then
            exit "${status}"
        fi
    fi
}

skip_stage() {
    local stage="$1"
    local reason="$2"
    printf '%s\tSKIP\t0\t%s\t-\n' "${stage}" "${reason}" >>"${RESULTS_FILE}"
}

write_summary() {
    {
        printf 'TurboQuant full Ascend 910B4 validation\n'
        printf 'Generated: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
        printf 'Repository: %s\n' "${REPO_ROOT}"
        printf 'Model: %s\n' "${MODEL}"
        printf 'Visible devices: %s\n' "${VISIBLE_DEVICES}"
        printf 'Tensor parallel size: %s\n' "${TP_SIZE}"
        printf 'Failed stages: %s\n\n' "${FAILURES}"
        column -t -s $'\t' "${RESULTS_FILE}" 2>/dev/null || cat "${RESULTS_FILE}"
        printf '\nConfiguration: %s\n' "${CONFIG_FILE}"
        printf 'Combined log: %s\n' "${COMBINED_LOG}"
    } >"${SUMMARY_FILE}"
    cat "${SUMMARY_FILE}"
}

finalize() {
    local status=$?
    if [[ "${FINALIZED}" == "1" ]]; then
        exit "${status}"
    fi
    FINALIZED=1
    trap - EXIT INT TERM
    stop_npu_monitor
    write_summary
    if ! tar -czf "${RUN_ROOT}.tar.gz" \
        -C "$(dirname "${RUN_ROOT}")" "$(basename "${RUN_ROOT}")"; then
        printf 'Failed to create validation archive.\n' >&2
        status=1
    fi
    printf 'Validation output: %s\n' "${RUN_ROOT}"
    printf 'Validation archive: %s.tar.gz\n' "${RUN_ROOT}"
    if [[ ${status} -eq 0 && ${FAILURES} -ne 0 ]]; then
        status=1
    fi
    exit "${status}"
}

collect_preflight() {
    local concurrency
    local numeric_name
    local numeric_value
    local concurrency_values=()
    local seen_concurrencies=" "
    for numeric_name in \
        PERF_MAX_MODEL_LEN PERF_INPUT_TOKENS PERF_OUTPUT_TOKENS \
        PERF_MEASURE_REQUESTS PERF_WARMUP_REQUESTS; do
        numeric_value="${!numeric_name}"
        if [[ ! "${numeric_value}" =~ ^[0-9]+$ ]]; then
            printf '%s must be a non-negative integer; got %s.\n' \
                "${numeric_name}" "${numeric_value}" >&2
            return 2
        fi
    done
    if ((
        PERF_MAX_MODEL_LEN == 0 || PERF_INPUT_TOKENS == 0 ||
            PERF_OUTPUT_TOKENS == 0 || PERF_MEASURE_REQUESTS == 0
    )); then
        printf 'Performance model length, token counts, and measured requests must be positive.\n' >&2
        return 2
    fi
    read -r -a concurrency_values <<<"${PERF_CONCURRENCY_LEVELS}"
    if ((${#concurrency_values[@]} == 0)); then
        printf 'PERF_CONCURRENCY_LEVELS must contain at least one positive integer.\n' >&2
        return 2
    fi
    for concurrency in "${concurrency_values[@]}"; do
        if [[ ! "${concurrency}" =~ ^[1-9][0-9]*$ ]]; then
            printf 'Invalid performance concurrency: %s\n' "${concurrency}" >&2
            return 2
        fi
        if [[ "${seen_concurrencies}" == *" ${concurrency} "* ]]; then
            printf 'Duplicate performance concurrency: %s\n' "${concurrency}" >&2
            return 2
        fi
        seen_concurrencies+="${concurrency} "
        if ((PERF_MEASURE_REQUESTS < concurrency)); then
            printf 'PERF_MEASURE_REQUESTS=%s is smaller than concurrency %s.\n' \
                "${PERF_MEASURE_REQUESTS}" "${concurrency}" >&2
            return 2
        fi
    done
    if [[ ! -d "/sys/class/net/${NETWORK_IFNAME}" ]]; then
        printf 'Network interface does not exist: %s\n' "${NETWORK_IFNAME}" >&2
        return 2
    fi
    if ((PERF_INPUT_TOKENS + PERF_OUTPUT_TOKENS > PERF_MAX_MODEL_LEN)); then
        printf 'Performance token budget exceeds model length: %s + %s > %s\n' \
            "${PERF_INPUT_TOKENS}" "${PERF_OUTPUT_TOKENS}" "${PERF_MAX_MODEL_LEN}" >&2
        return 2
    fi
    for command in python3 git curl pytest; do
        if ! command -v "${command}" >/dev/null 2>&1; then
            printf 'Required command is unavailable: %s\n' "${command}" >&2
            return 2
        fi
    done

    python3 - \
        "${REPO_ROOT}" "${MODEL}" "${TP_SIZE}" "${PORT}" \
        "${PERF_MAX_TTFT_RATIO}" "${PERF_MAX_TPOT_RATIO}" \
        "${PERF_MIN_THROUGHPUT_RATIO}" "${PERF_MIN_KV_CAPACITY_RATIO}" <<'PY'
import importlib.metadata
import math
import socket
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from transformers import AutoConfig

import vllm
import vllm_ascend

repo = Path(sys.argv[1]).resolve()
model = Path(sys.argv[2]).resolve()
tp_size = int(sys.argv[3])
port = int(sys.argv[4])
gate_names = (
    "PERF_MAX_TTFT_RATIO",
    "PERF_MAX_TPOT_RATIO",
    "PERF_MIN_THROUGHPUT_RATIO",
    "PERF_MIN_KV_CAPACITY_RATIO",
)
for name, raw_value in zip(gate_names, sys.argv[5:], strict=True):
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be finite and positive; got {raw_value}")

if not (model / "config.json").is_file():
    raise RuntimeError(f"Model config is not readable: {model / 'config.json'}")
ascend_source = Path(vllm_ascend.__file__).resolve()
if repo not in ascend_source.parents:
    raise RuntimeError(
        f"Loaded vllm_ascend is not this checkout: {ascend_source}; "
        f"run `pip install -e {repo}` first"
    )
if not torch.npu.is_available():
    raise RuntimeError("torch-npu cannot see an Ascend NPU")
device_count = torch.npu.device_count()
if device_count < tp_size:
    raise RuntimeError(f"Visible NPU count {device_count} is smaller than TP size {tp_size}")
with socket.socket() as sock:
    sock.settimeout(1)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise RuntimeError(f"Port {port} is already in use")

config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=True)
heads = getattr(config, "num_attention_heads", None)
kv_heads = getattr(config, "num_key_value_heads", heads)
head_dim = getattr(config, "head_dim", None)
if head_dim is None and heads:
    head_dim = config.hidden_size // heads
if heads and heads % tp_size:
    raise RuntimeError(f"Attention heads {heads} are not divisible by TP size {tp_size}")

print(f"vLLM={vllm.__version__} source={vllm.__file__}")
print(f"vLLM Ascend source={ascend_source}")
print(f"torch={torch.__version__}")
print(f"torch-npu={importlib.metadata.version('torch-npu')}")
print(f"visible_npus={device_count} tp_size={tp_size}")
for index in range(device_count):
    print(f"npu[{index}]={torch.npu.get_device_name(index)}")
print(
    f"model_type={config.model_type} architecture={config.architectures} "
    f"heads={heads} kv_heads={kv_heads} head_dim={head_dim}"
)
PY
    local python_status=$?
    if ((python_status != 0)); then
        return "${python_status}"
    fi

    printf '\nFilesystem capacity:\n'
    df -h "${REPO_ROOT}" "${MODEL}"
    printf '\nProcess limits:\n'
    ulimit -a
    printf '\nSelected network interface:\n'
    ip addr show "${NETWORK_IFNAME}" 2>/dev/null || true
    printf '\nNPU inventory:\n'
    npu-smi info 2>/dev/null || true
}

collect_source() {
    git branch --show-current
    git rev-parse HEAD
    git status --short
    git log -5 --oneline --decorate
    printf '\nKey source checksums:\n'
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum \
            csrc/attention/turbo_quant_paged_dequant/op_kernel/turbo_quant_paged_dequant.cpp \
            csrc/attention/turbo_quant_paged_dequant/op_host/turbo_quant_paged_dequant_tiling.cpp \
            vllm_ascend/attention/turboquant.py \
            vllm_ascend/ops/turboquant.py \
            vllm_ascend/ops/triton/turboquant_store.py \
            vllm_ascend/ops/triton/turboquant_decode.py
    else
        shasum -a 256 \
            csrc/attention/turbo_quant_paged_dequant/op_kernel/turbo_quant_paged_dequant.cpp \
            csrc/attention/turbo_quant_paged_dequant/op_host/turbo_quant_paged_dequant_tiling.cpp \
            vllm_ascend/attention/turboquant.py \
            vllm_ascend/ops/turboquant.py \
            vllm_ascend/ops/triton/turboquant_store.py \
            vllm_ascend/ops/triton/turboquant_decode.py
    fi
    printf '\nInstalled packages:\n'
    python3 -m pip show torch torch-npu triton-ascend vllm vllm-ascend || true
}

run_operator_correctness() {
    local output_dir="${RUN_ROOT}/operator_correctness"
    local failures=0
    mkdir -p "${output_dir}"

    run_operator_case() {
        local case_name="$1"
        shift
        printf '\n--- operator case: %s ---\n' "${case_name}"
        if ! env ASCEND_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1 \
            python3 -u "${REPO_ROOT}/scripts/turboquant_operators/operator_accuracy.py" \
            --device 0 \
            --output "${output_dir}/${case_name}.json" \
            "$@"; then
            printf 'Operator case failed: %s\n' "${case_name}" >&2
            failures=$((failures + 1))
        fi
    }

    run_operator_case 4bit_fp16 \
        --cache-dtype turboquant_4bit_nc \
        --activation-dtype float16 \
        --sequence-lengths 133 65 \
        --num-query-heads 16 --num-kv-heads 2 --head-dim 128 --num-kv-splits 4
    run_operator_case k3v4_fp16 \
        --cache-dtype turboquant_k3v4_nc \
        --activation-dtype float16 \
        --sequence-lengths 133 65 \
        --num-query-heads 16 --num-kv-heads 2 --head-dim 128 --num-kv-splits 4
    run_operator_case 3bit_fp16 \
        --cache-dtype turboquant_3bit_nc \
        --activation-dtype float16 \
        --sequence-lengths 133 65 \
        --num-query-heads 16 --num-kv-heads 2 --head-dim 128 --num-kv-splits 4
    run_operator_case qwen3_bf16 \
        --cache-dtype turboquant_4bit_nc \
        --activation-dtype bfloat16 \
        --sequence-lengths 1 17 129 \
        --num-query-heads 16 --num-kv-heads 8 --head-dim 128 --num-kv-splits 1

    if ! python3 "${REPO_ROOT}/scripts/turboquant_operators/summarize_results.py" \
        --result-root "${output_dir}" \
        --output-prefix "${output_dir}/summary"; then
        printf 'Failed to summarize operator correctness reports.\n' >&2
        failures=$((failures + 1))
    fi

    if ((failures > 0)); then
        printf '%s operator correctness case(s) failed.\n' "${failures}" >&2
        return 1
    fi
}

trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_configuration
cd "${REPO_ROOT}"

printf 'This full run starts multiple four-card model servers and can take several hours.\n'
printf 'Set QUICK=1 or individual RUN_* variables to reduce the matrix.\n'

run_stage "00_preflight" "model, source checkout, NPU, network, port, and token-budget validation" collect_preflight
if [[ ${LAST_STAGE_STATUS} -ne 0 ]]; then
    exit 2
fi
run_stage "01_source" "source revision, checksums, and installed package metadata" collect_source
run_stage \
    "02_environment" \
    "vLLM 0.20.2 API, packed cache spec, workspace, and NPU checks" \
    python3 "${TQ_COMMON_DIR}/check_environment.py"

if [[ "${RUN_RUNTIME_PROBE}" == "1" ]]; then
    run_stage \
        "02a_runtime_probe" \
        "isolated synchronous ACLNN launch, output aliasing, and loaded-library fingerprints" \
        env \
        ASCEND_LAUNCH_BLOCKING=1 \
        PYTHONFAULTHANDLER=1 \
        python3 -u "${REPO_ROOT}/scripts/turboquant_operators/operator_runtime_probe.py" \
        --device 0 \
        --cache-dtype "${TQ_CACHE_DTYPE}" \
        --activation-dtype "${ACTIVATION_DTYPE}" \
        --output "${RUN_ROOT}/operator_runtime_probe.json"
else
    skip_stage "02a_runtime_probe" "RUN_RUNTIME_PROBE=0"
fi

if [[ "${RUN_OPERATOR_CORRECTNESS}" == "1" ]]; then
    run_stage \
        "02b_operator_correctness" \
        "independent CPU, Triton, AscendC, attention, and quantization-quality gates" \
        run_operator_correctness
else
    skip_stage "02b_operator_correctness" "RUN_OPERATOR_CORRECTNESS=0"
fi

if [[ "${RUN_KERNELS}" == "1" ]]; then
    run_stage \
        "03_kernel_suite" \
        "isolated store presets, grouped/reference decode, edge cases, metadata, and ACLGraph" \
        env \
        LOG_ROOT="${RUN_ROOT}/kernel_suite" \
        MODEL="${MODEL}" \
        TP_SIZE="${TP_SIZE}" \
        RUN_MODEL_SMOKE=0 \
        RUN_ACLGRAPH=1 \
        DEBUG_SYNC=1 \
        bash "${TQ_DIAGNOSTICS_DIR}/run_910b4_retest.sh"
else
    skip_stage "03_kernel_suite" "RUN_KERNELS=0"
fi

if [[ "${RUN_OPERATOR_PROFILE}" == "1" ]]; then
    run_stage \
        "04_operator_profile" \
        "Qwen3-32B TP4 B16/S16K grouped-16, grouped-32, reference, store, and native decode" \
        env \
        OUTPUT_DIR="${RUN_ROOT}/operator_profile" \
        DEVICE=0 \
        ACTIVATION_DTYPE="${ACTIVATION_DTYPE}" \
        WARMUP=5 \
        ITERATIONS="${PROFILE_ITERATIONS}" \
        COLLECT_TRACE="${COLLECT_PROFILE_TRACE}" \
        bash "${TQ_PERFORMANCE_DIR}/profile_qwen3_32b_tp4.sh"
else
    skip_stage "04_operator_profile" "RUN_OPERATOR_PROFILE=0"
fi

if [[ "${RUN_ACCURACY}" == "1" ]]; then
    run_stage \
        "05_accuracy_native_auto" \
        "deterministic completion text/logprob comparison with prefix-cache reuse" \
        env -u ASCEND_LAUNCH_BLOCKING \
        MODEL="${MODEL}" \
        PORT="${PORT}" \
        TP_SIZE="${TP_SIZE}" \
        MAX_MODEL_LEN="${ACCURACY_MAX_MODEL_LEN}" \
        MAX_NUM_SEQS="${ACCURACY_MAX_NUM_SEQS}" \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        OUTPUT_DIR="${RUN_ROOT}/accuracy_native_auto" \
        BASE_LABEL=native \
        BASE_CACHE_DTYPE=auto \
        BASE_DECODE_IMPLEMENTATION=auto \
        TEST_LABEL=turboquant_auto \
        TEST_CACHE_DTYPE="${TQ_CACHE_DTYPE}" \
        TEST_DECODE_IMPLEMENTATION=auto \
        BASE_ENFORCE_EAGER=1 \
        TEST_ENFORCE_EAGER=1 \
        WARM_PREFIX=1 \
        MIN_TOKEN_PREFIX_RATE="${ACCURACY_MIN_PREFIX_RATE}" \
        SERVER_TIMEOUT="${SERVER_TIMEOUT}" \
        REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
        bash "${TQ_CORRECTNESS_DIR}/run_accuracy_comparison.sh"
else
    skip_stage "05_accuracy_native_auto" "RUN_ACCURACY=0"
fi

if [[ "${RUN_QUALITY}" == "1" ]]; then
    QUALITY_COMMON=(
        MODEL="${MODEL}"
        PORT="${PORT}"
        TP_SIZE="${TP_SIZE}"
        MAX_MODEL_LEN="${QUALITY_MAX_MODEL_LEN}"
        MAX_NUM_SEQS=4
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}"
        PROMPTS="${TQ_CORRECTNESS_DIR}/quality_cases.jsonl"
        REQUEST_MODE=chat
        WARM_PREFIX=0
        MIN_TURBOQUANT_ACCURACY="${QUALITY_MIN_ACCURACY}"
        MAX_QUALITY_REGRESSIONS="${QUALITY_MAX_REGRESSIONS}"
        SERVER_TIMEOUT="${SERVER_TIMEOUT}"
        REQUEST_TIMEOUT="${REQUEST_TIMEOUT}"
    )
    run_stage \
        "06_quality_native_reference" \
        "graded native versus FP32/per-query-head reference quality" \
        env -u ASCEND_LAUNCH_BLOCKING \
        "${QUALITY_COMMON[@]}" \
        OUTPUT_DIR="${RUN_ROOT}/quality_native_reference" \
        BASE_LABEL=native \
        BASE_CACHE_DTYPE=auto \
        BASE_DECODE_IMPLEMENTATION=auto \
        TEST_LABEL=turboquant_reference \
        TEST_CACHE_DTYPE="${TQ_CACHE_DTYPE}" \
        TEST_DECODE_IMPLEMENTATION=reference \
        bash "${TQ_CORRECTNESS_DIR}/run_accuracy_comparison.sh"
    run_stage \
        "07_quality_reference_auto" \
        "graded FP32 reference versus grouped/low-precision auto quality" \
        env -u ASCEND_LAUNCH_BLOCKING \
        "${QUALITY_COMMON[@]}" \
        OUTPUT_DIR="${RUN_ROOT}/quality_reference_auto" \
        BASE_LABEL=turboquant_reference \
        BASE_CACHE_DTYPE="${TQ_CACHE_DTYPE}" \
        BASE_DECODE_IMPLEMENTATION=reference \
        TEST_LABEL=turboquant_auto \
        TEST_CACHE_DTYPE="${TQ_CACHE_DTYPE}" \
        TEST_DECODE_IMPLEMENTATION=auto \
        bash "${TQ_CORRECTNESS_DIR}/run_accuracy_comparison.sh"
else
    skip_stage "06_quality_native_reference" "RUN_QUALITY=0"
    skip_stage "07_quality_reference_auto" "RUN_QUALITY=0"
fi

if [[ "${RUN_GRAPH}" == "1" ]]; then
    SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":1,"prompt_lookup_max":4}'
    run_stage \
        "08_aclgraph_spec" \
        "TurboQuant eager versus uniform multi-token ACLGraph/speculative replay" \
        env -u ASCEND_LAUNCH_BLOCKING \
        MODEL="${MODEL}" \
        PORT="${PORT}" \
        TP_SIZE="${TP_SIZE}" \
        MAX_MODEL_LEN="${GRAPH_MAX_MODEL_LEN}" \
        MAX_NUM_SEQS=8 \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        OUTPUT_DIR="${RUN_ROOT}/aclgraph_spec" \
        BASE_LABEL=turboquant_spec_eager \
        BASE_CACHE_DTYPE="${TQ_CACHE_DTYPE}" \
        BASE_DECODE_IMPLEMENTATION=auto \
        BASE_ENFORCE_EAGER=1 \
        BASE_SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG}" \
        TEST_LABEL=turboquant_spec_aclgraph \
        TEST_CACHE_DTYPE="${TQ_CACHE_DTYPE}" \
        TEST_DECODE_IMPLEMENTATION=auto \
        TEST_ENFORCE_EAGER=0 \
        TEST_SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG}" \
        WARM_PREFIX=0 \
        MIN_TOKEN_PREFIX_RATE="${GRAPH_MIN_PREFIX_RATE}" \
        SERVER_TIMEOUT="${SERVER_TIMEOUT}" \
        REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
        bash "${TQ_CORRECTNESS_DIR}/run_accuracy_comparison.sh"
else
    skip_stage "08_aclgraph_spec" "RUN_GRAPH=0"
fi

SERVING_COMMON=(
    MODEL="${MODEL}"
    PORT="${PORT}"
    TP_SIZE="${TP_SIZE}"
    MAX_MODEL_LEN="${PERF_MAX_MODEL_LEN}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}"
    INPUT_TOKENS="${PERF_INPUT_TOKENS}"
    OUTPUT_TOKENS="${PERF_OUTPUT_TOKENS}"
    CONCURRENCY="${PERF_CONCURRENCY}"
    CONCURRENCY_LEVELS="${PERF_CONCURRENCY_LEVELS}"
    WARMUP_REQUESTS="${PERF_WARMUP_REQUESTS}"
    MEASURE_REQUESTS="${PERF_MEASURE_REQUESTS}"
    REQUEST_TIMEOUT="${REQUEST_TIMEOUT}"
    SERVER_TIMEOUT="${SERVER_TIMEOUT}"
    NATIVE_CACHE_DTYPE=auto
    TQ_CACHE_DTYPE="${TQ_CACHE_DTYPE}"
    VLLM_ASCEND_TURBOQUANT_DECODE_IMPLEMENTATION="${PERF_DECODE_IMPLEMENTATION}"
    MAX_TTFT_RATIO="${PERF_MAX_TTFT_RATIO}"
    MAX_TPOT_RATIO="${PERF_MAX_TPOT_RATIO}"
    MIN_THROUGHPUT_RATIO="${PERF_MIN_THROUGHPUT_RATIO}"
    MIN_KV_CAPACITY_RATIO="${PERF_MIN_KV_CAPACITY_RATIO}"
)

if [[ "${RUN_SERVING_EAGER}" == "1" ]]; then
    run_stage \
        "09_serving_eager" \
        "native versus TurboQuant eager B1/B16 long-context TTFT, TPOT, throughput, and cache capacity" \
        env -u ASCEND_LAUNCH_BLOCKING \
        "${SERVING_COMMON[@]}" \
        ENFORCE_EAGER=1 \
        OUTPUT_DIR="${RUN_ROOT}/serving_eager" \
        bash "${TQ_PERFORMANCE_DIR}/run_serving_benchmark.sh"
else
    skip_stage "09_serving_eager" "RUN_SERVING_EAGER=0"
fi

if [[ "${RUN_SERVING_GRAPH}" == "1" ]]; then
    run_stage \
        "10_serving_aclgraph" \
        "native versus TurboQuant ACLGraph B1/B16 long-context TTFT, TPOT, throughput, and cache capacity" \
        env -u ASCEND_LAUNCH_BLOCKING \
        "${SERVING_COMMON[@]}" \
        ENFORCE_EAGER=0 \
        OUTPUT_DIR="${RUN_ROOT}/serving_aclgraph" \
        bash "${TQ_PERFORMANCE_DIR}/run_serving_benchmark.sh"
else
    skip_stage "10_serving_aclgraph" "RUN_SERVING_GRAPH=0"
fi

exit 0
