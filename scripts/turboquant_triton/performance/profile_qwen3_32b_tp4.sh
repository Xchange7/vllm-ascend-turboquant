#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/turboquant/qwen3_32b_tp4_${TIMESTAMP}}"
DEVICE="${DEVICE:-0}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bfloat16}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-50}"
COLLECT_TRACE="${COLLECT_TRACE:-0}"

COMMON_ARGS=(
    --cache-dtype turboquant_4bit_nc
    --activation-dtype "${ACTIVATION_DTYPE}"
    --batch-size 16
    --sequence-length 16384
    --store-tokens 16
    --num-query-heads 16
    --num-kv-heads 2
    --head-dim 128
    --num-kv-splits 32
    --device "${DEVICE}"
    --warmup "${WARMUP}"
    --iterations "${ITERATIONS}"
)

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
python3 "${TQ_COMMON_DIR}/check_environment.py" | tee "${OUTPUT_DIR}/environment.log"

run_case() {
    local name="$1"
    local operation="$2"
    local implementation="$3"
    local block_kv="$4"
    shift 4

    local case_dir="${OUTPUT_DIR}/${name}"
    local trace_args=(--no-trace)
    if [[ "${COLLECT_TRACE}" == "1" ]]; then
        trace_args=()
    fi
    mkdir -p "${case_dir}"
    python3 "${SCRIPT_DIR}/profile_kernels.py" \
        "${COMMON_ARGS[@]}" \
        --operation "${operation}" \
        --decode-implementation "${implementation}" \
        --grouped-block-kv "${block_kv}" \
        --trace-dir "${case_dir}" \
        "${trace_args[@]}" \
        "$@" 2>&1 | tee "${OUTPUT_DIR}/${name}.log"
}

# Pure decode isolates the packed attention kernel and includes the native
# paged-attention baseline. decode_step additionally measures current-token
# TurboQuant store overhead, matching one attention layer in generation.
run_case grouped_b16_decode decode auto 16 --native-baseline
run_case grouped_b16_decode_step decode_step auto 16
run_case grouped_b32_decode decode auto 32 --native-baseline
run_case grouped_b32_decode_step decode_step auto 32
run_case reference_decode decode reference 16 --native-baseline
run_case reference_decode_step decode_step reference 16

printf 'Qwen3-32B TP4 profile output: %s\n' "${OUTPUT_DIR}"
