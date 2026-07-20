#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/paths.sh"
PROFILE_ROOT="${PROFILE_ROOT:-${REPO_ROOT}/profiles/turboquant_matrix}"
ITERATIONS="${ITERATIONS:-20}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4096}"

cd "${REPO_ROOT}"
python3 "${TQ_COMMON_DIR}/check_environment.py"
mkdir -p "${PROFILE_ROOT}"

run_case() {
    local name="$1"
    shift
    python3 "${TQ_PERFORMANCE_DIR}/profile_kernels.py" \
        --operation all \
        --native-baseline \
        --sequence-length "${SEQUENCE_LENGTH}" \
        --iterations "${ITERATIONS}" \
        --warmup 5 \
        --no-trace \
        --trace-dir "${PROFILE_ROOT}/${name}" \
        "$@"
}

for cache_dtype in turboquant_4bit_nc turboquant_k3v4_nc turboquant_3bit_nc; do
    for activation_dtype in float16 bfloat16; do
        run_case "${cache_dtype}_${activation_dtype}_d128_s8" \
            --cache-dtype "${cache_dtype}" \
            --activation-dtype "${activation_dtype}" \
            --head-dim 128 \
            --num-kv-splits 8
    done
done

for head_dim in 64 256; do
    run_case "turboquant_4bit_nc_float16_d${head_dim}_s8" \
        --cache-dtype turboquant_4bit_nc \
        --activation-dtype float16 \
        --head-dim "${head_dim}" \
        --num-kv-splits 8
done

for num_splits in 16 32; do
    run_case "turboquant_4bit_nc_float16_d128_s${num_splits}" \
        --cache-dtype turboquant_4bit_nc \
        --activation-dtype float16 \
        --head-dim 128 \
        --num-kv-splits "${num_splits}"
done

printf 'Reports: %s\n' "${PROFILE_ROOT}"
