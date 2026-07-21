#!/usr/bin/env bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

OPERATOR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${OPERATOR_SCRIPT_DIR}/../.." && pwd)"
TQ_SCRIPT_ROOT="${REPO_ROOT}/scripts/turboquant_triton"
TQ_PROFILE_SCRIPT="${TQ_SCRIPT_ROOT}/performance/profile_kernels.py"
TQ_ENVIRONMENT_SCRIPT="${TQ_SCRIPT_ROOT}/common/check_environment.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-0}"
SOC_VERSION="${SOC_VERSION:-ascend910b4}"

export SOC_VERSION

readonly OPERATOR_SCRIPT_DIR REPO_ROOT TQ_SCRIPT_ROOT TQ_PROFILE_SCRIPT
readonly TQ_ENVIRONMENT_SCRIPT PYTHON_BIN DEVICE

check_910b4_target() {
    case "${SOC_VERSION}" in
        ascend910b4|ascend910b4-1)
            ;;
        *)
            printf 'WARNING: SOC_VERSION=%s; this suite defaults to Ascend 910B4.\n' "${SOC_VERSION}"
            ;;
    esac
}

collect_environment() {
    local output_dir=$1
    mkdir -p "${output_dir}"
    {
        printf 'timestamp=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
        printf 'repository=%s\n' "${REPO_ROOT}"
        printf 'revision=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'branch=%s\n' "$(git -C "${REPO_ROOT}" branch --show-current)"
        printf 'SOC_VERSION=%s\n' "${SOC_VERSION}"
        printf 'DEVICE=%s\n' "${DEVICE}"
        printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
        printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-<unset>}"
        printf 'ASCEND_TOOLKIT_HOME=%s\n' "${ASCEND_TOOLKIT_HOME:-<unset>}"
    } > "${output_dir}/environment.txt"

    check_910b4_target | tee -a "${output_dir}/environment.txt"
    "${PYTHON_BIN}" "${TQ_ENVIRONMENT_SCRIPT}" | tee -a "${output_dir}/environment.txt"
    "${PYTHON_BIN}" - <<'PY' | tee -a "${output_dir}/environment.txt"
from vllm_ascend.ops.turboquant import has_turboquant_paged_dequant

if not has_turboquant_paged_dequant():
    raise RuntimeError(
        "TurboQuant paged-dequant return/out schemas are not registered. Set "
        "SOC_VERSION=ascend910b4, clean build/csrc/build, and reinstall."
    )
print("npu_turboquant_paged_dequant=registered (return+out)")
PY

    if command -v npu-smi >/dev/null 2>&1; then
        npu-smi info > "${output_dir}/npu_smi_before.txt" 2>&1 || true
    fi
}

archive_output() {
    local output_dir=$1
    local archive_path="${output_dir}.tar.gz"
    tar -czf "${archive_path}" -C "$(dirname "${output_dir}")" "$(basename "${output_dir}")"
    printf 'Results: %s\n' "${output_dir}"
    printf 'Archive: %s\n' "${archive_path}"
}
