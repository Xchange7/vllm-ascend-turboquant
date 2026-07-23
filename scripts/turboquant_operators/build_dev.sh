#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
source "${SCRIPT_DIR}/soc_utils.sh"
SOC_VERSION="${SOC_VERSION:-}"
DETECTED_SOC_VERSION=""
MAX_JOBS="${MAX_JOBS:-8}"
INSTALL_DIR="${REPO_ROOT}/vllm_ascend/_cann_ops_custom"
BACKUP_DIR="${TURBOQUANT_DEV_BACKUP_DIR:-${REPO_ROOT}/.cache/turboquant_build_dev/full_cann_ops_custom}"
MARKER_FILE="${INSTALL_DIR}/.turboquant_dev_only"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${TURBOQUANT_BUILD_LOG_DIR:-${REPO_ROOT}/logs/turboquant/build_dev_${TIMESTAMP}}"
LOG_FILE="${LOG_DIR}/build.log"
MODE="build"
CLEAN_BUILD=0

usage() {
    cat <<'EOF'
Usage: bash scripts/turboquant_operators/build_dev.sh [--clean|--restore-full]

Build mode compiles and installs the TurboQuant dequant and attention ops plus the
vllm-ascend PyTorch extension for the selected A2/A3 SoC. The existing full
custom-op package is backed up before the first developer build.

Options:
  --clean         Discard csrc/build and perform a clean one-op build.
  --restore-full  Restore the custom-op package saved before developer mode.
  -h, --help      Show this help.
EOF
}

if (($# > 1)); then
    usage >&2
    exit 2
fi
if (($# == 1)); then
    case "$1" in
        --restore-full)
            MODE="restore"
            ;;
        --clean)
            CLEAN_BUILD=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
fi

restore_full_package() {
    if [[ ! -d "${BACKUP_DIR}/vendors" ]]; then
        printf 'No full custom-op backup found at %s\n' "${BACKUP_DIR}" >&2
        return 1
    fi
    rm -rf -- "${INSTALL_DIR}"
    mkdir -p -- "$(dirname -- "${INSTALL_DIR}")"
    cp -a -- "${BACKUP_DIR}" "${INSTALL_DIR}"
    rm -f -- "${MARKER_FILE}"
    printf 'Restored full custom-op package from %s\n' "${BACKUP_DIR}"
    printf 'Restart all vLLM processes before using the restored package.\n'
}

if [[ "${MODE}" == "restore" ]]; then
    restore_full_package
    exit
fi

if [[ -z "${SOC_VERSION}" ]]; then
    SOC_VERSION="$(detect_turboquant_soc_version "${PYTHON_BIN}")"
    DETECTED_SOC_VERSION="${SOC_VERSION}"
else
    SOC_VERSION="$(resolve_turboquant_soc_version "${SOC_VERSION}" "${PYTHON_BIN}")"
fi

case "${SOC_VERSION}" in
    ascend910b4|ascend910b4-1|ascend910_93*)
        ;;
    *)
        printf 'SOC_VERSION must target Ascend 910B4 or 910_93, got %s\n' "${SOC_VERSION}" >&2
        exit 2
        ;;
esac
validate_turboquant_soc_matches_device "${SOC_VERSION}" "${PYTHON_BIN}" "${DETECTED_SOC_VERSION}"
if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_JOBS must be a positive integer, got %s\n' "${MAX_JOBS}" >&2
    exit 2
fi
if ! command -v bisheng >/dev/null 2>&1; then
    printf 'bisheng was not found. Source the CANN set_env.sh before running this script.\n' >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    printf '%s cannot run pip.\n' "${PYTHON_BIN}" >&2
    exit 1
fi
if ! TRITON_ASCEND_VERSION="$("${PYTHON_BIN}" -c \
    'import importlib.metadata; print(importlib.metadata.version("triton-ascend"))')"; then
    printf '%s cannot find triton-ascend in the active Python environment.\n' "${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p -- "${LOG_DIR}" "$(dirname -- "${BACKUP_DIR}")"

# The CANN installer replaces aggregate op_impl/op_api directories. Preserve
# the complete package once, instead of allowing a one-op package to destroy it.
if [[ -d "${INSTALL_DIR}/vendors" && ! -f "${MARKER_FILE}" ]]; then
    rm -rf -- "${BACKUP_DIR}"
    cp -a -- "${INSTALL_DIR}" "${BACKUP_DIR}"
    printf 'Backed up full custom-op package to %s\n' "${BACKUP_DIR}"
fi

restore_after_failure() {
    local status=$?
    trap - ERR
    printf 'TurboQuant developer build failed with status %d.\n' "${status}" >&2
    if [[ -d "${BACKUP_DIR}/vendors" ]]; then
        printf 'Restoring the previous full custom-op package.\n' >&2
        restore_full_package || true
    fi
    exit "${status}"
}
trap restore_after_failure ERR

export MAX_JOBS
export SOC_VERSION
export VLLM_ASCEND_BUILD_CUSTOM_OPS="turbo_quant_paged_dequant;turbo_quant_paged_attention"
if ((CLEAN_BUILD == 0)); then
    export VLLM_ASCEND_ACLNN_INCREMENTAL_BUILD=1
else
    export VLLM_ASCEND_ACLNN_INCREMENTAL_BUILD=0
fi

{
    printf 'repository=%s\n' "${REPO_ROOT}"
    printf 'revision=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    printf 'SOC_VERSION=%s\n' "${SOC_VERSION}"
    printf 'MAX_JOBS=%s\n' "${MAX_JOBS}"
    printf 'VLLM_ASCEND_BUILD_CUSTOM_OPS=%s\n' "${VLLM_ASCEND_BUILD_CUSTOM_OPS}"
    printf 'VLLM_ASCEND_ACLNN_INCREMENTAL_BUILD=%s\n' "${VLLM_ASCEND_ACLNN_INCREMENTAL_BUILD}"
    printf 'python=%s\n' "$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
    printf 'triton-ascend=%s\n' "${TRITON_ASCEND_VERSION}"
    printf 'pip_build_isolation=disabled\n'
    printf '\n== Editable TurboQuant developer build ==\n'
} | tee "${LOG_FILE}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m pip install --no-deps --no-build-isolation -v -e . 2>&1 | tee -a "${LOG_FILE}"

touch "${MARKER_FILE}"
trap - ERR

printf '\nTurboQuant developer build completed.\n' | tee -a "${LOG_FILE}"
printf 'Log: %s\n' "${LOG_FILE}" | tee -a "${LOG_FILE}"
printf 'This installation contains only the TurboQuant custom ACLNN package.\n'
printf 'Run the operator smoke test next:\n'
printf '  bash scripts/turboquant_operators/run_smoke.sh\n'
printf 'Restore the previous full package with:\n'
printf '  bash scripts/turboquant_operators/build_dev.sh --restore-full\n'
