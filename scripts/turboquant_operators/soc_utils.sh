#!/usr/bin/env bash

# Shared A2/A3 SoC selection helpers. Keep build-time SOC_VERSION tied to the
# physical device so an A2 custom-op package is not accidentally run on A3 (or
# vice versa).

detect_turboquant_soc_version() {
    local python_bin="${1:-python3}"
    local device_name
    local normalized

    if ! device_name="$("${python_bin}" - <<'PY'
import torch
import torch_npu  # noqa: F401

print(torch.npu.get_device_name(0))
PY
    )"; then
        printf 'Unable to query the physical NPU with %s. Source CANN first or set SOC_VERSION explicitly.\n' \
            "${python_bin}" >&2
        return 1
    fi

    normalized="${device_name,,}"
    normalized="${normalized//[[:space:]]/}"
    case "${normalized}" in
        ascend910b4*)
            printf 'ascend910b4\n'
            ;;
        ascend910_93*)
            printf '%s\n' "${normalized}"
            ;;
        *)
            printf 'Unsupported TurboQuant NPU model: %s. Expected Ascend 910B4/A2 or 910_93/A3.\n' \
                "${device_name}" >&2
            return 1
            ;;
    esac
}

turboquant_soc_family() {
    local soc_version="${1,,}"
    case "${soc_version}" in
        ascend910b4*)
            printf 'a2\n'
            ;;
        ascend910_93*)
            printf 'a3\n'
            ;;
        *)
            printf 'unknown\n'
            ;;
    esac
}

resolve_turboquant_soc_version() {
    local configured_soc="${1:-}"
    local python_bin="${2:-python3}"

    if [[ -n "${configured_soc}" ]]; then
        printf '%s\n' "${configured_soc,,}"
        return
    fi
    detect_turboquant_soc_version "${python_bin}"
}

validate_turboquant_soc_matches_device() {
    local selected_soc="$1"
    local python_bin="${2:-python3}"
    local detected_soc="${3:-}"
    local selected_family
    local detected_family

    if [[ -z "${detected_soc}" ]] && ! detected_soc="$(detect_turboquant_soc_version "${python_bin}")"; then
        printf 'WARNING: physical SoC could not be checked; using SOC_VERSION=%s.\n' \
            "${selected_soc}" >&2
        return
    fi
    selected_family="$(turboquant_soc_family "${selected_soc}")"
    detected_family="$(turboquant_soc_family "${detected_soc}")"
    if [[ "${selected_family}" == "unknown" || "${selected_family}" != "${detected_family}" ]]; then
        printf 'SOC_VERSION=%s targets %s, but the physical NPU is %s (%s). Refusing an incompatible A2/A3 build or run.\n' \
            "${selected_soc}" "${selected_family}" "${detected_soc}" "${detected_family}" >&2
        return 1
    fi
    printf 'SoC target check: selected=%s physical=%s family=%s\n' \
        "${selected_soc}" "${detected_soc}" "${selected_family}"
}
