#!/usr/bin/env bash

# Shared absolute paths for scripts that may be launched from any directory.
TQ_SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${TQ_SCRIPT_ROOT}/../.." && pwd)"
TQ_COMMON_DIR="${TQ_SCRIPT_ROOT}/common"
TQ_CORRECTNESS_DIR="${TQ_SCRIPT_ROOT}/correctness"
TQ_PERFORMANCE_DIR="${TQ_SCRIPT_ROOT}/performance"
TQ_DIAGNOSTICS_DIR="${TQ_SCRIPT_ROOT}/diagnostics"

readonly TQ_SCRIPT_ROOT REPO_ROOT TQ_COMMON_DIR TQ_CORRECTNESS_DIR
readonly TQ_PERFORMANCE_DIR TQ_DIAGNOSTICS_DIR
