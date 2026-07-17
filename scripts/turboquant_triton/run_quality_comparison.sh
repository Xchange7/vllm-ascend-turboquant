#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PORT="${PORT:-18002}"
export TP_SIZE="${TP_SIZE:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export PROMPTS="${PROMPTS:-${SCRIPT_DIR}/quality_cases.jsonl}"
export REQUEST_MODE="chat"
export WARM_PREFIX="0"
export MAX_QUALITY_REGRESSIONS="${MAX_QUALITY_REGRESSIONS:-0}"

exec bash "${SCRIPT_DIR}/run_accuracy_comparison.sh"
