#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-turboquant-smoke}"
MAX_TOKENS="${MAX_TOKENS:-16}"
PROMPT="${PROMPT:-Say hi in one short sentence.}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
export SERVED_MODEL_NAME
export MAX_TOKENS
export PROMPT

base_url="http://${HOST}:${PORT}"
models_json="/tmp/turboquant_models_${PORT}.json"
response_json="/tmp/turboquant_chat_${PORT}.json"

echo "Waiting for ${base_url}/v1/models ..."
deadline=$((SECONDS + TIMEOUT_SECONDS))
while true; do
  if curl -sf "${base_url}/v1/models" > "${models_json}"; then
    echo "Readiness OK:"
    cat "${models_json}"
    echo
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for /v1/models after ${TIMEOUT_SECONDS}s"
    exit 1
  fi
  sleep 3
done

payload=$(
  python3 -c 'import json, os
print(json.dumps({
    "model": os.environ["SERVED_MODEL_NAME"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "temperature": 0,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
}))'
)

echo "Sending chat completion request ..."
curl -sf "${base_url}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "${payload}" > "${response_json}"

cat "${response_json}"
echo

python3 - "${response_json}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

choices = data.get("choices") or []
if not choices:
    raise SystemExit("Smoke request failed: response has no choices")

message = choices[0].get("message") or {}
content = message.get("content")
if not content:
    raise SystemExit("Smoke request failed: first choice has empty content")

print("Smoke request OK")
PY
