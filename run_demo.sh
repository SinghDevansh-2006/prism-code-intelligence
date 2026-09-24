#!/bin/bash
set -e

HOST="127.0.0.1"
PORT="8000"
BASE="http://${HOST}:${PORT}"

echo "========================================"
echo " PRISM Agentic Code Intelligence"
echo "========================================"

if curl -s "${BASE}/health" >/dev/null 2>&1; then
    echo "✓ Backend already running"
else
    echo "Starting backend..."

    uvicorn api:app \
      --host "${HOST}" \
      --port "${PORT}" \
      > /tmp/prism_api.log 2>&1 &

    echo $! > /tmp/prism_api.pid

    for i in $(seq 1 30); do
        if curl -s "${BASE}/health" >/dev/null 2>&1; then
            break
        fi

        sleep 1
    done

    if ! curl -s "${BASE}/health" >/dev/null 2>&1; then
        echo "ERROR: backend failed to start"
        cat /tmp/prism_api.log
        exit 1
    fi

    echo "✓ Backend ready"
fi


SEMANTIC_LOADED=$(
python - <<PY
import json
import urllib.request

try:
    with urllib.request.urlopen(
        "${BASE}/health",
        timeout=5,
    ) as response:
        data = json.load(response)

    print(
        "yes"
        if data["models_loaded"]["semantic"]
        else "no"
    )
except Exception:
    print("no")
PY
)


if [ "${SEMANTIC_LOADED}" != "yes" ]; then
    echo
    echo "Prewarming semantic models..."
    echo "First launch can take ~30 seconds."

    python - <<PY
import json
import urllib.request

body = json.dumps({
    "query":
        "Find code that validates a tic-tac-toe board state",
    "top_k": 1,
}).encode()

request = urllib.request.Request(
    "${BASE}/search",
    data=body,
    headers={
        "Content-Type": "application/json"
    },
    method="POST",
)

with urllib.request.urlopen(
    request,
    timeout=120,
) as response:
    data = json.load(response)

print(
    f"✓ Models warm — "
    f"{data['total_ms']:.0f} ms initial search"
)
PY

else
    echo "✓ Semantic models already warm"
fi


echo
echo "Opening demo:"
echo "${BASE}/"

open "${BASE}/"

echo
echo "✓ Demo ready"
