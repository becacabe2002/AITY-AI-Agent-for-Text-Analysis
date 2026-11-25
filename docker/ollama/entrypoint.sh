#!/usr/bin/env bash
# Runs `ollama serve` and the estimation shim side by side in one container,
# so the shim's cgroup counter (cpu.stat) is the same one ollama's work lands in.
# If either process exits, stop the other and exit, so Docker's restart policy
# restarts the whole container. Run under an init (compose `init: true`) so
# signals are forwarded and zombies are reaped.
set -uo pipefail

/bin/ollama serve &
OLLAMA_PID=$!

/opt/shim/venv/bin/uvicorn app:app \
  --app-dir /opt/shim/app \
  --host 0.0.0.0 --port 8788 \
  --log-level warning &
SHIM_PID=$!

shutdown() { kill -TERM "$OLLAMA_PID" "$SHIM_PID" 2>/dev/null || true; }
trap shutdown TERM INT

wait -n "$OLLAMA_PID" "$SHIM_PID"
STATUS=$?
shutdown
wait || true
exit "$STATUS"
