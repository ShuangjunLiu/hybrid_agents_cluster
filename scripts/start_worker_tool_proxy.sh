#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
LISTEN_HOST=${LISTEN_HOST:-127.0.0.1}
LISTEN_PORT=${LISTEN_PORT:-18012}
UPSTREAM_PORT=${UPSTREAM_PORT:-8012}
TIMEOUT=${TIMEOUT:-60}
PROXY_TIMEOUT=${PROXY_TIMEOUT:-600}
STATE_DIR=${STATE_DIR:-/tmp/${USER:-unknown}/hybrid-agent-proxy}
LOG_FILE=${LOG_FILE:-$STATE_DIR/worker-tool-proxy-${LISTEN_PORT}.log}
PID_FILE=${PID_FILE:-$STATE_DIR/worker-tool-proxy-${LISTEN_PORT}.pid}

if [[ -z "${WORKER_NODE:-}" ]]; then
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    WORKER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '2p')
  fi
fi

if [[ -z "${WORKER_NODE:-}" ]]; then
  echo "WORKER_NODE is required, or run inside a two-node Slurm allocation." >&2
  exit 2
fi

UPSTREAM=${UPSTREAM:-http://${WORKER_NODE}:${UPSTREAM_PORT}}
ENDPOINT=http://${LISTEN_HOST}:${LISTEN_PORT}/v1

mkdir -p "$STATE_DIR"

if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  if curl --noproxy "*" -fsS --max-time 5 "${ENDPOINT%/}/models" >/dev/null 2>&1; then
    echo "ENDPOINT=$ENDPOINT"
    echo "PID_FILE=$PID_FILE"
    echo "LOG_FILE=$LOG_FILE"
    exit 0
  fi
  kill "$(cat "$PID_FILE")" 2>/dev/null || true
fi

cd "$REPO_ROOT"
setsid nohup scripts/openai_tool_choice_proxy.py \
  --listen-host "$LISTEN_HOST" \
  --listen-port "$LISTEN_PORT" \
  --upstream "$UPSTREAM" \
  --timeout "$PROXY_TIMEOUT" \
  </dev/null >"$LOG_FILE" 2>&1 &
PROXY_PID=$!
printf '%s\n' "$PROXY_PID" >"$PID_FILE"

deadline=$((SECONDS + TIMEOUT))
until curl --noproxy "*" -fsS --max-time 5 "${ENDPOINT%/}/models" >/dev/null 2>&1; do
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "Proxy exited before readiness. Log: $LOG_FILE" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${ENDPOINT%/}/models. Log: $LOG_FILE" >&2
    exit 1
  fi
  sleep 1
done

echo "ENDPOINT=$ENDPOINT"
echo "PID_FILE=$PID_FILE"
echo "LOG_FILE=$LOG_FILE"
