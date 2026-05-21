#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
MODE="${1:-}"

if [[ "$MODE" != "--inside-allocation" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Already inside a Slurm allocation: job $SLURM_JOB_ID on partition ${SLURM_JOB_PARTITION:-unknown} at $HOSTNAME" >&2
        echo "Run: source $SCRIPT_PATH --inside-allocation" >&2
        exit 1
    fi

    echo "Requesting a 60-minute interactive session on sharing with 1x A100..."
    srun -p sharing \
        --nodes=1 \
        --ntasks=1 \
        --gres=gpu:a100:1 \
        --time="${SLURM_TIME:-01:00:00}" \
        --cpus-per-task="${CPUS_PER_TASK:-8}" \
        --mem="${SLURM_MEM:-64G}" \
        --pty \
        bash -lc "source '$SCRIPT_PATH' --inside-allocation"
    return 0 2>/dev/null || exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Internal error: --inside-allocation was used without an active Slurm allocation." >&2
    exit 1
fi

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
VLLM_PYTHON="${VLLM_PYTHON:-/projects/aclab/liu.shu/envs/qwen25-vllm092-torch27-cu126/bin/python}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"
JOBID="$SLURM_JOB_ID"
NODE="$(hostname -s)"

export MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/projects/aclab/liu.shu/model-cache}"
export HF_HOME="${HF_HOME:-$MODEL_CACHE_ROOT/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$MODEL_CACHE_ROOT/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$MODEL_CACHE_ROOT/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MODEL_CACHE_ROOT/cache}"
export TMPDIR="${TMPDIR:-$MODEL_CACHE_ROOT/tmp}"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/tmp/$USER/outlines-cache-${SLURM_JOB_ID:-manual}-$NODE-$PORT}"
export VLLM_CPU_BIND=none
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

if [[ "${USE_EXPLORER_PROXY:-1}" == "1" ]]; then
    EXPLORER_PROXY="${EXPLORER_PROXY:-http://10.99.0.130:3128}"
    export http_proxy="${http_proxy:-$EXPLORER_PROXY}"
    export https_proxy="${https_proxy:-$EXPLORER_PROXY}"
    export HTTP_PROXY="${HTTP_PROXY:-$EXPLORER_PROXY}"
    export HTTPS_PROXY="${HTTPS_PROXY:-$EXPLORER_PROXY}"
fi

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TMPDIR" "$OUTLINES_CACHE_DIR"

LOG_ROOT="${LOG_ROOT:-/projects/aclab/liu.shu/model-cache}"
OUT_LOG="$LOG_ROOT/qwen25_a100_${JOBID}_${NODE}_${PORT}.out"
ERR_LOG="$LOG_ROOT/qwen25_a100_${JOBID}_${NODE}_${PORT}.err"
STATE_FILE="$HOME/.current_a100_qwen25coder_job"

cat >"$STATE_FILE" <<EOF
JOBID=$JOBID
NODE=$NODE
MODEL=$MODEL
PORT=$PORT
OPENAI_BASE_URL=$OPENAI_BASE_URL
MAX_MODEL_LEN=$MAX_MODEL_LEN
OUTLINES_CACHE_DIR=$OUTLINES_CACHE_DIR
OUT_LOG=$OUT_LOG
ERR_LOG=$ERR_LOG
EOF

echo "Allocated job ID: $JOBID"
echo "Saved state to: $STATE_FILE"
echo "Node: $NODE"
echo "Model: $MODEL"
echo "vLLM Python: $VLLM_PYTHON"
echo "VLLM_USE_V1: $VLLM_USE_V1"
echo "Endpoint: $OPENAI_BASE_URL"
echo "Max model length: $MAX_MODEL_LEN"
echo "Outlines cache: $OUTLINES_CACHE_DIR"
echo "Explorer proxy for model download: ${USE_EXPLORER_PROXY:-1}"
echo "Logs:"
echo "  $OUT_LOG"
echo "  $ERR_LOG"
echo ""
echo "GPUs available:"
nvidia-smi
echo ""

echo "Starting vLLM with OpenAI tool-call parsing enabled..."
VLLM_ENV_PREFIX="$(cd "$(dirname "$VLLM_PYTHON")/.." && pwd)"
VLLM_LD_LIBRARY_PATH="$VLLM_ENV_PREFIX/lib"
if [[ -d "$VLLM_ENV_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib" ]]; then
    VLLM_LD_LIBRARY_PATH="$VLLM_ENV_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib:$VLLM_LD_LIBRARY_PATH"
fi
if [[ -d "$VLLM_ENV_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib" ]]; then
    VLLM_LD_LIBRARY_PATH="$VLLM_ENV_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$VLLM_LD_LIBRARY_PATH"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    VLLM_LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
fi

env -u NO_PROXY -u no_proxy -u PYTHONHOME -u PYTHONPATH -u VIRTUAL_ENV \
  LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" \
  VLLM_USE_V1="$VLLM_USE_V1" \
  "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --no-enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --served-model-name "$MODEL" \
  >"$OUT_LOG" 2>"$ERR_LOG" </dev/null &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_url() {
    local url="$1"
    local start="$SECONDS"
    local deadline=$((SECONDS + ${STARTUP_TIMEOUT:-900}))
    local next_status="$SECONDS"
    echo "Waiting for $url"
    while true; do
        if curl --noproxy "*" -fsS --max-time 5 "$url" >/dev/null 2>&1; then
            echo "Ready: $url"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM exited before readiness. Last stderr lines:" >&2
            tail -n 80 "$ERR_LOG" >&2 || true
            return 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for $url. Last stderr lines:" >&2
            tail -n 80 "$ERR_LOG" >&2 || true
            return 1
        fi
        if (( SECONDS >= next_status )); then
            echo "Still waiting... $((SECONDS - start))s elapsed"
            next_status=$((SECONDS + 30))
        fi
        sleep 5
    done
}

wait_for_url "$OPENAI_BASE_URL/models"

LOCAL_NO_PROXY="localhost,127.0.0.1,::1,$NODE,$NODE.explorer.neu.edu"
export NO_PROXY="$LOCAL_NO_PROXY"
export no_proxy="$LOCAL_NO_PROXY"

echo ""
echo "Server is up. Fast checks:"
echo "  scripts/validate_vllm_endpoints.py --control-url $OPENAI_BASE_URL --timeout 30"
echo "  scripts/probe_openai_tool_calls.py --endpoint $OPENAI_BASE_URL --out /tmp/qwen25-tool-probe.json"
echo ""
echo "To try 14B instead next time:"
echo "  MODEL=Qwen/Qwen2.5-Coder-14B-Instruct MAX_MODEL_LEN=8192 source $SCRIPT_PATH"
echo ""
exec bash
