#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
HYBRID_AGENTS_REPO_ROOT="${HYBRID_AGENTS_REPO_ROOT:-/projects/aclab/liu.shu/codePool/hybrid_agents}"
REPO_HELPER="$HYBRID_AGENTS_REPO_ROOT/scripts/start_vllm_tool_server.sh"
REPO_TIME_GUARD="$HYBRID_AGENTS_REPO_ROOT/scripts/slurm_time_guard.sh"
HELPER_SCRIPT="${HELPER_SCRIPT:-$SCRIPT_DIR/start_vllm_tool_server.sh}"
TIME_GUARD_SCRIPT="${TIME_GUARD_SCRIPT:-$SCRIPT_DIR/slurm_time_guard.sh}"
MODE="${1:-}"
DEFAULT_PARTITION="${DEFAULT_PARTITION:-multigpu}"
DEFAULT_SLURM_TIME="${DEFAULT_SLURM_TIME:-12:00:00}"
SLURM_EXCLUDE="${SLURM_EXCLUDE:-${EXCLUDE_NODES:-}}"

if [[ "$MODE" != "--inside-allocation" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Already inside a Slurm allocation: job $SLURM_JOB_ID on partition ${SLURM_JOB_PARTITION:-unknown} at $HOSTNAME" >&2
        echo "Run: source $SCRIPT_PATH --inside-allocation" >&2
        exit 1
    fi

    SALLOC_ARGS=(
        -p "${PARTITION:-$DEFAULT_PARTITION}"
        --nodes=2
        --ntasks=2
        --ntasks-per-node=1
        --gres=gpu:v100-sxm2:2
        --time="${SLURM_TIME:-$DEFAULT_SLURM_TIME}"
        --cpus-per-task="${CPUS_PER_TASK:-8}"
        --mem="${SLURM_MEM:-32G}"
    )
    if [[ -n "$SLURM_EXCLUDE" ]]; then
        SALLOC_ARGS+=(--exclude="$SLURM_EXCLUDE")
    fi

    echo "Requesting a ${SLURM_TIME:-$DEFAULT_SLURM_TIME} interactive allocation on ${PARTITION:-$DEFAULT_PARTITION} with 2x V100 SXM2 nodes, 2 GPUs per node..."
    if [[ -n "$SLURM_EXCLUDE" ]]; then
        echo "Excluding nodes: $SLURM_EXCLUDE"
    fi
    salloc "${SALLOC_ARGS[@]}" bash -lc "source '$SCRIPT_PATH' --inside-allocation"
    return 0 2>/dev/null || exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Internal error: --inside-allocation was used without an active Slurm allocation." >&2
    exit 1
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if (( ${#NODES[@]} < 2 )); then
    echo "Expected a 2-node allocation, but Slurm assigned only ${#NODES[@]} node(s): ${NODES[*]:-none}" >&2
    echo "Job node list: ${SLURM_JOB_NODELIST:-unknown}" >&2
    exit 1
fi

CONTROL_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
PORT_A="${CONTROL_PORT:-8011}"
PORT_B="${WORKER_PORT:-8012}"
JOBID="$SLURM_JOB_ID"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"
DTYPE="${DTYPE:-auto}"
VLLM_PYTHON="${VLLM_PYTHON:-/projects/aclab/liu.shu/envs/qwen25-vllm092-torch27-cu126/bin/python}"
WORKER_START_DELAY="${WORKER_START_DELAY:-30}"

if [[ ! -x "$HELPER_SCRIPT" && -x "$REPO_HELPER" ]]; then
    HELPER_SCRIPT="$REPO_HELPER"
fi
if [[ ! -x "$TIME_GUARD_SCRIPT" && -x "$REPO_TIME_GUARD" ]]; then
    TIME_GUARD_SCRIPT="$REPO_TIME_GUARD"
fi
if [[ ! -x "$HELPER_SCRIPT" ]]; then
    echo "Missing executable vLLM helper: $HELPER_SCRIPT" >&2
    echo "Set HELPER_SCRIPT or install scripts/start_vllm_tool_server.sh next to this launcher." >&2
    exit 1
fi
if [[ ! -x "$TIME_GUARD_SCRIPT" ]]; then
    echo "Warning: missing executable Slurm time guard: $TIME_GUARD_SCRIPT" >&2
    echo "Set TIME_GUARD_SCRIPT or install scripts/slurm_time_guard.sh to enable walltime reminders." >&2
    TIME_GUARD_SCRIPT=""
fi

export MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/projects/aclab/liu.shu/model-cache}"
export HF_HOME="${HF_HOME:-$MODEL_CACHE_ROOT/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$MODEL_CACHE_ROOT/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$MODEL_CACHE_ROOT/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MODEL_CACHE_ROOT/cache}"
export TMPDIR="${TMPDIR:-$MODEL_CACHE_ROOT/tmp}"
export VLLM_CPU_BIND=none

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TMPDIR"

LOG_ROOT="${LOG_ROOT:-/projects/aclab/liu.shu/model-cache}"
CTRL_OUT="$LOG_ROOT/qwen25_14b_tp2_2node_${JOBID}_${CONTROL_NODE}_${PORT_A}.out"
CTRL_ERR="$LOG_ROOT/qwen25_14b_tp2_2node_${JOBID}_${CONTROL_NODE}_${PORT_A}.err"
WORK_OUT="$LOG_ROOT/qwen25_14b_tp2_2node_${JOBID}_${WORKER_NODE}_${PORT_B}.out"
WORK_ERR="$LOG_ROOT/qwen25_14b_tp2_2node_${JOBID}_${WORKER_NODE}_${PORT_B}.err"
STATE_FILE="$HOME/.current_gpu_2node_qwen25coder14b_job"

cat >"$STATE_FILE" <<EOF
JOBID=$JOBID
CONTROL_NODE=$CONTROL_NODE
WORKER_NODE=$WORKER_NODE
MODEL=$MODEL
CONTROL_ENDPOINT=http://127.0.0.1:$PORT_A/v1
WORKER_ENDPOINT=http://$WORKER_NODE:$PORT_B/v1
MAX_MODEL_LEN=$MAX_MODEL_LEN
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
DTYPE=$DTYPE
VLLM_PYTHON=$VLLM_PYTHON
HYBRID_AGENTS_REPO_ROOT=$HYBRID_AGENTS_REPO_ROOT
CTRL_OUT=$CTRL_OUT
CTRL_ERR=$CTRL_ERR
WORK_OUT=$WORK_OUT
WORK_ERR=$WORK_ERR
EOF

echo "Allocated job ID: $JOBID"
echo "Saved state to: $STATE_FILE"
echo "Allocation launcher host: $(hostname -s)"
echo "Control node: $CONTROL_NODE"
echo "Worker node:  $WORKER_NODE"
echo "Model: $MODEL"
echo "Max model length: $MAX_MODEL_LEN"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "dtype: $DTYPE"
echo "vLLM Python: $VLLM_PYTHON"
echo "Hybrid agents repo: $HYBRID_AGENTS_REPO_ROOT"
echo "Worker start delay: ${WORKER_START_DELAY}s"
echo "vLLM helper: $HELPER_SCRIPT"
echo "Slurm time guard: ${TIME_GUARD_SCRIPT:-disabled}"
echo ""

export HA_HELPER_SCRIPT="$HELPER_SCRIPT"
export HA_TIME_GUARD_SCRIPT="$TIME_GUARD_SCRIPT"
export HA_REPO_ROOT="$HYBRID_AGENTS_REPO_ROOT"
export HA_MODEL="$MODEL"
export HA_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export HA_TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE"
export HA_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
export HA_DTYPE="$DTYPE"
export HA_VLLM_PYTHON="$VLLM_PYTHON"
export HA_MODEL_CACHE_ROOT="$MODEL_CACHE_ROOT"
export HA_HF_HOME="$HF_HOME"
export HA_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HA_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export HA_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export HA_TORCH_HOME="$TORCH_HOME"
export HA_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export HA_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export HA_TMPDIR="$TMPDIR"

echo "Starting Qwen2.5-Coder-14B TP=2 server on worker node $WORKER_NODE at port $PORT_B..."
export HA_WORKER_NODE="$WORKER_NODE"
export HA_WORKER_PORT="$PORT_B"
export HA_WORKER_START_DELAY="$WORKER_START_DELAY"
export HA_WORK_OUT="$WORK_OUT"
export HA_WORK_ERR="$WORK_ERR"

srun --nodes=1 --ntasks=1 --nodelist="$WORKER_NODE" \
     --gres=gpu:v100-sxm2:2 --cpus-per-task="${CPUS_PER_TASK:-8}" --mem="${SLURM_MEM:-32G}" --cpu-bind=none bash -lc '
    set -euo pipefail

    export MODEL="$HA_MODEL"
    export HYBRID_AGENTS_REPO_ROOT="$HA_REPO_ROOT"
    export MAX_MODEL_LEN="$HA_MAX_MODEL_LEN"
    export TENSOR_PARALLEL_SIZE="$HA_TENSOR_PARALLEL_SIZE"
    export GPU_MEMORY_UTILIZATION="$HA_GPU_MEMORY_UTILIZATION"
    export DTYPE="$HA_DTYPE"
    export VLLM_PYTHON="$HA_VLLM_PYTHON"
    export MODEL_CACHE_ROOT="$HA_MODEL_CACHE_ROOT"
    export HF_HOME="$HA_HF_HOME"
    export HUGGINGFACE_HUB_CACHE="$HA_HUGGINGFACE_HUB_CACHE"
    export TRANSFORMERS_CACHE="$HA_TRANSFORMERS_CACHE"
    export HF_DATASETS_CACHE="$HA_HF_DATASETS_CACHE"
    export TORCH_HOME="$HA_TORCH_HOME"
    export VLLM_CACHE_ROOT="$HA_VLLM_CACHE_ROOT"
    export XDG_CACHE_HOME="$HA_XDG_CACHE_HOME"
    export TMPDIR="$HA_TMPDIR"
    export VLLM_CPU_BIND=none
    export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/tmp/$USER/outlines-cache-${SLURM_JOB_ID:-manual}-${HA_WORKER_NODE}-${HA_WORKER_PORT}}"

    echo "Worker server will start after ${HA_WORKER_START_DELAY}s to avoid concurrent shared-cache locks."
    sleep "$HA_WORKER_START_DELAY"
    nohup "$HA_HELPER_SCRIPT" "$HA_WORKER_PORT" "$HA_WORK_OUT" "$HA_WORK_ERR" </dev/null &
    server_pid=$!
    echo "Worker server detached on ${HA_WORKER_NODE} port ${HA_WORKER_PORT}"
    wait "$server_pid"
' &
WORKER_STEP_PID=$!

cleanup_worker_step() {
    if kill -0 "$WORKER_STEP_PID" 2>/dev/null; then
        kill "$WORKER_STEP_PID" 2>/dev/null || true
        wait "$WORKER_STEP_PID" 2>/dev/null || true
    fi
}
trap cleanup_worker_step EXIT

export HA_PORT_A="$PORT_A"
export HA_PORT_B="$PORT_B"
export HA_CONTROL_NODE="$CONTROL_NODE"
export HA_WORKER_NODE="$WORKER_NODE"
export HA_CTRL_OUT="$CTRL_OUT"
export HA_CTRL_ERR="$CTRL_ERR"
export HA_WORK_OUT="$WORK_OUT"
export HA_WORK_ERR="$WORK_ERR"

echo "Starting interactive control step on $CONTROL_NODE..."
srun --nodes=1 \
     --ntasks=1 \
     --nodelist="$CONTROL_NODE" \
     --gres=gpu:v100-sxm2:2 \
     --cpus-per-task="${CPUS_PER_TASK:-8}" \
     --mem="${SLURM_MEM:-32G}" \
     --cpu-bind=none \
     --pty bash -lc '
    set -euo pipefail

    export MODEL="$HA_MODEL"
    export HYBRID_AGENTS_REPO_ROOT="$HA_REPO_ROOT"
    export MAX_MODEL_LEN="$HA_MAX_MODEL_LEN"
    export TENSOR_PARALLEL_SIZE="$HA_TENSOR_PARALLEL_SIZE"
    export GPU_MEMORY_UTILIZATION="$HA_GPU_MEMORY_UTILIZATION"
    export DTYPE="$HA_DTYPE"
    export VLLM_PYTHON="$HA_VLLM_PYTHON"
    export MODEL_CACHE_ROOT="$HA_MODEL_CACHE_ROOT"
    export HF_HOME="$HA_HF_HOME"
    export HUGGINGFACE_HUB_CACHE="$HA_HUGGINGFACE_HUB_CACHE"
    export TRANSFORMERS_CACHE="$HA_TRANSFORMERS_CACHE"
    export HF_DATASETS_CACHE="$HA_HF_DATASETS_CACHE"
    export TORCH_HOME="$HA_TORCH_HOME"
    export VLLM_CACHE_ROOT="$HA_VLLM_CACHE_ROOT"
    export XDG_CACHE_HOME="$HA_XDG_CACHE_HOME"
    export TMPDIR="$HA_TMPDIR"
    export VLLM_CPU_BIND=none
    export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/tmp/$USER/outlines-cache-${SLURM_JOB_ID:-manual}-${HA_CONTROL_NODE}-${HA_PORT_A}}"

    THIS_NODE="$(hostname -s)"
    if [[ "$THIS_NODE" != "$HA_CONTROL_NODE" ]]; then
        echo "Expected control node $HA_CONTROL_NODE, but running on $THIS_NODE" >&2
        exit 1
    fi

    if [[ -d "$HA_REPO_ROOT" ]]; then
        cd "$HA_REPO_ROOT"
    else
        echo "Warning: hybrid agents repo not found: $HA_REPO_ROOT" >&2
    fi

    if [[ -n "${HA_TIME_GUARD_SCRIPT:-}" && -x "$HA_TIME_GUARD_SCRIPT" ]]; then
        export SLURM_TIME_GUARD_LOG="${SLURM_TIME_GUARD_LOG:-/tmp/$USER/slurm-time-guard-${SLURM_JOB_ID:-manual}.log}"
        "$HA_TIME_GUARD_SCRIPT" "$SLURM_JOB_ID" &
        export SLURM_TIME_GUARD_PID=$!
        echo "Slurm time guard started: pid=$SLURM_TIME_GUARD_PID log=$SLURM_TIME_GUARD_LOG thresholds=${SLURM_TIME_GUARD_WARN_MINUTES:-60,30,15,5,1}m"
    fi

    export VLLM_CONTROL_ENDPOINT="http://127.0.0.1:$HA_PORT_A/v1"
    export VLLM_WORKER_ENDPOINT="http://$HA_WORKER_NODE:$HA_PORT_B/v1"
    export OPENAI_BASE_URL="$VLLM_CONTROL_ENDPOINT"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
    export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,$HA_CONTROL_NODE,$HA_WORKER_NODE,${HA_CONTROL_NODE}.explorer.neu.edu,${HA_WORKER_NODE}.explorer.neu.edu"
    export NO_PROXY="$no_proxy"

    echo "Starting Qwen2.5-Coder-14B TP=2 server on control node $HA_CONTROL_NODE at port $HA_PORT_A..."
    nohup "$HA_HELPER_SCRIPT" "$HA_PORT_A" "$HA_CTRL_OUT" "$HA_CTRL_ERR" </dev/null &
    CONTROL_SERVER_PID=$!

    wait_for_url() {
        local label="$1"
        local url="$2"
        local start="$SECONDS"
        local deadline=$((SECONDS + ${STARTUP_TIMEOUT:-900}))
        local next_status="$SECONDS"
        echo "Waiting for $label: $url"
        while true; do
            if curl --noproxy "*" -fsS --max-time 5 "$url" >/dev/null 2>&1; then
                echo "$label ready: $url"
                return 0
            fi
            if ! kill -0 "$CONTROL_SERVER_PID" 2>/dev/null; then
                echo "Control server exited while waiting for $label. Last stderr lines:" >&2
                tail -n 80 "$HA_CTRL_ERR" >&2 || true
                return 1
            fi
            if (( SECONDS >= deadline )); then
                echo "Timed out waiting for $label: $url" >&2
                return 1
            fi
            if (( SECONDS >= next_status )); then
                echo "Still waiting for $label... $((SECONDS - start))s elapsed"
                next_status=$((SECONDS + 30))
            fi
            sleep 5
        done
    }

    echo ""
    echo "Waiting for both endpoints to become ready..."
    wait_for_url "control server" "http://127.0.0.1:$HA_PORT_A/v1/models"
    wait_for_url "worker server" "http://$HA_WORKER_NODE:$HA_PORT_B/v1/models"

    echo ""
    echo "Servers are up."
    echo "Control endpoint: http://$HA_CONTROL_NODE:$HA_PORT_A/v1"
    echo "Worker endpoint:  http://$HA_WORKER_NODE:$HA_PORT_B/v1"
    echo "Client env:"
    echo "  VLLM_CONTROL_ENDPOINT=$VLLM_CONTROL_ENDPOINT"
    echo "  VLLM_WORKER_ENDPOINT=$VLLM_WORKER_ENDPOINT"
    echo "  OPENAI_BASE_URL=$OPENAI_BASE_URL"
    echo "  MODEL=$MODEL"
    echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
    echo "  TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
    echo "  VLLM_PYTHON=$VLLM_PYTHON"
    echo "  HYBRID_AGENTS_REPO_ROOT=$HA_REPO_ROOT"
    echo "  NO_PROXY=$NO_PROXY"
    echo "  SLURM_TIME_GUARD_PID=${SLURM_TIME_GUARD_PID:-disabled}"
    echo "  SLURM_TIME_GUARD_LOG=${SLURM_TIME_GUARD_LOG:-disabled}"
    echo "Logs:"
    echo "  $HA_CTRL_OUT"
    echo "  $HA_CTRL_ERR"
    echo "  $HA_WORK_OUT"
    echo "  $HA_WORK_ERR"
    echo ""
    echo "Smoke checks:"
    echo "  $HA_REPO_ROOT/scripts/validate_vllm_endpoints.py --timeout 60 --model Qwen/Qwen2.5-Coder-14B-Instruct --expected-max-model-len 16384 --json"
    echo "  $HA_REPO_ROOT/scripts/check_hybrid_worker.py"
    echo "  $HA_REPO_ROOT/scripts/run_mbpp_single_file_eval.py --run-id mbpp-3-14b-tp2 --limit 3 --model Qwen/Qwen2.5-Coder-14B-Instruct --expected-model Qwen/Qwen2.5-Coder-14B-Instruct"
    echo "  $HA_REPO_ROOT/scripts/run_mbpp_single_file_eval.py --run-id mbpp-25-14b-tp2 --limit 25 --model Qwen/Qwen2.5-Coder-14B-Instruct --expected-model Qwen/Qwen2.5-Coder-14B-Instruct"
    echo "  $HA_REPO_ROOT/scripts/aggregate_mbpp_single_file_eval.py /projects/aclab/liu.shu/model-cache/tmp/qwen_mbpp_single_file_eval/mbpp-25-14b-tp2"
    echo ""
    exec bash
'
