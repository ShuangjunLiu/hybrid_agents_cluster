#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO_HELPER="/projects/aclab/liu.shu/codePool/hybrid_agents/scripts/start_vllm_tool_server.sh"
HELPER_SCRIPT="${HELPER_SCRIPT:-$SCRIPT_DIR/start_vllm_tool_server.sh}"
MODE="${1:-}"

if [[ "$MODE" != "--inside-allocation" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Already inside a Slurm allocation: job $SLURM_JOB_ID on partition ${SLURM_JOB_PARTITION:-unknown} at $HOSTNAME" >&2
        echo "Run: source $SCRIPT_PATH --inside-allocation" >&2
        exit 1
    fi

    echo "Requesting a 12-hour interactive allocation on multigpu with 2x V100 SXM2 nodes..."
    salloc -p multigpu \
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --gres=gpu:v100-sxm2:4 \
        --time=12:00:00 \
        --cpus-per-task=16 \
        --mem=64G \
        bash -lc "source '$SCRIPT_PATH' --inside-allocation"
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
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
PORT_A=8011
PORT_B=8012
JOBID="$SLURM_JOB_ID"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
WORKER_START_DELAY="${WORKER_START_DELAY:-300}"

if [[ ! -x "$HELPER_SCRIPT" && -x "$REPO_HELPER" ]]; then
    HELPER_SCRIPT="$REPO_HELPER"
fi
if [[ ! -x "$HELPER_SCRIPT" ]]; then
    echo "Missing executable vLLM helper: $HELPER_SCRIPT" >&2
    echo "Set HELPER_SCRIPT or install scripts/start_vllm_tool_server.sh next to this launcher." >&2
    exit 1
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

mkdir -p "$HF_HOME" "$HF_HOME/hub" "$HF_HOME/transformers" "$HF_HOME/datasets" \
         "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TMPDIR"

CTRL_OUT="/projects/aclab/liu.shu/model-cache/qwen25_2node_${JOBID}_${CONTROL_NODE}_${PORT_A}.out"
CTRL_ERR="/projects/aclab/liu.shu/model-cache/qwen25_2node_${JOBID}_${CONTROL_NODE}_${PORT_A}.err"
WORK_OUT="/projects/aclab/liu.shu/model-cache/qwen25_2node_${JOBID}_${WORKER_NODE}_${PORT_B}.out"
WORK_ERR="/projects/aclab/liu.shu/model-cache/qwen25_2node_${JOBID}_${WORKER_NODE}_${PORT_B}.err"

echo "Allocated job ID: $JOBID"
echo "$JOBID" > "$HOME/.current_multigpu_2node_jobid"
echo "Saved job ID to $HOME/.current_multigpu_2node_jobid"
echo "Allocation launcher host: $(hostname -s)"
echo "Control node: $CONTROL_NODE"
echo "Worker node:  $WORKER_NODE"
echo "Max model length: $MAX_MODEL_LEN"
echo "Worker start delay: ${WORKER_START_DELAY}s"
echo "vLLM helper: $HELPER_SCRIPT"
echo ""

echo "Starting Qwen2.5-Coder-32B server on worker node $WORKER_NODE at port $PORT_B..."
export HA_HELPER_SCRIPT="$HELPER_SCRIPT"
export HA_MODEL="$MODEL"
export HA_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export HA_WORKER_NODE="$WORKER_NODE"
export HA_WORKER_PORT="$PORT_B"
export HA_WORKER_START_DELAY="$WORKER_START_DELAY"
export HA_WORK_OUT="$WORK_OUT"
export HA_WORK_ERR="$WORK_ERR"
export HA_MODEL_CACHE_ROOT="$MODEL_CACHE_ROOT"
export HA_HF_HOME="$HF_HOME"
export HA_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HA_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export HA_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export HA_TORCH_HOME="$TORCH_HOME"
export HA_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export HA_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export HA_TMPDIR="$TMPDIR"

srun --nodes=1 --ntasks=1 --nodelist="$WORKER_NODE" \
     --gres=gpu:v100-sxm2:4 --cpus-per-task=16 --mem=64G --cpu-bind=none bash -lc '
    set -euo pipefail

    MODEL="$HA_MODEL"
    MAX_MODEL_LEN="$HA_MAX_MODEL_LEN"
    WORKER_NODE="$HA_WORKER_NODE"
    PORT_B="$HA_WORKER_PORT"
    WORKER_START_DELAY="$HA_WORKER_START_DELAY"
    WORK_OUT="$HA_WORK_OUT"
    WORK_ERR="$HA_WORK_ERR"
    MODEL_CACHE_ROOT="$HA_MODEL_CACHE_ROOT"
    HF_HOME="$HA_HF_HOME"
    HUGGINGFACE_HUB_CACHE="$HA_HUGGINGFACE_HUB_CACHE"
    TRANSFORMERS_CACHE="$HA_TRANSFORMERS_CACHE"
    HF_DATASETS_CACHE="$HA_HF_DATASETS_CACHE"
    TORCH_HOME="$HA_TORCH_HOME"
    VLLM_CACHE_ROOT="$HA_VLLM_CACHE_ROOT"
    XDG_CACHE_HOME="$HA_XDG_CACHE_HOME"
    TMPDIR="$HA_TMPDIR"
    VLLM_CPU_BIND=none

    export MODEL
    export MAX_MODEL_LEN
    export MODEL_CACHE_ROOT
    export HF_HOME
    export HUGGINGFACE_HUB_CACHE
    export TRANSFORMERS_CACHE
    export HF_DATASETS_CACHE
    export TORCH_HOME
    export VLLM_CACHE_ROOT
    export XDG_CACHE_HOME
    export TMPDIR
    export VLLM_CPU_BIND

    echo "Worker server will start after ${WORKER_START_DELAY}s to avoid concurrent shared-cache locks."
    sleep "$WORKER_START_DELAY"
    nohup "$HA_HELPER_SCRIPT" "$PORT_B" "$WORK_OUT" "$WORK_ERR" </dev/null &
    server_pid=$!
    echo "Worker server detached on ${WORKER_NODE} port ${PORT_B}"
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

export HA_MODEL="$MODEL"
export HA_PORT_A="$PORT_A"
export HA_PORT_B="$PORT_B"
export HA_JOBID="$JOBID"
export HA_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export HA_CONTROL_NODE="$CONTROL_NODE"
export HA_WORKER_NODE="$WORKER_NODE"
export HA_CTRL_OUT="$CTRL_OUT"
export HA_CTRL_ERR="$CTRL_ERR"
export HA_WORK_OUT="$WORK_OUT"
export HA_WORK_ERR="$WORK_ERR"
export HA_MODEL_CACHE_ROOT="$MODEL_CACHE_ROOT"
export HA_HF_HOME="$HF_HOME"
export HA_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HA_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export HA_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export HA_TORCH_HOME="$TORCH_HOME"
export HA_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export HA_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export HA_TMPDIR="$TMPDIR"
export HA_HELPER_SCRIPT="$HELPER_SCRIPT"

echo "Starting interactive control step on $CONTROL_NODE..."
srun --nodes=1 \
     --ntasks=1 \
     --nodelist="$CONTROL_NODE" \
     --gres=gpu:v100-sxm2:4 \
     --cpus-per-task=16 \
     --mem=64G \
     --cpu-bind=none \
     --pty bash -lc '
    set -euo pipefail

    MODEL="$HA_MODEL"
    PORT_A="$HA_PORT_A"
    PORT_B="$HA_PORT_B"
    JOBID="$HA_JOBID"
    MAX_MODEL_LEN="$HA_MAX_MODEL_LEN"
    CONTROL_NODE="$HA_CONTROL_NODE"
    WORKER_NODE="$HA_WORKER_NODE"
    CTRL_OUT="$HA_CTRL_OUT"
    CTRL_ERR="$HA_CTRL_ERR"
    WORK_OUT="$HA_WORK_OUT"
    WORK_ERR="$HA_WORK_ERR"
    MODEL_CACHE_ROOT="$HA_MODEL_CACHE_ROOT"
    HF_HOME="$HA_HF_HOME"
    HUGGINGFACE_HUB_CACHE="$HA_HUGGINGFACE_HUB_CACHE"
    TRANSFORMERS_CACHE="$HA_TRANSFORMERS_CACHE"
    HF_DATASETS_CACHE="$HA_HF_DATASETS_CACHE"
    TORCH_HOME="$HA_TORCH_HOME"
    VLLM_CACHE_ROOT="$HA_VLLM_CACHE_ROOT"
    XDG_CACHE_HOME="$HA_XDG_CACHE_HOME"
    TMPDIR="$HA_TMPDIR"
    VLLM_CPU_BIND=none

    export MODEL
    export MAX_MODEL_LEN
    export MODEL_CACHE_ROOT
    export HF_HOME
    export HUGGINGFACE_HUB_CACHE
    export TRANSFORMERS_CACHE
    export HF_DATASETS_CACHE
    export TORCH_HOME
    export VLLM_CACHE_ROOT
    export XDG_CACHE_HOME
    export TMPDIR
    export VLLM_CPU_BIND

    THIS_NODE="$(hostname -s)"
    if [[ "$THIS_NODE" != "$CONTROL_NODE" ]]; then
        echo "Expected control node $CONTROL_NODE, but running on $THIS_NODE" >&2
        exit 1
    fi

    export VLLM_CONTROL_ENDPOINT="http://127.0.0.1:$PORT_A/v1"
    export VLLM_WORKER_ENDPOINT="http://$WORKER_NODE:$PORT_B/v1"
    export OPENAI_BASE_URL="$VLLM_CONTROL_ENDPOINT"
    export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,$CONTROL_NODE,$WORKER_NODE,${CONTROL_NODE}.explorer.neu.edu,${WORKER_NODE}.explorer.neu.edu"
    export NO_PROXY="$no_proxy"

    echo "Starting Qwen2.5-Coder-32B server on control node $CONTROL_NODE at port $PORT_A..."
    nohup "$HA_HELPER_SCRIPT" "$PORT_A" "$CTRL_OUT" "$CTRL_ERR" </dev/null &
    CONTROL_SERVER_PID=$!

    wait_for_url() {
        local label="$1"
        local url="$2"
        local start="$SECONDS"
        local deadline=$((SECONDS + 1800))
        local next_status="$SECONDS"
        echo "Waiting for $label: $url"
        while true; do
            if curl --noproxy "*" -fsS --max-time 5 "$url" >/dev/null 2>&1; then
                echo "$label ready: $url"
                return 0
            fi
            if kill -0 "$CONTROL_SERVER_PID" 2>/dev/null; then
                :
            else
                echo "Control server exited while waiting for $label." >&2
                tail -n 80 "$CTRL_ERR" >&2 || true
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
    wait_for_url "control server" "http://127.0.0.1:$PORT_A/v1/models"
    wait_for_url "worker server" "http://$WORKER_NODE:$PORT_B/v1/models"

    echo ""
    echo "Servers are up."
    echo "Control endpoint: http://$CONTROL_NODE:$PORT_A/v1"
    echo "Worker endpoint:  http://$WORKER_NODE:$PORT_B/v1"
    echo "Client env:"
    echo "  VLLM_CONTROL_ENDPOINT=$VLLM_CONTROL_ENDPOINT"
    echo "  VLLM_WORKER_ENDPOINT=$VLLM_WORKER_ENDPOINT"
    echo "  OPENAI_BASE_URL=$OPENAI_BASE_URL"
    echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
    echo "  NO_PROXY=$NO_PROXY"
    echo "Logs:"
    echo "  $CTRL_OUT"
    echo "  $CTRL_ERR"
    echo "  $WORK_OUT"
    echo "  $WORK_ERR"
    echo ""
    echo "Run ~/env_sh/start_code_server.sh now if you want a browser editor on $CONTROL_NODE."
    echo "Then run Codex from that shell and point it at both vLLM endpoints."
    echo ""
    exec bash
'
