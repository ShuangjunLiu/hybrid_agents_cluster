#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <port> <stdout-log> <stderr-log>" >&2
  exit 2
fi

PORT="$1"
OUT_LOG="$2"
ERR_LOG="$3"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/projects/aclab/liu.shu/model-cache}"

export MODEL_CACHE_ROOT
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
  "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TMPDIR" \
  "$(dirname "$OUT_LOG")" "$(dirname "$ERR_LOG")"

exec /projects/aclab/liu.shu/envs/qwen3-vllm-cu118/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len "$MAX_MODEL_LEN" \
  --no-enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --served-model-name "$MODEL" \
  >"$OUT_LOG" 2>"$ERR_LOG"
