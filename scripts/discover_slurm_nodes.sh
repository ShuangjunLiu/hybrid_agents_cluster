#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "SLURM_JOB_NODELIST is not set. Run this inside the active Slurm allocation." >&2
  exit 2
fi

if ! command -v scontrol >/dev/null 2>&1; then
  echo "scontrol is not available on PATH." >&2
  exit 2
fi

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

if [[ "${#nodes[@]}" -lt 2 ]]; then
  echo "Expected at least 2 Slurm nodes, got ${#nodes[@]} from SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}" >&2
  exit 1
fi

control_node="${nodes[0]}"
worker_node="${nodes[1]}"

cat <<EOF
CONTROL_NODE=${control_node}
WORKER_NODE=${worker_node}
CONTROL_ENDPOINT=http://127.0.0.1:8011/v1
WORKER_ENDPOINT=http://${worker_node}:8012/v1
EOF
