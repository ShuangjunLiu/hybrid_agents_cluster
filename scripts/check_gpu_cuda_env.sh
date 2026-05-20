#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-${VLLM_PYTHON:-python}}"

echo "Host: $(hostname -s)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo ""

echo "== nvidia-smi =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi
echo ""

echo "== CUDA toolkit =="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not found"
fi
echo ""

echo "== Python/vLLM env =="
if [[ -x "$PYTHON_BIN" ]] || command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  "$PYTHON_BIN" - <<'PY'
import os
import sys
import sysconfig

print("python", sys.executable)
print("prefix", sys.prefix)
print("base_prefix", sys.base_prefix)
print("stdlib", sysconfig.get_paths().get("stdlib"))
print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
for name in ["numpy", "numba", "torch", "vllm", "transformers"]:
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "?"), getattr(module, "__file__", "?"))
        if name == "torch":
            print("torch.cuda", getattr(module.version, "cuda", None))
            print("torch.cuda.is_available", module.cuda.is_available())
    except Exception as exc:
        print(name, "ERROR", repr(exc))
PY
else
  echo "Python not found or not executable: $PYTHON_BIN"
fi
echo ""

echo "== CUDA runtime libraries visible in env =="
if [[ -x "$PYTHON_BIN" ]]; then
  ENV_PREFIX="$("$PYTHON_BIN" - <<'PY'
import sys
print(sys.prefix)
PY
)"
  find "$ENV_PREFIX" -name 'libcudart.so*' -o -name 'libcuda.so*' 2>/dev/null | sort | sed -n '1,80p'
fi
