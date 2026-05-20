#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/projects/aclab/liu.shu/envs/qwen25-vllm092-torch27-cu126}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VLLM_VERSION="${VLLM_VERSION:-0.9.2}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
XFORMERS_VERSION="${XFORMERS_VERSION:-0.0.30}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy<2.3}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.51.3}"
HF_HUB_SPEC="${HF_HUB_SPEC:-huggingface-hub<1.0}"
TOKENIZERS_SPEC="${TOKENIZERS_SPEC:-tokenizers<0.22,>=0.21.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if [[ -e "$ENV_PREFIX" && "${RESUME_EXISTING:-0}" != "1" ]]; then
    echo "Refusing to overwrite existing environment: $ENV_PREFIX" >&2
    echo "Set ENV_PREFIX to a new path, or remove the old env yourself after checking it is unused." >&2
    echo "If conda created the env but pip install did not finish, rerun with RESUME_EXISTING=1." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not on PATH. Load/initialize conda first." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Creating clean conda env: $ENV_PREFIX"
if [[ ! -e "$ENV_PREFIX" ]]; then
    conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" pip
else
    echo "Resuming existing env: $ENV_PREFIX"
fi

set +u
conda activate "$ENV_PREFIX"
set -u

python -m pip install --upgrade pip setuptools wheel
python -m pip install "$NUMPY_SPEC"
python -m pip install --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION"
python -m pip install \
    "xformers==$XFORMERS_VERSION" \
    "vllm==$VLLM_VERSION" \
    "$NUMPY_SPEC" \
    "$TRANSFORMERS_SPEC" \
    "$HF_HUB_SPEC" \
    "$TOKENIZERS_SPEC"

python - <<'PY'
import sys
import sysconfig
print("python", sys.executable)
print("prefix", sys.prefix)
print("base_prefix", sys.base_prefix)
print("stdlib", sysconfig.get_paths().get("stdlib"))
for name in ["numpy", "numba", "vllm", "torch", "transformers", "xformers", "triton"]:
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "?"), getattr(module, "__file__", "?"))
        if name == "torch":
            print("torch.version.cuda", module.version.cuda)
    except Exception as exc:
        print(name, "ERROR", repr(exc))
PY

cat <<EOF

Done.

Use it with:

  VLLM_PYTHON=$ENV_PREFIX/bin/python source ~/env_sh/sharing_a100_qwen25coder_interactive.sh

Or make it the default in the launcher after verifying it works.
EOF
