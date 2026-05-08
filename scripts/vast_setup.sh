#!/usr/bin/env bash
set -euo pipefail

# One-shot environment setup on a fresh Vast.ai PyTorch (cuda12) box.
# Assumes pytorch/pytorch:2.x-cuda12.x-cudnn-devel image.

cd /root || cd "$HOME"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d sandbag-bench ]; then
  echo "[vast_setup] expected sandbag-bench/ in CWD; rsync or git clone first" >&2
  exit 1
fi

cd sandbag-bench

uv venv --python 3.10
source .venv/bin/activate

uv pip install -e .
uv pip install "sglang[all]>=0.5.6"
uv pip install "git+https://github.com/kitft/nla-inference.git"

mkdir -p results/exp0/activations hf_cache
export HF_HOME="$PWD/hf_cache"

if [ -n "${HF_TOKEN:-}" ]; then
  python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
fi

echo "[vast_setup] done. activate with: source .venv/bin/activate"
