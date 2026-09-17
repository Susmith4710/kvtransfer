#!/usr/bin/env bash
# Creates an isolated virtual environment for kvtransfer on a DGX Spark.  Touches nothing outside it:
# no system pip, no system torch/CUDA/driver changes.  Re-runnable.
# Usage: bash scripts/dgx_spark/setup_venv.sh [venv_dir]   (default ~/.venvs/kvtransfer)
set -euo pipefail
VENV=${1:-$HOME/.venvs/kvtransfer}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /etc/dgx-release ] && python3 -c "import torch" 2>/dev/null; then
  echo "note: a system torch exists; it will NOT be modified. Installing a separate copy inside $VENV."
fi
python3 -m venv "$VENV"                      # plain venv: no --system-site-packages, fully isolated
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
# DGX OS ships CUDA 13 only -> cu130 wheels (>= torch 2.9).  Installed inside the venv only.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e "$REPO[data,eval,nvml]"
echo
echo "venv ready: source $VENV/bin/activate"
python - <<'PY'
import sys, torch
assert sys.prefix != sys.base_prefix, "not inside the venv?"
print("python", sys.executable)
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
