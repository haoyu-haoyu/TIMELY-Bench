#!/bin/bash
set -euo pipefail

ROOT=/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
VENV="${ROOT}/.venv_gemma4_vllm"

if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m venv "${VENV}"
fi

source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
if python - <<'PY'
import importlib.metadata
versions = {
    "vllm": importlib.metadata.version("vllm"),
    "transformers": importlib.metadata.version("transformers"),
}
assert versions["vllm"] == "0.19.0", versions
assert versions["transformers"] == "5.5.3", versions
print(versions)
PY
then
  echo "vLLM environment already ready"
else
  python -m pip install --upgrade --pre vllm
  python -m pip install --upgrade transformers==5.5.3
fi

python - <<'PY'
import torch
import vllm
import transformers
print({"torch": torch.__version__, "vllm": vllm.__version__})
print({"transformers": transformers.__version__})
PY
