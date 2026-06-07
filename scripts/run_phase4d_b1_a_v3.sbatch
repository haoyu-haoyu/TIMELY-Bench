#!/bin/bash
#SBATCH --job-name=phase4d_b1_a_v3
#SBATCH --output=logs/v3/phase4d_b1_a_%j.out
#SBATCH --error=logs/v3/phase4d_b1_a_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

ROOT="${PROJECT_ROOT:-$PWD}"
cd "$ROOT"

mkdir -p logs/v3 results/v3/representations

python3 -B code/v3/build_phase4_b1_a_v3.py \
  --conditions aki \
  --combined-summary-json results/v3/representations/phase4d_B1_A_build_summary_aki.json

python3 -B code/v3/build_phase4_b1_a_v3.py \
  --conditions delirium \
  --combined-summary-json results/v3/representations/phase4d_B1_A_build_summary_delirium.json

python3 -B code/v3/build_phase4_b1_a_v3.py \
  --conditions sepsis \
  --combined-summary-json results/v3/representations/phase4d_B1_A_build_summary_sepsis.json

python3 -B code/v3/build_phase4_b1_a_v3.py \
  --conditions stroke \
  --combined-summary-json results/v3/representations/phase4d_B1_A_build_summary_stroke.json

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
results = root / "results" / "v3" / "representations"

combined = {
    "phase": "4D",
    "representation_families": ["B1", "A"],
    "build_mode": "slurm_sequential_per_condition",
    "conditions": {},
}

for condition in ["aki", "delirium", "sepsis", "stroke"]:
    cond_dir = root / "results" / "v3" / condition
    with open(cond_dir / f"{condition}_B1_build_summary.json", "r", encoding="utf-8") as f:
        b1 = json.load(f)
    with open(cond_dir / f"{condition}_A_build_summary.json", "r", encoding="utf-8") as f:
        a = json.load(f)
    combined["conditions"][condition] = {"B1": b1, "A": a}

with open(results / "phase4d_B1_A_build_summary.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2)

print(results / "phase4d_B1_A_build_summary.json")
PY
