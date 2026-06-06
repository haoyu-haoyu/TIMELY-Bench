#!/bin/bash
#SBATCH --job-name=phase65c_gpt54mini_pilot
#SBATCH --partition=interruptible_cpu
#SBATCH --output=logs/v3/phase65c_gpt54mini_pilot_%j.out
#SBATCH --error=logs/v3/phase65c_gpt54mini_pilot_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

ROOT="/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final"
cd "$ROOT"

mkdir -p logs/v3 results/cres_v3/phase65c_tier1a_pilot

if [ -f /scratch/users/k25113331/venvs/timer/bin/activate ]; then
  source /scratch/users/k25113331/venvs/timer/bin/activate
fi

export CLIPROXY_KEY="$(cat /users/k25113331/.phase65c_gpt54mini.key)"
export OPENAI_API_KEY="$CLIPROXY_KEY"
export OPENAI_BASE_URL="http://127.0.0.1:18317/v1"

/users/k25113331/create_job_cliproxy_proxy.sh \
  python3 -u code/v3/run_phase65c_tier1a_pilot_v3.py \
    --mode run_provider \
    --root "$ROOT" \
    --provider gpt54mini \
    --model-name gpt-5.4-mini
