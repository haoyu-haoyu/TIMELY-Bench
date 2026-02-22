#!/bin/bash
#SBATCH -J repair_closure
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

source /scratch/users/k25113331/venvs/timer/bin/activate
cd /scratch/users/k25113331/TIMELY-Bench_Final
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

python code/condition_graphs/build_condition_graphs.py
python code/condition_graphs/validate_condition_graph.py --all-graphs --check-mapping
python code/data_processing/generate_predefined_splits.py
python code/state_space/reconstruct_state_space.py
python code/data_processing/build_final_release_bundle.py
python code/data_processing/run_final_qa.py --timeseries-rows 5000000
