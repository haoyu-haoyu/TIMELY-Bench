#!/bin/bash
set -euo pipefail

OUTDIR="results/note_centered/progression_tasks"
RUNNER="scripts/run_single_progression_experiment.sh"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/progression"
SUBMIT_LOG="${OUTDIR}/submitted_jobs_${TS}.txt"

mkdir -p "${OUTDIR}" "${LOGDIR}"

# Archive previous progression task outputs to avoid stale read.
shopt -s nullglob
old_files=("${OUTDIR}"/aki_progression_*.json "${OUTDIR}"/sepsis_shock_*.json)
if (( ${#old_files[@]} > 0 )); then
  archive_dir="${OUTDIR}/archive_${TS}"
  mkdir -p "${archive_dir}"
  mv "${old_files[@]}" "${archive_dir}/"
  echo "Archived ${#old_files[@]} previous JSON files to ${archive_dir}"
fi
shopt -u nullglob

touch "${SUBMIT_LOG}"
count=0

submit_one() {
  local name="$1"
  shift
  local jid
  jid=$(sbatch --parsable --job-name="${name}" "${RUNNER}" "$@")
  echo "${jid} ${name} $*" | tee -a "${SUBMIT_LOG}"
  count=$((count + 1))
}

for task in aki_progression sepsis_shock; do
  # Group 1: Structured-only (2)
  submit_one "${task}_s_W24"   "${task}" structured xgb W24    none              "${OUTDIR}"
  submit_one "${task}_s_leak"  "${task}" structured xgb leaked  none              "${OUTDIR}"

  # Group 2: Text-only (3)
  submit_one "${task}_t_W24"   "${task}" text_only  lr  W24    original          "${OUTDIR}"
  submit_one "${task}_t_leak"  "${task}" text_only  lr  leaked original          "${OUTDIR}"
  submit_one "${task}_t_clean" "${task}" text_only  lr  clean  weighted_no_after "${OUTDIR}"

  # Group 3: 2x2 decomposition (4)
  submit_one "${task}_A"       "${task}" fusion xgb leaked  original          "${OUTDIR}"
  submit_one "${task}_B"       "${task}" fusion xgb leaked  weighted_no_after "${OUTDIR}"
  submit_one "${task}_C"       "${task}" fusion xgb W24     original          "${OUTDIR}"
  submit_one "${task}_D"       "${task}" fusion xgb clean   weighted_no_after "${OUTDIR}"

  # Group 4: Window comparison (2)
  submit_one "${task}_W6"      "${task}" fusion xgb W6   original "${OUTDIR}"
  submit_one "${task}_W12"     "${task}" fusion xgb W12  original "${OUTDIR}"
done

echo "Submitted ${count} experiments."
echo "Submission log: ${SUBMIT_LOG}"
