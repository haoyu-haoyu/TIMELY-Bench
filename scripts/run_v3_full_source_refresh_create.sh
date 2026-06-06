#!/usr/bin/env bash
set -euo pipefail

# Full TIMELY-Bench v3 source refresh on CREATE:
# rebuilds cohort, structured backbone, and raw notes from BigQuery,
# then runs the standard v3 build pipeline on top of refreshed sources.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

STAY_LIMIT_ARGS=()
if [[ "${1:-}" == "--stay-limit" && -n "${2:-}" ]]; then
  STAY_LIMIT_ARGS=(--stay-limit "$2")
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_v3_full_source_refresh_create.sh [--stay-limit N]

Environment variables:
  PYTHON_BIN         Python runner (default: "python3 -u")
  SOURCE_BATCH_SIZE  Batch size for structured backbone extraction (default: 2000)
  NOTE_BATCH_SIZE    Batch size for note extraction (default: 1500)
EOF
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3 -u}"
SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE:-5000}"
NOTE_BATCH_SIZE="${NOTE_BATCH_SIZE:-1500}"
BQ_BILLING_PROJECT="${BQ_BILLING_PROJECT:-timely-bench-mimic}"
BQ_ARGS=(--billing-project "${BQ_BILLING_PROJECT}")

run_python() {
  # shellcheck disable=SC2086
  $PYTHON_BIN "$@"
}

echo "[1/4] Extracting refreshed v3 cohort from BigQuery"
run_python code/v3/extract_cohort_bq.py "${BQ_ARGS[@]}" "${STAY_LIMIT_ARGS[@]}"

echo "[2/4] Extracting refreshed 168h structured backbone from BigQuery"
run_python code/v3/extract_structured_backbone_bq.py \
  "${BQ_ARGS[@]}" \
  --batch-size "${SOURCE_BATCH_SIZE}" \
  "${STAY_LIMIT_ARGS[@]}"

echo "[3/4] Extracting refreshed 168h note sources from BigQuery"
run_python code/v3/extract_notes_bq.py \
  "${BQ_ARGS[@]}" \
  --batch-size "${NOTE_BATCH_SIZE}" \
  "${STAY_LIMIT_ARGS[@]}"

echo "[4/4] Running standard v3 build pipeline on refreshed sources"
BQ_BILLING_PROJECT="${BQ_BILLING_PROJECT}" bash scripts/run_v3_create_pipeline.sh "${STAY_LIMIT_ARGS[@]}"

echo "TIMELY-Bench v3 full source refresh completed."
