#!/usr/bin/env bash
set -euo pipefail

# TIMELY-Bench v3 CREATE-side foundation pipeline.
# Intended for large-data execution on CREATE/HPC with BigQuery credentials.
#
# Usage:
#   bash scripts/run_v3_create_pipeline.sh [--stay-limit 1000]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

STAY_LIMIT_ARGS=()
if [[ "${1:-}" == "--stay-limit" && -n "${2:-}" ]]; then
  STAY_LIMIT_ARGS=(--stay-limit "$2")
fi

PYTHON_BIN="${PYTHON_BIN:-python3 -u}"
BATCH_SIZE_ARGS=(--batch-size "${BATCH_SIZE:-5000}")
GRID_CHUNK_SIZE="${GRID_CHUNK_SIZE:-1000}"
CONTEXT_STAY_BATCH_SIZE="${CONTEXT_STAY_BATCH_SIZE:-250}"
CONTEXT_NOTE_READ_CHUNKSIZE="${CONTEXT_NOTE_READ_CHUNKSIZE:-10000}"
BQ_BILLING_PROJECT="${BQ_BILLING_PROJECT:-timely-bench-mimic}"
BQ_ARGS=(--billing-project "${BQ_BILLING_PROJECT}")

echo "[1/5] Building v3 feature dictionary"
$PYTHON_BIN code/v3/build_feature_dictionary.py

echo "[2/5] Extracting v3 event tables from BigQuery"
$PYTHON_BIN code/v3/extract_events_bq.py "${BQ_ARGS[@]}" "${BATCH_SIZE_ARGS[@]}" "${STAY_LIMIT_ARGS[@]}"

echo "[3/7] Extracting v3 hourly extension features from BigQuery"
$PYTHON_BIN code/v3/extract_hourly_features_bq.py "${BQ_ARGS[@]}" "${BATCH_SIZE_ARGS[@]}" "${STAY_LIMIT_ARGS[@]}"

echo "[4/7] Building diagnosis pathway events"
$PYTHON_BIN code/v3/build_diagnosis_pathway_events.py "${STAY_LIMIT_ARGS[@]}"

echo "[5/7] Building 168h hourly state grid"
$PYTHON_BIN code/v3/build_hourly_state_grid.py \
  --hours 168 \
  --chunk-size-stays "${GRID_CHUNK_SIZE}" \
  --include-missingness-masks \
  --add-empty-backbone-cols \
  --meta-json results/v3/hourly_state_grid_168h_meta.json \
  "${STAY_LIMIT_ARGS[@]}"

echo "[6/7] Building v3 condition artefacts"
$PYTHON_BIN code/v3/build_delirium_labels.py "${STAY_LIMIT_ARGS[@]}"
$PYTHON_BIN code/v3/build_stroke_proxy_markers.py "${STAY_LIMIT_ARGS[@]}"
$PYTHON_BIN code/v3/build_arf_labels.py --resume "${STAY_LIMIT_ARGS[@]}"

echo "[7/7] Building state vectors and multimodal contexts"
$PYTHON_BIN code/v3/build_state_vectors.py "${STAY_LIMIT_ARGS[@]}"
$PYTHON_BIN code/v3/build_time_aware_contexts.py \
  --resume \
  --stay-batch-size "${CONTEXT_STAY_BATCH_SIZE}" \
  --note-read-chunksize "${CONTEXT_NOTE_READ_CHUNKSIZE}" \
  "${STAY_LIMIT_ARGS[@]}"

echo "TIMELY-Bench v3 CREATE pipeline completed."
