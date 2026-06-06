#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

STAY_LIMIT_ARGS=()
if [[ "${1:-}" == "--stay-limit" && -n "${2:-}" ]]; then
  STAY_LIMIT_ARGS=(--stay-limit "$2")
fi

PYTHON_BIN="${PYTHON_BIN:-python3 -u}"
GRID_CHUNK_SIZE="${GRID_CHUNK_SIZE:-1000}"
CONTEXT_STAY_BATCH_SIZE="${CONTEXT_STAY_BATCH_SIZE:-250}"
CONTEXT_NOTE_READ_CHUNKSIZE="${CONTEXT_NOTE_READ_CHUNKSIZE:-10000}"

echo "[recovery 1/4] Resuming 168h hourly state grid"
$PYTHON_BIN code/v3/build_hourly_state_grid.py \
  --hours 168 \
  --chunk-size-stays "${GRID_CHUNK_SIZE}" \
  --include-missingness-masks \
  --add-empty-backbone-cols \
  --resume \
  --meta-json results/v3/hourly_state_grid_168h_meta.json \
  "${STAY_LIMIT_ARGS[@]}"

echo "[recovery 2/4] Rebuilding v3 condition artefacts"
$PYTHON_BIN code/v3/build_delirium_labels.py "${STAY_LIMIT_ARGS[@]}"
$PYTHON_BIN code/v3/build_stroke_proxy_markers.py "${STAY_LIMIT_ARGS[@]}"
$PYTHON_BIN code/v3/build_arf_labels.py --resume "${STAY_LIMIT_ARGS[@]}"

echo "[recovery 3/4] Rebuilding state vectors"
$PYTHON_BIN code/v3/build_state_vectors.py "${STAY_LIMIT_ARGS[@]}"

echo "[recovery 4/4] Rebuilding multimodal contexts"
$PYTHON_BIN code/v3/build_time_aware_contexts.py \
  --resume \
  --stay-batch-size "${CONTEXT_STAY_BATCH_SIZE}" \
  --note-read-chunksize "${CONTEXT_NOTE_READ_CHUNKSIZE}" \
  "${STAY_LIMIT_ARGS[@]}"

echo "TIMELY-Bench v3 post-refresh recovery completed."
