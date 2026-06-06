#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 HOLD_JOB_ID" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
ROOT=/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
OUT_DIR="${ROOT}/results/cres_v3/phase65d_tier1b_full"
LOG_DIR="${ROOT}/logs/v3"
PORT="${PORT:-8017}"
MAX_WORKERS="${MAX_WORKERS:-16}"
PILOT_SIZE="${PILOT_SIZE:-500}"
DIRECT_IN_JOB="${PHASE65D_VLLM_DIRECT_IN_JOB:-0}"

PILOT_MANIFEST="${OUT_DIR}/phase65d_gemma4_vllm_pilot500_remaining.jsonl"
PILOT_BUILD_SUMMARY="${OUT_DIR}/phase65d_gemma4_vllm_pilot500_build_summary.json"
PILOT_SUMMARY="${OUT_DIR}/phase65d_gemma4_vllm_pilot500_summary.json"
FULL_MANIFEST="${OUT_DIR}/phase65d_gemma4_vllm_remaining_full.jsonl"
FULL_BUILD_SUMMARY="${OUT_DIR}/phase65d_gemma4_vllm_remaining_full_build_summary.json"
CUTOVER_STATUS="${OUT_DIR}/phase65d_gemma4_vllm_cutover_status.json"
SERVER_LOG="${LOG_DIR}/phase65d_gemma4_vllm_server_$(date +%Y%m%d_%H%M%S).out"
CLIENT_PILOT_LOG="${LOG_DIR}/phase65d_gemma4_vllm_client_pilot_$(date +%Y%m%d_%H%M%S).out"
CLIENT_FULL_LOG="${LOG_DIR}/phase65d_gemma4_vllm_client_full_$(date +%Y%m%d_%H%M%S).out"
FALLBACK_LOG="${LOG_DIR}/phase65d_gemma4_local_fallback_$(date +%Y%m%d_%H%M%S).out"

write_status() {
  local phase="$1"
  local state="$2"
  local note="$3"
  python3 - <<PY
import json
from pathlib import Path
path = Path("${CUTOVER_STATUS}")
payload = {
    "phase": ${phase@Q},
    "state": ${state@Q},
    "note": ${note@Q},
}
path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, ensure_ascii=False))
PY
}

stop_plain_transformers() {
  pkill -f 'run_phase65d_gemma4_local_h200_v1.py' || true
  pkill -f 'run_phase65d_gemma4_local_h200_v1.sh' || true
}

stop_vllm() {
  pkill -f 'run_phase65d_gemma4_vllm_server_v1.sh' || true
  pkill -f 'vllm serve google/gemma-4-26B-A4B-it' || true
}

restart_plain_transformers() {
  if [ "${DIRECT_IN_JOB}" = "1" ]; then
    return 0
  fi
  nohup bash "${ROOT}/scripts/run_phase65d_gemma4_local_h200_v1.sh" "${HOLD_JOB_ID}" >"${FALLBACK_LOG}" 2>&1 < /dev/null &
}

wait_for_server() {
  local attempts=120
  local sleep_seconds=5
  for _ in $(seq 1 "${attempts}"); do
    if [ "${DIRECT_IN_JOB}" = "1" ]; then
      if bash -lc "curl -fsS http://127.0.0.1:${PORT}/v1/models | python3 -c 'import json,sys; data=json.load(sys.stdin); names=[item.get(\"id\") for item in data.get(\"data\", [])]; assert \"arc:lite\" in names, names; print({\"models\": names})'"; then
        return 0
      fi
    elif srun --overlap --jobid="${HOLD_JOB_ID}" --ntasks=1 --cpus-per-task=1 --mem=1G --export=ALL \
      bash -lc "curl -fsS http://127.0.0.1:${PORT}/v1/models | python3 -c 'import json,sys; data=json.load(sys.stdin); names=[item.get(\"id\") for item in data.get(\"data\", [])]; assert \"arc:lite\" in names, names; print({\"models\": names})'"; then
      return 0
    fi
    if ! pgrep -f 'run_phase65d_gemma4_vllm_server_v1.sh' >/dev/null && ! pgrep -f 'vllm serve google/gemma-4-26B-A4B-it' >/dev/null; then
      return 1
    fi
    sleep "${sleep_seconds}"
  done
  return 1
}

evaluate_pilot() {
  local elapsed_seconds="$1"
  python3 - <<PY
import json
summary = json.load(open("${PILOT_SUMMARY}", "r", encoding="utf-8"))
elapsed = max(1, int(${elapsed_seconds}))
throughput = summary["parse_success_rows"] / (elapsed / 3600.0)
summary["pilot_elapsed_seconds"] = elapsed
summary["parse_success_per_hour"] = throughput
summary["pilot_pass"] = (
    summary["rows_requested"] == ${PILOT_SIZE}
    and summary["rows_found"] == ${PILOT_SIZE}
    and (summary["ok_rows"] / summary["rows_requested"]) >= 0.99
    and (summary["parse_success_rows"] / summary["rows_requested"]) >= 0.98
    and ((summary["status_counts"].get("http_error", 0) + summary["status_counts"].get("exception", 0)) / summary["rows_requested"]) <= 0.01
    and summary["has_reasoning_true"] == 0
    and throughput >= 300.0
)
json.dump(summary, open("${PILOT_SUMMARY}", "w", encoding="utf-8"), indent=2)
print(json.dumps(summary, ensure_ascii=False))
PY
}

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
write_status "setup" "running" "initializing Gemma4 vLLM cutover"

if ! squeue -h -j "${HOLD_JOB_ID}" | grep -q .; then
  write_status "setup" "failed" "hold job not active"
  exit 1
fi

bash "${ROOT}/scripts/setup_phase65d_gemma4_vllm_env_v1.sh"
stop_plain_transformers
stop_vllm

python3 "${ROOT}/code/v3/run_phase65d_tier1b_v3.py" \
  --mode build_remaining_manifest \
  --root "${ROOT}" \
  --provider gemma4_26b \
  --model-name arc:lite \
  --manifest-path "${OUT_DIR}/phase65d_full_manifest_full_multimodal.jsonl" \
  --manifest-output-path "${PILOT_MANIFEST}" \
  --summary-path "${PILOT_BUILD_SUMMARY}" \
  --limit "${PILOT_SIZE}"

write_status "pilot_server" "running" "starting vLLM server"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before launch}"
export HF_HOME="${HF_HOME:-/scratch/prj/bhi_haoyu_benchmarking/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
nohup bash "${ROOT}/scripts/run_phase65d_gemma4_vllm_server_v1.sh" "${HOLD_JOB_ID}" >"${SERVER_LOG}" 2>&1 < /dev/null &

if ! wait_for_server; then
  write_status "pilot_server" "failed" "vLLM server failed to become ready"
  stop_vllm
  restart_plain_transformers
  exit 1
fi

write_status "pilot_client" "running" "running 500 prompt pilot"
pilot_started="$(date +%s)"
bash "${ROOT}/scripts/run_phase65d_gemma4_vllm_client_v1.sh" "${HOLD_JOB_ID}" "${PILOT_MANIFEST}" "${MAX_WORKERS}" >"${CLIENT_PILOT_LOG}" 2>&1
pilot_elapsed="$(( $(date +%s) - pilot_started ))"

python3 "${ROOT}/code/v3/run_phase65d_tier1b_v3.py" \
  --mode summarize_manifest_subset \
  --root "${ROOT}" \
  --output-dir "${OUT_DIR}" \
  --provider gemma4_26b \
  --model-name arc:lite \
  --manifest-path "${PILOT_MANIFEST}" \
  --summary-path "${PILOT_SUMMARY}"

pilot_eval_json="$(evaluate_pilot "${pilot_elapsed}")"
echo "${pilot_eval_json}"

pilot_pass="$(python3 - <<PY
import json
summary = json.load(open("${PILOT_SUMMARY}", "r", encoding="utf-8"))
print("1" if summary.get("pilot_pass") else "0")
PY
)"

if [ "${pilot_pass}" != "1" ]; then
  write_status "pilot_eval" "failed" "pilot did not meet pass criteria"
  stop_vllm
  restart_plain_transformers
  exit 1
fi

write_status "full_build" "running" "building full remaining manifest after pilot"
python3 "${ROOT}/code/v3/run_phase65d_tier1b_v3.py" \
  --mode build_remaining_manifest \
  --root "${ROOT}" \
  --provider gemma4_26b \
  --model-name arc:lite \
  --manifest-path "${OUT_DIR}/phase65d_full_manifest_full_multimodal.jsonl" \
  --manifest-output-path "${FULL_MANIFEST}" \
  --summary-path "${FULL_BUILD_SUMMARY}"

write_status "full_client" "running" "starting full remaining run on vLLM"
bash "${ROOT}/scripts/run_phase65d_gemma4_vllm_client_v1.sh" "${HOLD_JOB_ID}" "${FULL_MANIFEST}" "${MAX_WORKERS}" >"${CLIENT_FULL_LOG}" 2>&1

python3 "${ROOT}/code/v3/run_phase65d_tier1b_v3.py" \
  --mode summarize \
  --root "${ROOT}" \
  --output-dir "${OUT_DIR}" \
  --providers deepseek_chat qwen35 gemma4_26b

stop_vllm
write_status "done" "completed" "Gemma4 vLLM full run finished and summaries refreshed"
