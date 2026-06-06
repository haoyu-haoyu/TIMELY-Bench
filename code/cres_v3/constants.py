from __future__ import annotations

from pathlib import Path

try:
    from config import ROOT_DIR  # type: ignore
except Exception:  # pragma: no cover
    ROOT_DIR = Path(__file__).resolve().parents[2]

from v3.constants import V3_CONTEXTS_DIR, V3_EVENTS_DIR, V3_PROCESSED_DIR, V3_RESULTS_DIR  # type: ignore


CRES_V3_RESULTS_DIR = ROOT_DIR / "results" / "cres_v3"
V3_KNOWLEDGE_DIR = V3_RESULTS_DIR / "knowledge"
V3_STATE_SPACE_DIR = V3_RESULTS_DIR / "state_space"
V3_PHASES_DIR = V3_PROCESSED_DIR / "phases"

PROJECT_ROOT = ROOT_DIR.parents[1]

DEFAULT_REFERENCE_CONDITION_GRAPHS = (
    PROJECT_ROOT / "项目进度" / "进度计划" / "condition_graphs.json"
)
DEFAULT_REFERENCE_PHYSIOLOGY_TEMPLATES = (
    PROJECT_ROOT / "项目进度" / "进度计划" / "physiology_templates.json"
)

DEFAULT_GRAPH_NODE_REGISTRY = V3_KNOWLEDGE_DIR / "graph_node_registry_v3.csv"
DEFAULT_KNOWLEDGE_GAP_REPORT = V3_KNOWLEDGE_DIR / "knowledge_gap_report_v3.json"
DEFAULT_KNOWLEDGE_SUMMARY = V3_KNOWLEDGE_DIR / "knowledge_build_summary_v3.json"

DEFAULT_PHASE_SUMMARY_JSON = V3_PHASES_DIR / "phase_label_summary_v3.json"
DEFAULT_STATE_SPACE_SUMMARY_JSON = V3_STATE_SPACE_DIR / "state_space_build_summary_v3.json"

DEFAULT_CRES_V3_TASK_DIR = CRES_V3_RESULTS_DIR


def ensure_cres_v3_directories() -> None:
    for path in (
        CRES_V3_RESULTS_DIR,
        V3_KNOWLEDGE_DIR,
        V3_STATE_SPACE_DIR,
        V3_PHASES_DIR,
        V3_CONTEXTS_DIR,
        V3_EVENTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
