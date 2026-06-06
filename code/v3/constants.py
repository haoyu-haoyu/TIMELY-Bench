from __future__ import annotations

from pathlib import Path

try:  # Reuse v2 single-source paths when available.
    from config import (  # type: ignore
        COHORT_FILE as V2_COHORT_FILE,
        DATA_DIR,
        MERGE_OUTPUT_DIR,
        PROCESSED_DIR,
        RAW_DATA_DIR,
        RESULTS_DIR,
        ROOT_DIR,
        TIMESERIES_FILE as V2_TIMESERIES_FILE,
    )
except Exception:  # pragma: no cover - used when executed standalone
    _HERE = Path(__file__).resolve()
    ROOT_DIR = _HERE.parents[2]
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    MERGE_OUTPUT_DIR = PROCESSED_DIR / "merge_output"
    RESULTS_DIR = ROOT_DIR / "results"
    V2_TIMESERIES_FILE = RAW_DATA_DIR / "timeseries_sorted_extended.csv"
    V2_COHORT_FILE = MERGE_OUTPUT_DIR / "cohort_final_extended.csv"


V3_MAX_HOURS = 168
V3_NOTES_MAX_HOURS = 168

V3_PROCESSED_DIR = PROCESSED_DIR / "v3"
V3_RAW_DATA_DIR = RAW_DATA_DIR / "v3"
V3_EVENTS_DIR = V3_PROCESSED_DIR / "events"
V3_HOURLY_FEATURES_DIR = V3_PROCESSED_DIR / "hourly_features"
V3_CONTEXTS_DIR = V3_PROCESSED_DIR / "contexts"
V3_STATE_VECTORS_DIR = V3_PROCESSED_DIR / "state_vectors"
V3_RESULTS_DIR = RESULTS_DIR / "v3"

V3_SOURCE_COHORT_FILE = V3_PROCESSED_DIR / "cohort_v3.csv"
V3_SOURCE_TIMESERIES_FILE = V3_PROCESSED_DIR / "structured_backbone_hourly_v3.parquet"


def _fallback_csv_path(path: Path) -> Path:
    return path.with_suffix(".csv")


def _chunk_dir_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.parts")


def _table_like_exists(path: Path) -> bool:
    return path.exists() or _fallback_csv_path(path).exists() or _chunk_dir_path(path).exists()


def _prefer_table(primary: Path, fallback: Path) -> Path:
    return primary if _table_like_exists(primary) else fallback


DEFAULT_V3_TIMESERIES_FILE = _prefer_table(V3_SOURCE_TIMESERIES_FILE, V2_TIMESERIES_FILE)
DEFAULT_V3_COHORT_FILE = _prefer_table(V3_SOURCE_COHORT_FILE, V2_COHORT_FILE)
DEFAULT_BQ_FEATURE_DIR = V3_PROCESSED_DIR / "_legacy_bq_features_disabled"
DEFAULT_TEXT_FEATURES_FILE = PROCESSED_DIR / "temporal_alignment" / "patient_text_features.csv"
DEFAULT_NOTE_METADATA_FILE = PROCESSED_DIR / "text_embeddings" / "note_level_metadata.csv"
DEFAULT_DISEASE_TIMELINES_FILE = PROCESSED_DIR / "disease_timelines" / "disease_timelines_full.json"

V3_RAW_NOTE_FILES = {
    "discharge": V3_RAW_DATA_DIR / "discharge_notes_v3.parquet",
    "nursing": V3_RAW_DATA_DIR / "nursing_notes_168h.parquet",
    "lab_comment": V3_RAW_DATA_DIR / "lab_comments_168h.parquet",
    "radiology": V3_RAW_DATA_DIR / "radiology_notes_168h.parquet",
}
LEGACY_RAW_NOTE_FILES = {
    "discharge": RAW_DATA_DIR / "discharge_notes.csv",
    "nursing": RAW_DATA_DIR / "nursing_notes.csv",
    "lab_comment": RAW_DATA_DIR / "lab_comments.csv",
    "radiology": RAW_DATA_DIR / "note_time.csv",
}
RAW_NOTE_FILES = {
    key: _prefer_table(V3_RAW_NOTE_FILES[key], LEGACY_RAW_NOTE_FILES[key])
    for key in V3_RAW_NOTE_FILES
}

DEFAULT_FEATURE_DICTIONARY_JSON = V3_RESULTS_DIR / "feature_dictionary_v3.json"
DEFAULT_FEATURE_DICTIONARY_CSV = V3_RESULTS_DIR / "feature_dictionary_v3.csv"
DEFAULT_FEATURE_DICTIONARY_MD = V3_RESULTS_DIR / "feature_dictionary_v3.md"

DEFAULT_DIAGNOSIS_PATHWAY_EVENTS = V3_EVENTS_DIR / "diagnosis_pathway_events_v3.parquet"
DEFAULT_DIAGNOSIS_COMORBIDITIES = V3_EVENTS_DIR / "diagnosis_comorbidities_v3.parquet"
DEFAULT_MEDICATION_EVENTS = V3_EVENTS_DIR / "medication_events_bq.parquet"
DEFAULT_PROCEDURE_EVENTS = V3_EVENTS_DIR / "procedure_events_bq.parquet"
DEFAULT_HOURLY_STATE_GRID = V3_PROCESSED_DIR / "hourly_state_grid_168h.parquet"
DEFAULT_CONTEXT_JSONL = V3_CONTEXTS_DIR / "time_aware_patient_contexts_168h.jsonl"
DEFAULT_CONTEXT_CLEAN_JSONL = V3_CONTEXTS_DIR / "time_aware_patient_contexts_168h_clean.jsonl"
DEFAULT_CONTEXT_SUMMARY_JSON = V3_CONTEXTS_DIR / "time_aware_patient_contexts_168h_summary.json"
DEFAULT_STATE_VECTORS = V3_STATE_VECTORS_DIR / "state_vectors_168h.parquet"


def ensure_v3_directories() -> None:
    for path in (
        V3_PROCESSED_DIR,
        V3_RAW_DATA_DIR,
        V3_EVENTS_DIR,
        V3_HOURLY_FEATURES_DIR,
        V3_CONTEXTS_DIR,
        V3_STATE_VECTORS_DIR,
        V3_RESULTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
