from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results" / "note_centered"
CORE_EXPERIMENTS_DIR = RESULTS_DIR / "core_experiments"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

WINDOW_ORDER = ["D0", "W6", "W12", "W24", "leaked", "clean"]

CANONICAL_BASENAME_BONUS = {
    "mortality_lr_clean_clinicalbert_typed.json": 200,
    "prolonged_los_lr_clean_clinicalbert_typed.json": 200,
    "mortality_late_stacking_lr_clean_original_typed.json": 200,
    "prolonged_los_late_stacking_lr_clean_original_typed.json": 200,
}


@dataclass
class ExperimentRecord:
    path: Path
    data: Dict
    mtime: float

    @property
    def key(self) -> Tuple:
        d = self.data
        # normalize no-ablation alias values
        note_ablation = d.get("note_ablation")
        if isinstance(note_ablation, str) and note_ablation.lower() in {"none", "null", ""}:
            note_ablation = None
        return (
            d.get("task"),
            d.get("modality"),
            d.get("fusion_strategy"),
            d.get("model"),
            d.get("window"),
            d.get("text_method"),
            note_ablation,
        )


def _record_priority(path: Path) -> int:
    name = path.name
    score = 0
    score += CANONICAL_BASENAME_BONUS.get(name, 0)
    if "__req_" not in name:
        score += 20
    if "clean_weighted_no_after.json" in name:
        score += 10
    if "clean_weighted_typed_no_after__req_original_typed.json" in name:
        score += 5
    return score


def load_raw_records(core_dir: Path = CORE_EXPERIMENTS_DIR) -> List[ExperimentRecord]:
    records: List[ExperimentRecord] = []
    for path in sorted(core_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        records.append(ExperimentRecord(path=path, data=data, mtime=path.stat().st_mtime))
    return records


def deduplicate_records(records: Iterable[ExperimentRecord]) -> List[ExperimentRecord]:
    best: Dict[Tuple, ExperimentRecord] = {}
    for rec in records:
        current = best.get(rec.key)
        if current is None:
            best[rec.key] = rec
            continue

        score_new = (_record_priority(rec.path), rec.mtime)
        score_old = (_record_priority(current.path), current.mtime)
        if score_new > score_old:
            best[rec.key] = rec
    return sorted(
        best.values(),
        key=lambda r: (
            r.data.get("task") or "",
            r.data.get("modality") or "",
            r.data.get("fusion_strategy") or "",
            r.data.get("model") or "",
            r.data.get("window") or "",
            r.data.get("text_method") or "",
            str(r.data.get("note_ablation") or ""),
        ),
    )


def load_experiments(deduplicate: bool = True) -> List[ExperimentRecord]:
    records = load_raw_records()
    return deduplicate_records(records) if deduplicate else records


def find_one(
    records: Iterable[ExperimentRecord],
    *,
    task: str,
    modality: str,
    model: str,
    window: str,
    text_method: Optional[str] = None,
    fusion_strategy: Optional[str] = None,
    note_ablation: Optional[str] = None,
) -> ExperimentRecord:
    hits: List[ExperimentRecord] = []
    for rec in records:
        d = rec.data
        current_ablation = d.get("note_ablation")
        if isinstance(current_ablation, str) and current_ablation.lower() in {"none", "null", ""}:
            current_ablation = None

        if d.get("task") != task:
            continue
        if d.get("modality") != modality:
            continue
        if d.get("model") != model:
            continue
        if d.get("window") != window:
            continue
        if fusion_strategy is not None and d.get("fusion_strategy") != fusion_strategy:
            continue
        if text_method is not None and d.get("text_method") != text_method:
            continue
        if note_ablation is not None and current_ablation != note_ablation:
            continue
        if note_ablation is None and current_ablation is not None:
            continue
        hits.append(rec)

    if len(hits) != 1:
        details = [str(h.path) for h in hits]
        raise ValueError(
            "Expected 1 record, got "
            f"{len(hits)} for task={task}, modality={modality}, model={model}, "
            f"window={window}, text_method={text_method}, fusion_strategy={fusion_strategy}, "
            f"note_ablation={note_ablation}. Hits={details}"
        )
    return hits[0]


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
