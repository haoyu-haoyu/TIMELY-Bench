# TIMELY-Bench Dataset (v2.0)

TIMELY-Bench is a benchmark dataset for **time-aligned fusion** of ICU structured time-series and clinical text derived from MIMIC-IV.

This repository currently contains **two episode formats**:

1. **Canonical v2.0 episodes (recommended):** per-episode JSON files in `episodes/episodes_enhanced/` (schema version is recorded in `metadata.schema_version`).
2. **Legacy v1 JSONL snapshot:** `dataset/timely_bench_episodes.jsonl` (older schema using `notes_spans` / `pattern_detections`).

All dataset-level reporting in the paper and in `results/` uses the **v2.0 episode format**.

---

## 1. Canonical Episode Structure (v2.0)

Each episode file `episodes/episodes_enhanced/TIMELY_v2_*.json` is a single ICU stay and follows this high-level structure:

```json
{
  "episode_id": "TIMELY_v2_<stay_id>",
  "stay_id": 30000153,
  "patient": { "subject_id": 123456 },
  "metadata": {
    "schema_version": "TIMELY-Episode/2.x",
    "observation_window_hours": 24,
    "source_database": "MIMIC-IV",
    "source_version": "v3.1"
  },
  "timeseries": {
    "start_hour": 0,
    "end_hour": 24,
    "resolution_hours": 1,
    "n_timepoints": 24,
    "missing_rate": { "heart_rate": 0.042, "creatinine": 0.917, "...": 0.0 },
    "vitals": [
      { "hour": 0, "timestamp": "...", "heart_rate": 85, "sbp": 120, "...": null },
      { "hour": 1, "timestamp": "...", "heart_rate": 90, "sbp": 118, "...": null }
    ],
    "labs": [
      { "hour": 2, "timestamp": "...", "creatinine": 1.2, "wbc": 10.4, "...": null }
    ]
  },
  "clinical_text": {
    "n_notes": 77,
    "note_types": ["nursing", "lab_comment"],
    "notes": [
      {
        "note_id": "218174_nursing",
        "note_type": "nursing",
        "note_category": "Routine Vital Signs: Heart Rhythm",
        "chart_hour": 2,
        "text_relevant": "SR (Sinus Rhythm)",
        "text_full": null
      }
    ],
    "aligned_spans": [],
    "coverage_hours": [],
    "llm_features": {}
  },
  "reasoning": {
    "detected_patterns": { "tachycardia": { "...": "..." } },
    "n_alignments": 77,
    "n_supportive": 0,
    "n_contradictory": 0,
    "condition_graph": {
      "nodes": [{ "id": "pattern_tachycardia_8", "level": "pattern", "onset_hour": 8, "...": "..." }],
      "edges": [{ "source_id": "pattern_tachycardia_8", "target_id": "condition_sirs_criteria", "relationship": "indicates", "...": "..." }]
    }
  },
  "labels": {
    "outcome": { "mortality": 0, "prolonged_los": 0 },
    "process": { "sepsis_onset_hour": 1, "aki_stage_max": 0 },
    "has_sepsis": false,
    "has_aki": false,
    "has_ards": false
  }
}
```

Notes:
- `timeseries.vitals` and `timeseries.labs` are **timepoint-major** lists (each entry is a timepoint record).
- `reasoning.condition_graph` is an **episode-level simplified graph** intended as a lightweight event scaffold.
- Canonical, guideline-anchored condition graphs and physiology templates are released separately under `final_release/`.

---

## 2. Dataset Statistics (v2.0)

Canonical statistics are stored in `dataset/dataset_stats.json`:

- Total Episodes: 74,829
- Episodes with Notes: 74,811
- Episodes with Patterns: 74,812
- Episodes with Alignments: 74,786
- Total Notes: 6,975,132
- Total Pattern Events: 3,760,396
- Total Alignments: 6,974,406

---

## 3. Files and Folders

### Recommended (v2.0)

- `episodes/episodes_enhanced/`: canonical per-episode JSON files (v2.0 schema).
- `dataset/dataset_stats.json`: dataset-level counts used by the report/paper.
- `final_release/`: release bundle (condition graphs, physiology templates, QC, evidence, CRES).

### Legacy (v1 snapshot)

- `dataset/timely_bench_episodes.jsonl`: older JSONL episode format.
- `dataset/sample_episodes.json`, `dataset/condition_graph.json`: older artefacts kept for backward compatibility.

---

## 4. Usage (v2.0 Episodes)

```python
import json
from pathlib import Path

episodes_dir = Path("episodes/episodes_enhanced")
sample_file = next(episodes_dir.glob("TIMELY_v2_*.json"))

ep = json.loads(sample_file.read_text())
print(ep["stay_id"], ep["labels"]["outcome"]["mortality"])
print(ep["clinical_text"]["note_types"])
print(len(ep["timeseries"]["vitals"]))
```

---

## License

This dataset is derived from MIMIC-IV and requires PhysioNet credentialing.
