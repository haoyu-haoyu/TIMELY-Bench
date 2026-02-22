"""
Build explicit state-space reconstruction artifacts from enhanced episodes.

Outputs:
- results/state_space/state_space_schema.json
- results/state_space/episode_state_trajectory.jsonl
- results/state_space/state_transition_summary.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR


SEVERITY_TO_STATE = {
    "mild": "at_risk",
    "moderate": "active",
    "severe": "severe",
    "critical": "severe",
}

STATE_SCHEMA = {
    "schema": "TIMELY-Bench-StateSpace/1.0",
    "states": [
        {"name": "baseline", "description": "No clinically salient pattern observed in the episode"},
        {"name": "at_risk", "description": "Mild abnormality present"},
        {"name": "active", "description": "Moderate syndrome activity present"},
        {"name": "severe", "description": "Severe/critical syndrome activity present"},
        {"name": "recovering", "description": "State de-escalation after severe or active phase"},
    ],
    "transition_rules": [
        "baseline -> at_risk/active/severe by first detected pattern severity",
        "at_risk -> active/severe by higher-severity events",
        "active -> severe by severe/critical events",
        "severe -> recovering when later events drop to mild/moderate",
        "active/severe -> recovering at hour 24 if no later escalation",
    ],
}


def _state_from_severity(severity: str) -> str:
    return SEVERITY_TO_STATE.get(str(severity).lower(), "at_risk")


def _extract_events(episode: dict):
    detected = episode.get("reasoning", {}).get("detected_patterns", [])
    events = []
    for item in detected:
        if not isinstance(item, dict):
            continue
        hour = item.get("detection_hour")
        if hour is None:
            continue
        try:
            hour = int(float(hour))
        except Exception:
            continue
        events.append({
            "hour": max(0, min(24, hour)),
            "pattern_name": str(item.get("pattern_name", "")),
            "severity": str(item.get("severity", "mild")).lower(),
            "disease": str(item.get("disease", "")),
            "feature": str(item.get("feature", "")),
        })
    events.sort(key=lambda x: (x["hour"], x["pattern_name"]))
    return events


def reconstruct_episode_state(episode: dict):
    stay_id = episode.get("stay_id")
    events = _extract_events(episode)
    trajectory = []

    current_state = "baseline"
    current_hour = 0
    trajectory.append({
        "stay_id": stay_id,
        "hour": 0,
        "state": "baseline",
        "from_state": None,
        "trigger_pattern": None,
        "trigger_severity": None,
        "trigger_disease": None,
        "transition_type": "init",
    })

    for event in events:
        next_state = _state_from_severity(event["severity"])
        if current_state == "severe" and next_state in {"at_risk", "active"}:
            next_state = "recovering"
        if current_state == "active" and next_state == "at_risk":
            next_state = "recovering"

        if next_state != current_state or event["hour"] != current_hour:
            trajectory.append({
                "stay_id": stay_id,
                "hour": event["hour"],
                "state": next_state,
                "from_state": current_state,
                "trigger_pattern": event["pattern_name"],
                "trigger_severity": event["severity"],
                "trigger_disease": event["disease"],
                "transition_type": "pattern_trigger",
            })
            current_state = next_state
            current_hour = event["hour"]

    if current_state in {"active", "severe"} and current_hour < 24:
        trajectory.append({
            "stay_id": stay_id,
            "hour": 24,
            "state": "recovering",
            "from_state": current_state,
            "trigger_pattern": None,
            "trigger_severity": None,
            "trigger_disease": None,
            "transition_type": "end_of_window_decay",
        })
        current_state = "recovering"

    return trajectory, current_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=ROOT_DIR / "episodes" / "episodes_enhanced",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "state_space",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.episodes_dir.glob("TIMELY_v2_*.json"))
    if args.max_episodes and args.max_episodes > 0:
        files = files[:args.max_episodes]

    state_counter = Counter()
    transition_counter = Counter()
    final_state_counter = Counter()
    disease_state_counter = defaultdict(Counter)
    n_failed = 0

    trajectory_path = args.output_dir / "episode_state_trajectory.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as out:
        for idx, ep_path in enumerate(files, start=1):
            try:
                episode = json.loads(ep_path.read_text(encoding="utf-8"))
            except Exception:
                n_failed += 1
                continue

            trajectory, final_state = reconstruct_episode_state(episode)
            final_state_counter[final_state] += 1
            for row in trajectory:
                out.write(json.dumps(row, ensure_ascii=True) + "\n")
                state_counter[row["state"]] += 1
                if row["from_state"] is not None:
                    transition_counter[f"{row['from_state']}->{row['state']}"] += 1
                disease = row.get("trigger_disease")
                if disease:
                    disease_state_counter[disease][row["state"]] += 1

            if idx % 5000 == 0:
                print(f"[state-space] processed {idx}/{len(files)} episodes")

    schema = dict(STATE_SCHEMA)
    schema["generated_at"] = Path(trajectory_path).stat().st_mtime
    schema_path = args.output_dir / "state_space_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=True))

    summary = {
        "n_episodes_processed": len(files),
        "n_episodes_failed": n_failed,
        "state_counts": dict(state_counter),
        "transition_counts": dict(transition_counter),
        "final_state_counts": dict(final_state_counter),
        "disease_state_counts": {k: dict(v) for k, v in disease_state_counter.items()},
        "outputs": {
            "trajectory_jsonl": str(trajectory_path),
            "schema_json": str(schema_path),
        },
    }
    summary_path = args.output_dir / "state_transition_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))

    print(f"Wrote {trajectory_path}")
    print(f"Wrote {schema_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
