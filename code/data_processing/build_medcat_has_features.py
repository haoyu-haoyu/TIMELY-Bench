"""
Build stay-level has_* MedCAT features from note-level concepts.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE_DIR / 'data' / 'processed' / 'medcat_full' / 'medcat_note_concepts_24h.csv'
DEFAULT_OUTPUT = BASE_DIR / 'data' / 'processed' / 'medcat_full' / 'medcat_has_concepts_24h.csv'
DEFAULT_EPISODES_DIR = BASE_DIR / 'episodes' / 'episodes_enhanced'


PATTERNS = {
    'has_sepsis': re.compile(r'\bsepsis\b|\bseptic\b|\bsepticemia\b|\bsepticaemia\b', re.I),
    'has_pneumonia': re.compile(r'\bpneumonia\b|\bpneumonitis\b', re.I),
    'has_infection': re.compile(
        r'\binfection\b|\binfectious\b|\bbacteremia\b|\bbacteraemia\b|\bcellulitis\b'
        r'|\babscess\b|\buti\b|\burinary tract infection\b|\bmeningitis\b'
        r'|\bendocarditis\b|\bosteomyelitis\b',
        re.I
    ),
    'has_antibiotic': re.compile(
        r'\bantibiotic\b|\bantibacterial\b|\bantimicrobial\b|\banti-?infective\b',
        re.I
    ),
    'has_aki': re.compile(
        r'\bacute kidney injury\b|\bacute renal failure\b|\bacute renal injury\b'
        r'|\bacute kidney failure\b|\bacute renal insufficiency\b|\baki\b',
        re.I
    ),
    'has_ards': re.compile(
        r'\bacute respiratory distress syndrome\b|\badult respiratory distress syndrome\b'
        r'|\bacute lung injury\b|\bards\b',
        re.I
    ),
}

NEGATION_MARKERS = ('neg', 'no', 'denied', 'absent', 'without')
HISTORICAL_MARKERS = ('histor', 'past', 'prior')


def normalize_stay_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_all_stays(episodes_dir: Path):
    if not episodes_dir.exists():
        return []
    stay_ids = []
    for path in episodes_dir.glob('TIMELY_v2_*.json'):
        stay_id = normalize_stay_id(path.stem.split('_')[-1])
        if stay_id is not None:
            stay_ids.append(stay_id)
    return sorted(set(stay_ids))


def _is_negated(value) -> bool:
    if value is None:
        return False
    val = str(value).strip().lower()
    if not val or val == 'nan':
        return False
    return any(marker in val for marker in NEGATION_MARKERS)


def _is_historical(value) -> bool:
    if value is None:
        return False
    val = str(value).strip().lower()
    if not val or val == 'nan':
        return False
    return any(marker in val for marker in HISTORICAL_MARKERS)


def build_features(input_path: Path, output_path: Path, include_all: bool, episodes_dir: Path):
    counts_pos = defaultdict(lambda: {key: 0 for key in PATTERNS})
    counts_neg = defaultdict(lambda: {key: 0 for key in PATTERNS})
    matched_rows = 0

    with open(input_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay_id = normalize_stay_id(row.get('stay_id'))
            name = (row.get('name') or '').strip().lower()
            negation = row.get('negation')
            temporality = row.get('temporality')
            if stay_id is None or not name:
                continue
            is_neg = _is_negated(negation) or _is_historical(temporality)
            for key, pattern in PATTERNS.items():
                if pattern.search(name):
                    if is_neg:
                        counts_neg[stay_id][key] += 1
                    else:
                        counts_pos[stay_id][key] += 1
                    matched_rows += 1

    if include_all:
        stay_ids = load_all_stays(episodes_dir)
    else:
        stay_ids = sorted(set(counts_pos.keys()) | set(counts_neg.keys()))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos_keys = list(PATTERNS.keys())
    neg_keys = [f'{k}_neg' for k in pos_keys]
    fieldnames = ['stay_id'] + pos_keys + neg_keys
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stay_id in stay_ids:
            row = {'stay_id': stay_id}
            features_pos = counts_pos.get(stay_id, {})
            features_neg = counts_neg.get(stay_id, {})
            for key in PATTERNS:
                row[key] = 1 if features_pos.get(key, 0) > 0 else 0
                row[f'{key}_neg'] = 1 if features_neg.get(key, 0) > 0 else 0
            writer.writerow(row)

    print("MedCAT has_* feature build complete")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Stays: {len(stay_ids):,}")
    print(f"  Matched concept rows: {matched_rows:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument('--include-all-stays', action='store_true', help='Fill zero rows for all episodes')
    parser.add_argument('--episodes-dir', type=str, default=str(DEFAULT_EPISODES_DIR))
    args = parser.parse_args()

    build_features(
        input_path=Path(args.input),
        output_path=Path(args.output),
        include_all=args.include_all_stays,
        episodes_dir=Path(args.episodes_dir),
    )


if __name__ == '__main__':
    main()
