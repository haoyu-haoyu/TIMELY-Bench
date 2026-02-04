import hashlib
import json
from pathlib import Path

root = Path("/scratch/users/k25113331/TIMELY-Bench_Final")
out_dir = root / "results" / "llm_annotations"
meta_path = out_dir / "ANNOTATION_METADATA.json"
ann_path = out_dir / "annotations_rule_based_20260127_151413.jsonl"
sample_path = out_dir / "llm_annotation_set.csv"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

meta = json.loads(meta_path.read_text())
meta["annotation_set_path"] = str(sample_path)
meta["annotation_set_sha256"] = sha256_file(sample_path)
meta["annotations_path"] = str(ann_path)
meta["annotations_sha256"] = sha256_file(ann_path)
meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True))
print("updated", meta_path)
