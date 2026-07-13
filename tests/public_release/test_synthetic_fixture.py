from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_generator():
    path = ROOT / "synthetic" / "generate.py"
    spec = importlib.util.spec_from_file_location("timely_synthetic_generate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


class SyntheticFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_dir = ROOT / "synthetic" / "fixtures"
        self.fixture_path = self.fixture_dir / "synthetic_cases.json"
        self.summary_path = self.fixture_dir / "golden_summary.json"
        self.fixture = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        self.summary = json.loads(self.summary_path.read_text(encoding="utf-8"))

    def test_committed_artifacts_match_generator_byte_for_byte(self) -> None:
        artifacts = GENERATOR.build_artifacts()
        self.assertEqual([], GENERATOR.check_artifacts(self.fixture_dir, artifacts))

    def test_fixture_satisfies_release_invariants(self) -> None:
        self.assertEqual([], GENERATOR.validate_fixture(self.fixture))
        self.assertFalse(self.fixture["provenance"]["contains_real_patient_data"])
        self.assertFalse(self.fixture["provenance"]["contains_mimic_derived_data"])
        self.assertEqual(
            {"aki", "delirium", "sepsis", "stroke"},
            {case["condition"] for case in self.fixture["cases"]},
        )
        for case in self.fixture["cases"]:
            self.assertTrue(case["synthetic_id"].startswith("SYN-"))
            self.assertTrue(all(event["relative_hour"] <= 0 for event in case["events"]))
            self.assertTrue(all(note["relative_hour"] <= 0 for note in case["notes"]))
            self.assertTrue(
                all(note["text"].startswith("Entirely fictional note:") for note in case["notes"])
            )

    def test_golden_digest_and_counts_are_current(self) -> None:
        digest = hashlib.sha256(self.fixture_path.read_bytes()).hexdigest()
        self.assertEqual(digest, self.summary["fixture_sha256"])
        self.assertEqual(4, self.summary["case_count"])
        self.assertEqual(11, self.summary["event_count"])
        self.assertEqual(4, self.summary["note_count"])
        self.assertEqual(0, self.summary["relative_hour_max"])
        self.assertTrue(self.summary["all_observations_anchor_bounded"])

    def test_schema_is_valid_json_and_declares_synthetic_contract(self) -> None:
        schema = json.loads((ROOT / "synthetic" / "schema.json").read_text(encoding="utf-8"))
        self.assertEqual("https://json-schema.org/draft/2020-12/schema", schema["$schema"])
        provenance = schema["properties"]["provenance"]["properties"]
        self.assertIs(provenance["contains_real_patient_data"]["const"], False)
        self.assertIs(provenance["contains_mimic_derived_data"]["const"], False)


if __name__ == "__main__":
    unittest.main()
