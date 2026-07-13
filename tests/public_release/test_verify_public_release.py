from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_verifier():
    path = ROOT / "tools" / "verify_public_release.py"
    spec = importlib.util.spec_from_file_location("timely_verify_public_release", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_minimal_release(root: Path) -> None:
    readme_lines = [
        "# Test release",
        "AKI, delirium, sepsis, stroke.",
    ]
    readme_lines.extend(VERIFIER.REQUIRED_README_NUMBERS.values())
    readme_lines.extend(VERIFIER.REQUIRED_AGGREGATE_FILES)
    _write(root / "README.md", "\n".join(readme_lines) + "\n")
    for relative in VERIFIER.REQUIRED_AGGREGATE_FILES:
        path = root / relative
        if path.suffix == ".json":
            _write(path, "{}\n")
        elif path.suffix == ".csv":
            _write(path, "metric,value\nexample,1\n")
        else:
            _write(path, "# Aggregate summary\n")


class PublicReleaseVerifierTests(unittest.TestCase):
    def test_minimal_compliant_release_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_minimal_release(root)
            report = VERIFIER.verify_repository(root)
            self.assertTrue(report.ok, report.findings)
            self.assertEqual(3, report.json_files)
            self.assertEqual(8, report.csv_files)

    def test_detects_prohibited_artifacts_and_row_level_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_minimal_release(root)
            _write(root / "data" / "processed" / "rows.csv", "subject_id,value\n1,2\n")
            _write(root / "results" / "hidden.jsonl", "{}\n")
            report = VERIFIER.verify_repository(root)
            checks = {finding.check for finding in report.findings}
            self.assertIn("path-policy", checks)
            self.assertIn("row-level-data", checks)

    def test_detects_malformed_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_minimal_release(root)
            _write(root / "broken.json", '{"unfinished": true')
            _write(root / "broken.csv", 'metric,value\n"unfinished,1\n')
            report = VERIFIER.verify_repository(root)
            parse_paths = {
                finding.path
                for finding in report.findings
                if finding.check in {"json-parse", "csv-parse"}
            }
            self.assertEqual({"broken.csv", "broken.json"}, parse_paths)

    def test_detects_absolute_user_path_and_high_confidence_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_minimal_release(root)
            user_path = "/" + "Users/example/private/file.txt"
            fake_key = "sk-" + ("x" * 32)
            _write(root / "unsafe.md", f"path={user_path}\nkey={fake_key}\n")
            report = VERIFIER.verify_repository(root)
            checks = {finding.check for finding in report.findings}
            self.assertIn("absolute-path", checks)
            self.assertIn("secret-scan", checks)

    def test_detects_missing_readme_fact_and_aggregate_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_minimal_release(root)
            readme = (root / "README.md").read_text(encoding="utf-8")
            (root / "README.md").write_text(readme.replace("477,630", "many"), encoding="utf-8")
            missing = root / VERIFIER.REQUIRED_AGGREGATE_FILES[-1]
            missing.unlink()
            report = VERIFIER.verify_repository(root)
            self.assertTrue(any(finding.check == "readme" for finding in report.findings))
            self.assertTrue(any(finding.check == "aggregate-artifact" for finding in report.findings))


if __name__ == "__main__":
    unittest.main()
