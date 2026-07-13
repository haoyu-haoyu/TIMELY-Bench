# Synthetic reproducibility fixture

This directory contains a tiny, deterministic fixture for testing the public
TIMELY-Bench software without credentialed clinical data. Every case, identifier,
measurement, label, and note was manually invented for this repository. Nothing
here was sampled, paraphrased, transformed, or statistically derived from a
MIMIC-IV record. It is not clinical data and must not be used for clinical claims.

The four cases cover AKI, delirium, sepsis, and stroke. Times are relative to a
fictional assessment anchor at hour `0`; negative values precede the anchor, and
no event or note occurs after it. Notes carry an explicit `Entirely fictional
note:` prefix, and every identifier uses the `SYN-` namespace.

Files:

- `schema.json`: JSON Schema for the fixture contract.
- `generate.py`: standard-library-only deterministic generator and validator.
- `fixtures/synthetic_cases.json`: four fictional input cases.
- `fixtures/golden_summary.json`: expected counts, time bounds, safety flags, and
  SHA-256 digest of the canonical fixture bytes.

Verify the committed golden files without changing them:

```bash
python synthetic/generate.py --check
```

Regenerate them deterministically:

```bash
python synthetic/generate.py
```

Generate an isolated copy for experimentation:

```bash
python synthetic/generate.py --output-dir /tmp/timely-bench-synthetic
```

The generator intentionally has no dependency on the project extraction code or
any controlled-data directory. The schema is provided for interoperability; the
generator performs the release-critical validation with only the Python standard
library, so the check works before installing optional research dependencies.
