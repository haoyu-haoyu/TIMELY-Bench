# V3 public release metadata

This directory describes the public reproducibility layer of TIMELY-Bench V3.
It does **not** contain MIMIC-IV patient records or patient-level derivatives.

- `v3_public_release.json` is a machine-readable snapshot of the aggregate data,
  CRES, frozen-provider, and judge-evaluation counts reported by the repository.
- `PUBLIC_ARTIFACT_INVENTORY.csv` records which artifact classes are public and
  which require credentialed controlled access.

Before creating a tag or GitHub release, run:

```bash
make public-checks
```

Then follow [`docs/PUBLIC_RELEASE_CHECKLIST.md`](../docs/PUBLIC_RELEASE_CHECKLIST.md).
The release commit and generated checksums should be filled only after the tree
has passed privacy review.
