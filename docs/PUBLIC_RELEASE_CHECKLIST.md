# Public release checklist

Use this checklist before publishing a TIMELY-Bench V3 tag, GitHub release, or
archival DOI. It complements, but does not replace, institutional review or
PhysioNet guidance.

## 1. Freeze the software tree

- [ ] Confirm `git status --short` is empty.
- [ ] Record the final commit in `release/v3_public_release.json`.
- [ ] Create an annotated, immutable version tag.
- [ ] Obtain project approval for a software license and add the corresponding
      `LICENSE` file; do not infer a license from the manuscript or data DUA.
- [ ] Confirm the README, manuscript, and aggregate result tables use the same
      cohort definition and counts.

## 2. Enforce the public/controlled boundary

- [ ] Run `make verify-public` and inspect every reported exception.
- [ ] Confirm no raw or derived patient-level files are tracked.
- [ ] Confirm no filled prompts, canonical responses, per-instance scores, or
      long-form judge rationales are tracked.
- [ ] Confirm synthetic examples were generated from rules and were not adapted
      from real MIMIC-IV records.
- [ ] Review small-cell aggregate outputs and rare subgroup summaries.
- [ ] Remove API keys, credentials, internal endpoints, user home paths, CREATE
      scratch paths, Slurm logs, and temporary environment files.

## 3. Audit Git history, not only the current branch

- [ ] Scan every reachable object, tag, release asset, workflow artifact, and
      fork for controlled files.
- [ ] Review the historical `phase65f_scored_prompts.parquet` and
      `phase65f_judge_scores_long.csv` objects.
- [ ] If they are not approved for public retention, rewrite history with a
      reviewed procedure or publish from a new clean repository.
- [ ] Re-run secret and large-file scans after history cleanup.

## 4. Verify reproducibility metadata

- [ ] Pin the public environment and record the Python version.
- [ ] Record MIMIC-IV version/DOI without redistributing its files.
- [ ] Record model identifiers, revisions where available, decoding parameters,
      random seeds, inference dates, and retry rules.
- [ ] State that hosted-model outputs may drift and that exact frozen rescoring
      requires the controlled canonical response package.
- [ ] Preserve the local-final-sync provenance of the three-judge evaluation.

## 5. Create the release

- [ ] Generate checksums for the final public tree.
- [ ] Run `make public-checks` from a fresh clone.
- [ ] Build and inspect the manuscript from the tagged tree.
- [ ] Create the GitHub release and, if desired, archive the same tag with
      Zenodo or another approved software repository.
- [ ] Keep patient-level derivatives in a PhysioNet credentialed project or an
      equivalently governed institutional environment.

## Recommended release statement

> The public release contains complete methods code, aggregate results, schemas,
> and synthetic end-to-end validation. Credentialed MIMIC-IV access and the
> controlled derived-artifact package are required to rebuild the patient-level
> benchmark and verify the exact frozen per-instance outputs.
