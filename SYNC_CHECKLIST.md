# TIMELY-Bench Sync Checklist (Canonical)

Status: ACTIVE
Updated: 2026-02-20

## Path Variables

- `CREATE_PROJECT_ROOT`: `${CREATE_PROJECT_ROOT}` (example: `/scratch/users/<user>/TIMELY-Bench_Final`)
- `LOCAL_PROJECT_ROOT`: `${LOCAL_PROJECT_ROOT}`

## Rules

1. CREATE is the runtime source of truth for heavy jobs.
2. Release artefacts must remain path-portable (`${PROJECT_ROOT}` markers only).
3. Sync must be checksum-based for canonical files.

## Canonical Sync Commands

```bash
# Local -> CREATE (release + key scripts)
rsync -av --delete "${LOCAL_PROJECT_ROOT}/final_release/" "<user>@hpc.create.kcl.ac.uk:${CREATE_PROJECT_ROOT}/final_release/"
rsync -av "${LOCAL_PROJECT_ROOT}/code/data_processing/build_final_release_bundle.py" "<user>@hpc.create.kcl.ac.uk:${CREATE_PROJECT_ROOT}/code/data_processing/"

# CREATE -> Local (if CREATE produced newer artefacts)
rsync -av --delete "<user>@hpc.create.kcl.ac.uk:${CREATE_PROJECT_ROOT}/final_release/" "${LOCAL_PROJECT_ROOT}/final_release/"
```

## Verification Gates

```bash
# 1) No plaintext keys
rg -n "sk-[A-Za-z0-9_-]{10,}" code scripts final_release README*.md docs || true

# 2) Release checksums
cd final_release && sha256sum -c CHECKSUMS.sha256

# 3) Canonical QA evidence
ls final_release/evidence/final_qa_*.json final_release/evidence/final_qa_*.md
```

## Notes

- Historical sync notes were archived under `documentation/archive_legacy/`.
- Do not use obsolete job IDs or stale QA filenames in active documents.
