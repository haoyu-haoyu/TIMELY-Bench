# npj Digital Medicine Submission Bundle

This directory contains the current `npj Digital Medicine` submission-oriented
LaTeX bundle for TIMELY Bench.

## Files

- `timely_bench_npj_article.tex`
  - main article draft
- `sn-jnl.cls`
  - Springer Nature class file used by the draft bundle
- `sn-nature.bst`
  - Nature-style bibliography file for later citation integration
- `FORMAT_REQUIREMENTS.md`
  - venue-specific formatting notes and source links
- `figures/main/`
  - main-text figure assets
- `tables/`
  - generated table environments from the frozen CSV outputs

## Figure 1 asset

The main TeX file points to:

- `figures/main/figure1_final.png`

This is the dedicated swap-in path for the user-approved final Figure 1 image.
If a different raster or vector version is preferred later, replacing that file
is sufficient and no TeX edits are required.

## Known placeholders

The following manuscript fields are intentionally left as placeholders because
they should not be invented:

- author list
- affiliations
- corresponding author email
- acknowledgements
- author contributions
- competing interests
- data availability
- code availability
- final citation layer

## Main source provenance

- manuscript text originates from the frozen Phase 6.5 writing drafts under
  `results/cres_v3/`
- table environments are generated from the frozen CSV outputs under
  `results/cres_v3/paper_tables/`
- table generation script:
  - `code/v3/build_npj_submission_tables.py`

## Compile

From this directory, try:

```bash
tectonic timely_bench_npj_article.tex
```

If the final journal submission requires a single-file `.tex` with inlined
tables rather than `\input{}` table fragments, this bundle can be flattened in
a later cleanup step without changing the manuscript text.
