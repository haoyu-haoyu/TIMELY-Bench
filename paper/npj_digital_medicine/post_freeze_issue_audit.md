# Post-freeze Issue Audit

## Summary

- Total issue categories checked: 13
- Dominant status counts: PRESENT 9, ABSENT 3, UNCERTAIN 1
- Severity counts among PRESENT/UNCERTAIN findings: BLOCKING 1, MAJOR 2, MINOR 5, STYLE 2

High-level conclusion:
- The current figures and compiled manuscript PDF are synchronized at the figure/caption level.
- No undefined citations, undefined references, duplicate labels, or missing figure files were found.
- A minimal patch is recommended before submission.
- Author input is required for the author block, acknowledgements, and author contributions.

## Issues

### 1. Figure 6 panel reference consistency

Issue: Figure 6 body references do not match the current three-panel Figure 6.

Status: PRESENT

Severity: MAJOR

Evidence:
- `timely_bench_npj_article.tex:194`
- Exact snippet: `Stay-level template-support coverage from the frozen Phase 5 summaries was 6\% for AKI, 96\% for delirium, and 37\% for sepsis (Fig.~\ref{fig:trajectory_tier}, Panel A).`
- `timely_bench_npj_article.tex:196`
- Exact snippet: `Row-level performance then varied systematically across trajectory tiers (Fig.~\ref{fig:trajectory_tier}, Panel B).`
- `timely_bench_npj_article.tex:203`
- Exact snippet: `Panel a shows stay-level template support coverage... Panel b shows sample-instance composition... Panel c shows row-level auto-scored mean primary score by trajectory tier.`

Why it matters:
- Current Figure 6 uses lowercase panel labels and has three panels: panel a support coverage, panel b composition, panel c performance.
- The body text uses uppercase `Panel A/B`.
- More importantly, row-level performance is currently panel c, not panel B.

Recommended fix:
- Change line 194 to refer to `panel a`.
- Change line 196 to refer to `panel c`.
- If mentioning composition, refer to `panel b`.
- Use lowercase panel references consistently.

### 2. Figure 2 comparator definition

Issue: Figure 2 comparator definition is ambiguous relative to Table 2.

Status: PRESENT

Severity: MAJOR

Evidence:
- `timely_bench_npj_article.tex:96`
- Exact snippet: `Using the best available B1 model for each task, B1 improved AUROC on only three of seven eligible tasks... It improved AUPRC on only one task...`
- `timely_bench_npj_article.tex:105`
- Exact snippet: `Panel a shows delta AUROC and panel b shows delta AUPRC, both computed as Branch B1 minus Branch A...`
- `tables/table2.tex:16-18`
- Exact snippet:
  - `DEL-S1 & Delirium & A & xgboost & 0.918 & 0.702 \\`
  - `DEL-S1 & Delirium & B1 & temporal\_transformer & 0.911 & 0.689 \\`
  - `DEL-S1 & Delirium & B1 & bilstm\_attention & 0.910 & 0.702 \\`
- `figures/main/figure2_branch_delta_first.svg:419`
- Exact snippet: `-0.013`

Why it matters:
- Figure 2 panel b reports DEL-S1 AUPRC delta as `-0.013`, which equals `0.689 - 0.702`.
- Table 2 shows BiLSTM-attention has DEL-S1 AUPRC 0.702, matching Branch A.
- Therefore Figure 2 appears to use the AUROC-selected B1 model's AUPRC for each task, not the metric-specific best B1 model for AUPRC.
- The current phrase `best available B1 model for each task` can be read as metric-specific best B1, which would make DEL-S1 AUPRC delta 0.000 rather than -0.013.

Recommended fix:
- Do not recompute Figure 2 unless the intended comparator is metric-specific best B1.
- If the intended comparator is AUROC-selected B1 per task, clarify in the Figure 2 caption and Results text, for example:
  - `For each task, Branch B1 was represented by the B1 model selected by AUROC, and both AUROC and AUPRC deltas use that same selected model.`
- If the intended comparator is metric-specific best B1, then Figure 2 panel b should be recomputed and the DEL-S1 AUPRC delta should not be `-0.013`.

### 3. Final PDF synchronization with current TeX and figures

Issue: Current manuscript PDF may be stale relative to latest TeX/figures.

Status: ABSENT

Severity: N/A

Evidence:
- Current PDF metadata from `pdfinfo`: `CreationDate: Tue May 5 07:18:15 2026 BST`.
- `timely_bench_npj_article.tex` timestamp: `2026-05-05 07:17:50 BST`.
- Current figure PNG timestamps are earlier than the PDF timestamp.
- `timely_bench_npj_article.tex:55`, `104`, `127`, `166`, `183`, `202`, `221` include the current Figure 1-7 PNG paths.
- PDF text checks confirm:
  - Figure 1 caption says `The bottom scale strip summarizes the frozen evaluation scale.`
  - Figure 3 caption includes D1-D5 auto-scored and D6 judge-deferred wording.
  - Figure 6 caption describes panel a/b/c.
  - Figure 7 caption says all provider points are labeled and includes judge packet details.

Why it matters:
- Stale PDFs are a common post-freeze failure mode.

Recommended fix:
- No action needed.

### 4. Figure includegraphics format

Issue: Manuscript includes PNG figures even though PDF/SVG outputs exist.

Status: PRESENT

Severity: STYLE

Evidence:
- `timely_bench_npj_article.tex:55`: `\includegraphics[width=\textwidth]{figures/main/figure1_information_state_anatomy.png}`
- `timely_bench_npj_article.tex:104`: `\includegraphics[width=\textwidth]{figures/main/figure2_branch_delta_first.png}`
- `timely_bench_npj_article.tex:127`: `\includegraphics[width=\textwidth]{figures/main/figure3_provider_condition_cres.png}`
- `timely_bench_npj_article.tex:166`: `\includegraphics[width=\textwidth]{figures/main/figure4_temporal_stress_test.png}`
- `timely_bench_npj_article.tex:183`: `\includegraphics[width=\textwidth]{figures/main/figure5_stroke_layer_mechanism.png}`
- `timely_bench_npj_article.tex:202`: `\includegraphics[width=\textwidth]{figures/main/figure6_template_state_space_stratification.png}`
- `timely_bench_npj_article.tex:221`: `\includegraphics[width=\textwidth]{figures/main/figure7_judge_validation_calibration.png}`
- Corresponding PDF and SVG files exist for all seven figures.
- PNG resolutions:
  - Figure 1: 1672 x 941
  - Figures 2-7: width 3188 px, heights 1372-2767 px

Why it matters:
- Nature-style submissions generally prefer vector graphics for charts when feasible.
- However, manuscript-size visual QA passed with current PNGs.

Recommended fix:
- Optional: switch Figure 2-7 to PDF includes for vector output, if journal submission workflow accepts embedded PDFs.
- Safe alternative: keep PNG includes because current page-scale QA passed and high-resolution PNGs are available.

### 5. Table 1B layer spacing

Issue: Table 1B uses `Layer1` and `Layer2` without spaces.

Status: PRESENT

Severity: STYLE

Evidence:
- `tables/table1b.tex:18`: `S-R1 & Stroke & Layer2 retrospective`
- `tables/table1b.tex:19`: `S-R2 & Stroke & Layer2 retrospective`
- `tables/table1b.tex:20`: `S-R3 & Stroke & Layer2 retrospective`
- `tables/table1b.tex:21`: `S-R4 & Stroke & Layer2 retrospective`
- `tables/table1b.tex:22`: `S-T1 & Stroke & Layer1 temporal`
- `tables/table1b.tex:23`: `S-T2 & Stroke & Layer1 temporal`
- `tables/table1b.tex:24`: `S-T3 & Stroke & Layer1 temporal`
- `tables/table1b.tex:25`: `S-T4 & Stroke & Layer1 temporal`

Why it matters:
- The manuscript prose and Figure 5 use `Layer 1` and `Layer 2`, so the table should match.

Recommended fix:
- Replace `Layer1 temporal` with `Layer 1 temporal`.
- Replace `Layer2 retrospective` with `Layer 2 retrospective`.

### 6. Table 3 latency and token column clarity

Issue: Table 3 column labels lack unit/semantic specificity.

Status: UNCERTAIN

Severity: MINOR

Evidence:
- `tables/table3.tex:8`
- Exact snippet: `Provider & Tier & Macro score & Pair wins & Avg latency & Tokens \\`
- Values include `15.938`, `30.038`, `343,643,633`, etc.

Why it matters:
- `Avg latency` does not state the unit.
- `Tokens` does not state whether the value is input tokens, output tokens, total tokens, or tokens processed.
- The numeric values suggest seconds for latency and total/processed tokens for tokens, but the file alone does not confirm this.

Recommended fix:
- Confirm units from the source summary before editing.
- If latency is seconds, consider `Avg latency (s)`.
- If tokens are total processed tokens, consider `Total tokens` or `Tokens processed`.

### 7. Table 5 judge score scale

Issue: Table 5 caption does not state that judge scores are on a 1-5 scale.

Status: PRESENT

Severity: MINOR

Evidence:
- `tables/table5.tex:2`
- Exact snippet: `Judge-based provider summary for the four fixed contestants, reporting mean scores across the judge dimensions.`
- `tables/table5.tex:8-13` contains judge mean values such as `4.397`, `3.894`, `1.813`.

Why it matters:
- Readers need the score scale to interpret judge mean values.
- Figure 7 colorbar states `Mean score (1-5)`, but Table 5 does not.

Recommended fix:
- Add scale information to Table 5 caption, for example:
  - `Scores are mean rubric scores on a 1-5 scale.`

### 8. Table 6 exact match wording

Issue: Table 6 column label says `Exact match` although values appear to be rates/proportions.

Status: PRESENT

Severity: MINOR

Evidence:
- `tables/table6.tex:8`
- Exact snippet: `Judge A & Judge B & Score & Rows & Spearman $\rho$ & Exact match \\`
- `tables/table6.tex:10-15` contains values `0.283`, `0.286`, `0.527`, `0.511`, `0.410`, `0.411`.

Why it matters:
- Decimal values in this context are exact-match rates, not counts or binary labels.

Recommended fix:
- Rename column to `Exact match rate`.

### 9. Submission-blocking placeholders and author-input fields

Issue: Submission metadata placeholders remain.

Status: PRESENT

Severity: BLOCKING

Evidence:
- `timely_bench_npj_article.tex:21`: `% Replace this placeholder author block before submission.`
- `timely_bench_npj_article.tex:22`: `\author*[1]{\fnm{Placeholder} \sur{Author}}\email{corresponding.author@example.com}`
- `timely_bench_npj_article.tex:23`: `\affil*[1]{\orgdiv{To be completed}, \orgname{To be completed}, \orgaddress{\city{To be completed}, \country{To be completed}}}`
- `timely_bench_npj_article.tex:390`: `\textit{To be completed before submission.}`
- `timely_bench_npj_article.tex:394`: `\textit{To be completed before submission.}`
- `timely_bench_npj_article.tex:382`: `Derived benchmark artifacts... will be released upon publication.`
- `timely_bench_npj_article.tex:386`: `The benchmark assembly, prompt serialization, and scoring code will be released in a public repository upon publication.`
- `timely_bench_npj_article.tex:398`: `The authors declare no competing interests.`

Why it matters:
- Author block, affiliation, acknowledgement, and author contribution placeholders are blocking before submission.
- Data/code release language may be acceptable until final submission if repository URLs are not yet public, but the journal may require stronger availability details.
- Competing interests is complete if accurate.

Recommended fix:
- Author input required:
  - Replace placeholder author, email, and affiliation.
  - Fill acknowledgements.
  - Fill author contributions.
- Confirm whether data/code availability should include repository URLs, accession identifiers, or embargo language before submission.

### 10. Bibliography, references, and labels

Issue: Undefined citations, undefined references, duplicate labels, or missing labels.

Status: ABSENT

Severity: N/A

Evidence:
- `.bib` file exists: `timely_bench_refs.bib`.
- Compile command used:
  - `tectonic --keep-logs --keep-intermediates --outdir /private/tmp/timely_bench_compile_audit_logs timely_bench_npj_article.tex`
- BibTeX log:
  - `Database file #1: timely_bench_refs.bib`
- Static label/citation check across main TeX and table files:
  - labels: 15
  - duplicate labels: none
  - missing references: none
  - citations: 57
  - missing citation keys: none
- Log grep did not find undefined citations or undefined references.

Why it matters:
- Undefined citations/references are submission-blocking formatting defects.

Recommended fix:
- No action needed for citation/reference resolution.

### 11. LaTeX compile warnings

Issue: Compile succeeds but emits overfull/underfull warnings and a rerun warning from Tectonic.

Status: PRESENT

Severity: MINOR

Evidence:
- Compile output includes repeated underfull vbox warnings.
- Compile output includes:
  - `tables/table1b.tex:7: Overfull \hbox (85.08363pt too wide)`
  - `tables/table1b.tex:11: Overfull \hbox (85.08363pt too wide)`
  - `tables/table1b.tex:27: Overfull \hbox (85.08363pt too wide)`
  - `tables/table6.tex:19: Overfull \hbox (14.22708pt too wide)`
  - `tables/table7.tex:18: Overfull \hbox (9.10036pt too wide)`
  - `warning: TeX rerun seems needed, but stopping at 6 passes`
- Output PDF was still written successfully:
  - `/private/tmp/timely_bench_compile_audit_logs/timely_bench_npj_article.pdf`

Why it matters:
- Most underfull warnings are not content problems.
- Overfull table warnings can indicate table width stress, especially Table 1B.
- Page-scale visual inspection showed the affected tables are readable, but this remains a layout polish risk.

Recommended fix:
- Not blocking if current visual PDF is accepted.
- Consider fixing Table 1B width/spacing and Table 6/7 minor overfulls in a later polish patch.

### 12. Caption-to-figure consistency

Issue: Caption mismatch with current figures.

Status: PRESENT for Figure 2/6 text-level issues; ABSENT for other figure captions.

Severity: MAJOR for Figure 2/6; N/A for others.

Evidence:
- Figure 1 caption:
  - `timely_bench_npj_article.tex:56`
  - Correctly describes panels a/b/c and bottom scale strip.
- Figure 2 caption:
  - `timely_bench_npj_article.tex:105`
  - States delta as Branch B1 minus Branch A but does not specify whether B1 is AUROC-selected or metric-specific.
- Figure 3 caption:
  - `timely_bench_npj_article.tex:128`
  - Includes D1-D5 auto-scored and D6 judge-deferred wording; no CI wording.
- Figure 4 caption:
  - `timely_bench_npj_article.tex:167`
  - Explains T6/T24/T48 and `fewer than 25 instances` as insufficient support rather than missing data.
- Figure 5 caption:
  - `timely_bench_npj_article.tex:184`
  - Explains Layer 1/Layer 2 regimes and says Layer 2 outperformed Layer 1 for all nine providers.
- Figure 6 caption:
  - `timely_bench_npj_article.tex:203`
  - Correctly describes panel a/b/c.
- Figure 7 caption:
  - `timely_bench_npj_article.tex:222`
  - States all provider points are labeled and gives judge packet details.

Why it matters:
- Figure captions are mostly synchronized.
- The remaining issue is not caption-panel mismatch for Figure 6; it is body text panel reference mismatch.
- Figure 2 comparator ambiguity remains.

Recommended fix:
- Apply the Figure 2 comparator clarification.
- Apply the Figure 6 body panel-reference fix.

### 13. Old terminology / old-version residue

Issue: Old high-risk terminology may remain.

Status: ABSENT for high-risk old figure/provider terms; PRESENT for expected table wording issues.

Severity: STYLE/MINOR for table wording issues.

Evidence:
- Grep command searched:
  - `B2/B3 prompt`
  - `B2 prompt + state metadata`
  - `Panel d`
  - `unlabeled points`
  - `Gemini 3.1` not followed by `Pro`
  - `Gemma4` not followed by `-26B`
  - `MedGemma` not followed by `-4B`
  - `Meditron` not followed by `3-8B`
  - `Layer1`
  - `Layer2`
  - `Exact match`
  - `Avg latency`
  - `Tokens`
- No hits for old Figure 1 prompt wording, old Panel d wording, old unlabeled-points wording, or noncanonical provider labels.
- Hits remain for:
  - `tables/table1b.tex:18-25` (`Layer1`, `Layer2`)
  - `tables/table6.tex:8` (`Exact match`)
  - `tables/table3.tex:8` (`Avg latency`, `Tokens`)

Why it matters:
- High-risk old figure/caption residues are absent.
- Remaining hits are the table wording issues already listed above.

Recommended fix:
- No action needed for old figure/provider terminology.
- Address table wording issues in a minimal patch if desired.

## Files inspected

- `paper/npj_digital_medicine/timely_bench_npj_article.tex`
- `paper/npj_digital_medicine/tables/table1a.tex`
- `paper/npj_digital_medicine/tables/table1b.tex`
- `paper/npj_digital_medicine/tables/table2.tex`
- `paper/npj_digital_medicine/tables/table3.tex`
- `paper/npj_digital_medicine/tables/table4.tex`
- `paper/npj_digital_medicine/tables/table5.tex`
- `paper/npj_digital_medicine/tables/table6.tex`
- `paper/npj_digital_medicine/tables/table7.tex`
- `paper/npj_digital_medicine/figures/main/figure1_information_state_anatomy.pdf`
- `paper/npj_digital_medicine/figures/main/figure1_information_state_anatomy.svg`
- `paper/npj_digital_medicine/figures/main/figure1_information_state_anatomy.png`
- `paper/npj_digital_medicine/figures/main/figure2_branch_delta_first.pdf`
- `paper/npj_digital_medicine/figures/main/figure2_branch_delta_first.svg`
- `paper/npj_digital_medicine/figures/main/figure2_branch_delta_first.png`
- `paper/npj_digital_medicine/figures/main/figure3_provider_condition_cres.pdf`
- `paper/npj_digital_medicine/figures/main/figure3_provider_condition_cres.svg`
- `paper/npj_digital_medicine/figures/main/figure3_provider_condition_cres.png`
- `paper/npj_digital_medicine/figures/main/figure4_temporal_stress_test.pdf`
- `paper/npj_digital_medicine/figures/main/figure4_temporal_stress_test.svg`
- `paper/npj_digital_medicine/figures/main/figure4_temporal_stress_test.png`
- `paper/npj_digital_medicine/figures/main/figure5_stroke_layer_mechanism.pdf`
- `paper/npj_digital_medicine/figures/main/figure5_stroke_layer_mechanism.svg`
- `paper/npj_digital_medicine/figures/main/figure5_stroke_layer_mechanism.png`
- `paper/npj_digital_medicine/figures/main/figure6_template_state_space_stratification.pdf`
- `paper/npj_digital_medicine/figures/main/figure6_template_state_space_stratification.svg`
- `paper/npj_digital_medicine/figures/main/figure6_template_state_space_stratification.png`
- `paper/npj_digital_medicine/figures/main/figure7_judge_validation_calibration.pdf`
- `paper/npj_digital_medicine/figures/main/figure7_judge_validation_calibration.svg`
- `paper/npj_digital_medicine/figures/main/figure7_judge_validation_calibration.png`
- `paper/npj_digital_medicine/timely_bench_npj_article.pdf`
- `paper/npj_digital_medicine/timely_bench_refs.bib`
- `/private/tmp/timely_bench_compile_audit_logs/timely_bench_npj_article.log`
- `/private/tmp/timely_bench_compile_audit_logs/timely_bench_npj_article.aux`
- `/private/tmp/timely_bench_compile_audit_logs/timely_bench_npj_article.bbl`
- `/private/tmp/timely_bench_compile_audit_logs/timely_bench_npj_article.blg`

## Commands run

- `rg --files TIMELY-Bench_Final/paper/npj_digital_medicine | rg 'timely_bench_npj_article\.tex|table[1-7].*\.tex|timely_bench_npj_article\.pdf|\.bib$|\.log$|\.aux$|\.bbl$|\.blg$'`
- `find TIMELY-Bench_Final/paper/npj_digital_medicine/figures/main -maxdepth 1 \( -name 'figure[1-7]_*.pdf' -o -name 'figure[1-7]_*.png' -o -name 'figure[1-7]_*.svg' \) | sort`
- `rg -n 'fig:trajectory_tier|Panel [ABCabc]|panel [ABCabc]|trajectory tier|Template support|template support|Figure 6' timely_bench_npj_article.tex`
- `rg -n 'Figure 2|Branch B1|B1|best available B1|better of|selected by AUROC|AUROC-selected|AUPRC|DEL-S1' timely_bench_npj_article.tex tables/table2.tex`
- `rg -n 'B2/B3 prompt|B2 prompt \+ state metadata|Panel d|unlabeled points|Gemini 3\.1(?! Pro)|Gemma4(?!-26B)|MedGemma(?!-4B)|Meditron(?!3-8B)|Layer1|Layer2|Exact match|Avg latency|Tokens' timely_bench_npj_article.tex tables/table*.tex`
- `tectonic --outdir /private/tmp/timely_bench_compile_audit timely_bench_npj_article.tex`
- `tectonic --keep-logs --keep-intermediates --outdir /private/tmp/timely_bench_compile_audit_logs timely_bench_npj_article.tex`
- `pdftotext` page checks for Figures 1, 3, 6, and 7.
- `pdftoppm` page rendering for pages 5, 9, 11, 14, 15, 16, 18 and selected table pages.
- `pdfinfo timely_bench_npj_article.pdf`
- Static Python check for labels, refs, cites, and BibTeX keys.
- `sips -g pixelWidth -g pixelHeight figures/main/figure*.png`
- `head -n 3 results/cres_v3/paper_tables/table3_provider_ranking.csv`
- `rg -n 'avg_latency_seconds|usage_total_tokens' code/v3 results/cres_v3/paper_tables/table3_provider_ranking.csv`
- `tectonic timely_bench_npj_article.tex`
- `tectonic --keep-logs --keep-intermediates --outdir /private/tmp/timely_bench_minimal_patch_logs timely_bench_npj_article.tex`
- `pdftotext timely_bench_npj_article.pdf /private/tmp/timely_bench_minimal_patch_text.txt`
- `pdftoppm -f 7 -l 10 -r 180 -png timely_bench_npj_article.pdf /private/tmp/timely_bench_minimal_patch_pages/p07_10`
- `pdftoppm -f 15 -l 17 -r 180 -png timely_bench_npj_article.pdf /private/tmp/timely_bench_minimal_patch_pages/p15_17`

## Minimal consistency patch applied

- Figure 6 panel reference fixed: YES. The template-support coverage reference now points to `Fig.~\ref{fig:trajectory_tier}, panel a`, and the row-level performance reference now points to `panel c`.
- Figure 2 comparator definition clarified: YES. Results text and Figure 2 caption now state that Branch B1 is represented by the AUROC-selected B1 model for each task, and both AUROC and AUPRC deltas use that same selected comparator relative to Branch A.
- Table 1B Layer spacing fixed: YES. `Layer1 temporal` and `Layer2 retrospective` were changed to `Layer 1 temporal` and `Layer 2 retrospective`.
- Table 5 1--5 scale caption added: YES. The caption now states that the judge values are mean rubric scores on a 1--5 scale.
- Table 6 Exact match rate header fixed: YES. The column header is now `Exact match rate`.
- Table 3 latency/token units clarified: YES. Source confirmation: `results/cres_v3/paper_tables/table3_provider_ranking.csv` uses `avg_latency_seconds` and `usage_total_tokens`; `code/v3/run_phase65f_frozen_eval_v1.py` aggregates the same fields. The table now uses `Avg latency (s)` and `Total tokens`.
- Submission placeholders still require author input: YES. Author block, affiliation, corresponding email, acknowledgements, and author contributions still require author input before submission.
- PDF recompiled successfully: YES. `tectonic timely_bench_npj_article.tex` updated `timely_bench_npj_article.pdf`; `tectonic --keep-logs --keep-intermediates --outdir /private/tmp/timely_bench_minimal_patch_logs timely_bench_npj_article.tex` produced logs for inspection.
- Undefined refs/cites status: PASS. No undefined citation/reference warnings or multiply defined labels were found in `/private/tmp/timely_bench_minimal_patch_logs/timely_bench_npj_article.log` or `.blg`; `.aux` contains only expected hyperref internal `\undefined` guard macros, not unresolved manuscript references.
- Remaining warnings: EXISTING / NON-BLOCKING. Tectonic still reports underfull/overfull box warnings, including existing table width warnings for `tables/table1b.tex`, `tables/table6.tex`, and `tables/table7.tex`, plus the known Tectonic rerun warning. Rendered patch-relevant pages showed the modified figure caption/body references and tables remained readable.
- Figure files modified: NO. No Figure 1--7 image/PDF/SVG assets were regenerated or edited during this patch.
- Rendered pages checked: YES. Pages 7, 9, 10, 15, 16, and 17 were rendered to `/private/tmp/timely_bench_minimal_patch_pages/` and inspected for the modified Table 1B wording, Figure 2 caption, Table 3 headers, Figure 6 panel references, and Tables 5--6 captions/headers.

## Author metadata and placeholder cleanup patch

Resolved:
- Placeholder Author removed: YES. The title-page author block now lists Haoyu Wang, Zitong Li, Linglong Qian, and Zina Ibrahim.
- Placeholder affiliation removed: YES. The affiliation now reads `King's College London, London, United Kingdom`.
- Placeholder email removed: YES. `corresponding.author@example.com` was removed from the TeX source.
- Four author names inserted: YES.
- Four author emails inserted: YES: `haoyu.7.wang@kcl.ac.uk`, `zitong.2.li@kcl.ac.uk`, `linglong.qian@kcl.ac.uk`, and `zina.ibrahim@kcl.ac.uk`.
- Equal contribution / shared first authorship statement added: YES. The author metadata now states that Haoyu Wang, Zitong Li, Linglong Qian, and Zina Ibrahim contributed equally.
- Acknowledgements placeholder replaced: YES. The acknowledgements now thank the MIMIC-IV and PhysioNet teams for maintaining the de-identified critical care resource.
- Author contributions placeholder replaced: YES. The author contribution statement now uses initials H.W., Z.L., L.Q., and Z.I. and states equal contribution.

Needs confirmation:
- Corresponding author assignment: NEEDS AUTHOR CONFIRMATION. Haoyu Wang is currently marked as the starred corresponding author because no unique corresponding author was specified.
- Data/code repository URLs or accession details: NEEDS AUTHOR CONFIRMATION. Current data and code availability language remains release-upon-publication wording without repository URLs or accessions.
- Funding statement: NEEDS AUTHOR CONFIRMATION. No funding source or grant number has been inserted because none was provided.

Still OK:
- Competing interests statement retained: YES. The manuscript still states that the authors declare no competing interests.
- Figure 1--7 files unchanged: YES. No figure assets were edited or regenerated in this patch.
- Result data unchanged: YES.
- Table data unchanged: YES. Only table wording/headers/caption text from the prior minimal consistency patch were adjusted; numeric table values were not changed.

Compile and title-page/end-matter QA:
- PDF recompiled after author metadata patch: YES. `tectonic timely_bench_npj_article.tex` updated `timely_bench_npj_article.pdf`; `tectonic --keep-logs --keep-intermediates --outdir /private/tmp/timely_bench_author_patch_logs timely_bench_npj_article.tex` produced logs for inspection.
- Title page checked for author names, emails, affiliation, and equal contribution statement: YES. Rendered page 1 shows Haoyu Wang, Zitong Li, Linglong Qian, and Zina Ibrahim; all four emails are present; affiliation appears as `King's College London, London, United Kingdom`; and the equal contribution statement appears in the author metadata.
- End matter checked for acknowledgements, author contributions, competing interests, and data/code availability wording: YES. Rendered pages 29--30 show the updated acknowledgements, updated author contributions, retained competing interests statement, and release-upon-publication data/code availability language.
- Placeholder grep after author metadata patch: PASS for manuscript source/PDF. `Placeholder`, `To be completed`, and `corresponding.author@example.com` are absent from `timely_bench_npj_article.tex`, table sources, and extracted manuscript PDF text.
- Compile warning status after author metadata patch: existing underfull/overfull box warnings and the known Tectonic rerun warning remain; no undefined citation/reference warnings or multiply defined labels were found in `/private/tmp/timely_bench_author_patch_logs/timely_bench_npj_article.log` or `.blg`.

## Clarity and reproducibility wording patch

Resolved:
- Abstract data source clarified: YES. The abstract now states that TIMELY Bench was developed using MIMIC-IV ICU stays.
- Multimodal terminology clarified: YES. The abstract and Methods now define multimodal as multimodal EHR evidence spanning structured measurements, medications, procedures, clinical notes, and condition-specific observations; it is not used to imply imaging or audio data.
- CRES full name added: YES. `Clinical Reasoning Evaluation Suite (CRES)` is now expanded at first manuscript use, in Methods, and in Table 7 caption.
- RQ labels clarified: YES. The three research questions in the Introduction are now explicitly marked as RQ1, RQ2, and RQ3.
- Medical-domain overclaim tightened: YES. The manuscript now states that general-purpose models led the aggregate ranking and leading condition positions while Tier 2 medical-domain models remained below Tier 1A/1B in the aggregate macro comparison; it no longer claims uniform dominance across every condition-level comparison.
- Tier definitions added: YES. Tier 1A, Tier 1B, and Tier 2 are now defined as analysis strata rather than performance-derived categories.
- Macro primary score aggregation clarified: YES. Methods now state that provider-level macro primary score is the equal-weight mean across the 20 supported task-dimension primary scores, not a row-weighted mean; condition-level and CRES-dimension summaries use the same equal-weight macro aggregation within subsets.
- Structured baseline method clarified: YES. Methods now state that Branch A used XGBoost on compact anchor-state summaries, Branch B1 used BiLSTM-attention and temporal-transformer models on hourly sequences, and all structured baselines used five-fold subject-level cross-validation.
- Figure 2 comparator policy reinforced: YES. Methods now reiterate that Figure 2 uses the AUROC-selected B1 model within each task as the shared comparator for both AUROC and AUPRC deltas.
- Provider inference/provenance wording clarified without unsupported uniform settings: YES. Methods now state that provider-specific decoding defaults, output-token limits, and repair-stage limits were not assumed identical; release manifests record model provenance, canonical row counts, token totals, latency summaries, and repair-chain status.
- Confidence terminology tightened: YES. The subsection and Discussion now use `confidence behavior` / `self-reported confidence behavior` rather than implying formal probability calibration.
- Table 1B task code glossary added: YES. The caption now distinguishes disease prefixes (`AKI`, `DEL`, `SEP`, `S`) from task-family letters (`T`, `S`, `R`).
- Table 3 pair wins definition added: YES. The caption now defines pair wins as within-tier supported task-dimension primary-score co-win counts, with ties counted for all co-winning providers.

Needs confirmation:
- LLM decoding settings: PARTIALLY DOCUMENTED. The manuscript intentionally does not claim a uniform temperature, top-p, or max-token setting because max-token limits and repair chains varied across providers and top-p was not confirmed as fixed.
- MIMIC-IV version/date details: NEEDS AUTHOR CONFIRMATION if the final submission requires exact MIMIC-IV version and extraction date beyond the existing PhysioNet/MIMIC-IV citation and access statement.
- Data/code repository URLs or accession details: NEEDS AUTHOR CONFIRMATION. Release-upon-publication language remains unchanged.

Still OK:
- Figure 1--7 files unchanged: YES. No figure assets were edited or regenerated in this patch.
- Result data unchanged: YES.
- Table data unchanged: YES. Only caption/wording changes were made to tables.

## Final recommendation

- Ready for patch? Patch applied.
- Needs author input? Yes, before submission, for corresponding author confirmation, data/code repository URLs or accessions, any funding statement, and optional exact MIMIC-IV version/extraction-date details.
- Safe to proceed after this patch? Yes, pending remaining author-confirmation items and final author approval.
