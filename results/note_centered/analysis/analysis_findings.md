# Phase 5 Analysis Findings

## Q1. Window Effect
- Mortality structured XGBoost shows monotonic lookback behavior: W6=0.8863 < W12=0.8960 < W24=0.9042.
- Mortality D0 remains stronger than W6: D0=0.9007 vs W6=0.8863.
- LOS structured XGBoost is flat across lookback windows and best at D0: D0=0.8972, W6=0.8802, W12=0.8814, W24=0.8817.
- LOS text-only has anti-monotonic behavior: mean W6=0.8431 > W24=0.8355 (delta=-0.0076); typed W6=0.8281 > W24=0.8230 (delta=-0.0051).

Interpretation: mortality benefits from longer context, while LOS appears trajectory-type dominated with weaker gain from longer lookback.

## Q2. Leakage Decomposition
- Mortality: A=0.9232, B=0.9231, C=0.9079, D=0.9079.
  premium_total=+0.0154, premium_struct=+0.0153 (99.3%), premium_text=-0.0000, interaction=+0.0001.
- Prolonged LOS: A=0.9368, B=0.9370, C=0.8856, D=0.8860.
  premium_total=+0.0508, premium_struct=+0.0510 (100.5%), premium_text=-0.0004, interaction=+0.0002.

Interpretation: leakage premium is dominated by structural leakage; text AFTER leakage is approximately zero with note-level ClinicalBERT pooling.

## Q3. Note-Type Contribution
- No text baseline (tabular only): 0.9042
- Nursing only: 0.9079
- Radiology only: 0.9018
- Lab only: 0.9037
- All typed: 0.9073
- All mean: 0.9079

Interpretation: text adds only marginal AUROC over the strong 42-feature structured baseline.

## Q4. Typed vs Mean Pooling
- Mortality text-only clean: mean=0.8501, typed=0.8388, delta=-0.0113.
- Mortality late fusion clean: mean=0.8952, typed=0.8947, delta=-0.0005.
- Mortality all-notes ablation: mean=0.9079, typed=0.9073, delta=-0.0006.

Interpretation: mean pooling is equal or better than typed pooling in this setup, suggesting dimensionality overhead without measurable gain.

## Q5. LOS vs Mortality Comparison
- Clean early fusion AUROC: mortality=0.9079, prolonged_los=0.8860.
- Mortality text marginal gain (all mean vs no text): +0.0037.

Interpretation: text contribution remains small for both tasks once 42 structured variables are available.

## Q6. Truncation Impact (W24)
Source: `truncation_analysis.csv`

bucket  n_notes  pct_notes  mean_feature_completeness
  0-6h  1833143  15.268899                   0.349141
 6-12h  1800434  14.996455                   0.347222
12-18h  1710805  14.249903                   0.345544
18-24h  1666423  13.880229                   0.342888
24-48h  4994926  41.604514                   0.336731

Interpretation: feature completeness varies only mildly across chart-hour buckets, indicating truncation does not induce severe quality collapse for note-window features.

## Q7. D0 Boundary Effect
Source: `d0_boundary_analysis.csv`

bucket  n_notes  pct_notes  mean_feature_completeness  mean_n_measurements
  0-2h  1155364   9.623437                   0.351737             0.545607
  2-6h  2154546  17.945979                   0.339010             1.521055
 6-12h  3118749  25.977169                   0.340944             3.230266
12-24h  5577072  46.453415                   0.339164             6.232492

Interpretation: severe D0 truncation (<2h) accounts for 9.62% of notes in this calculation and does not prevent D0 from being competitive (or best on LOS structured baselines).

## Q8. Cross-Task Leakage Decomposition (Progression Tasks)
- AKI progression (24h lookahead): A=0.9176, B=0.9172, C=0.8709, D=0.8714; premium_total=+0.0463, premium_struct=+0.0459, premium_text(C-D)=-0.0004.
- Sepsis to shock (12h lookahead): A=0.9845, B=0.9844, C=0.9446, D=0.9446; premium_total=+0.0399, premium_struct=+0.0399, premium_text(C-D)=+0.0000.

Key finding: contrary to the initial hypothesis, text leakage premium (C-D) remains negligible across all four tasks (mortality, prolonged LOS, AKI progression, sepsis-shock).

Mechanistic interpretation: the near-zero text leakage across task types is consistent with note-level ClinicalBERT pooling behavior rather than task acuity. Future-note content is diluted when embeddings are mean-pooled across all notes in the window, making leakage signal inaccessible to downstream classifiers.

Implication for benchmark design: structural leakage remains the dominant and consistent source across tasks (approximately all measurable leakage premium). To test task-dependent text leakage effects, future work should evaluate sentence-level or span-level representations instead of stay-level pooled document embeddings.
