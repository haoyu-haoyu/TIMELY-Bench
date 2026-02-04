# TIMELY-Bench Benchmark Results Summary

Generated: 2026-01-04

---

## 1. Model Performance (Mortality Task)

| Rank | Model | CV AUROC | Test AUROC | Test AUPRC |
|------|-------|----------|------------|------------|
| 1 | **Full Feature Fusion (GB)** | - | **0.844** | 0.473 |
| 2 | BERT + Annotation (GB) | - | 0.840 | 0.471 |
| 3 | Enhanced GRU | 0.837 ± 0.004 | 0.831 | 0.468 |
| 4 | Temporal GRU | 0.834 ± 0.004 | 0.824 | 0.455 |
| 5 | XGBoost (Tabular) | 0.795 ± 0.007 | 0.804 | 0.409 |
| 6 | Early Fusion | 0.774 ± 0.006 | 0.779 | - |
| 7 | Text-only (XGBoost) | 0.743 ± 0.004 | 0.759 | 0.329 |

---

## 2. Enhanced Feature Ablation Study

| Feature Set | Description | LR AUROC | GB AUROC | # Features |
|-------------|-------------|----------|----------|------------|
| Timeseries | Vital signs only | 0.710 | 0.761 | 24 |
| + Annotation | + Pattern annotations | 0.747 | 0.787 | 28 |
| + BERT | + ClinicalBERT embeddings | 0.824 | 0.840 | 78 |
| + Concept | + NER medical concepts | 0.774 | 0.804 | 67 |
| **All** | Full feature fusion | **0.834** | **0.844** | 117 |

**Key Finding**: BERT embeddings provide the largest performance boost (+7.9% AUROC).

---

## 3. Disease-Stratified Analysis

| Condition | N Samples | Mortality | LR AUROC (5-fold) | GB AUROC (5-fold) |
|-----------|-----------|-----------|-------------------|-------------------|
| AKI | 57,263 | 14.5% | 0.810 ± 0.003 | **0.820 ± 0.002** |
| Sepsis | 34,152 | 18.2% | 0.793 ± 0.009 | **0.807 ± 0.006** |
| ARDS | 822 | 39.9% | 0.666 ± 0.046 | 0.676 ± 0.015 |

**Key Finding**: Model performance varies by disease; smaller cohorts (ARDS) show higher variance.

---

## 4. Prolonged LOS Task (New)

| Feature Set | Description | LR AUROC | GB AUROC | # Samples |
|-------------|-------------|----------|----------|-----------|
| Timeseries | Vital signs only | 0.645 | 0.689 | 74,819 |
| + Annotation | + Pattern annotations | 0.782 | 0.802 | 74,819 |
| + BERT | + ClinicalBERT embeddings | 0.829 | 0.841 | 74,819 |
| + Concept | + NER medical concepts | 0.796 | 0.815 | 74,819 |
| **All** | Full feature fusion | **0.833** | **0.844** | 74,819 |

**Task Definition**: LOS > 7 days | **Positive Rate**: 16.2%

### Task Comparison

| Task | Positive Rate | Best AUROC | Best Model |
|------|---------------|------------|------------|
| Mortality | 12.4% | 0.844 | Full Fusion (GB) |
| Prolonged LOS | 16.2% | 0.844 | Full Fusion (GB) |
| **30-Day Readmission** | 14.5% | **0.632** | Full Fusion (GB) |

> **Note**: Readmission prediction shows lower AUROC as expected - this is a challenging task since ICU data may not capture all factors influencing readmission.

---

## 5. Time Window Comparison

| Window | CV AUROC | Test AUROC |
|--------|----------|------------|
| ±6h | 0.775 ± 0.005 | 0.777 |
| ±12h | 0.805 ± 0.008 | 0.800 |
| **±24h** | 0.835 ± 0.006 | **0.833** |

---

## 5. Calibration Results (ECE / Hosmer-Lemeshow)

| Model | AUROC | ECE ↓ | HL p-value |
|-------|-------|-------|------------|
| EarlyFusion_XGBoost | 0.777 | 0.0067 | 0.0001 |
| Tabular_XGBoost | 0.763 | 0.0065 | 0.0382 |
| TextOnly_XGBoost | 0.621 | **0.0015** | 0.0002 |

---

## 6. Note Category Ablation

| Ablation | Test AUROC | Insight |
|----------|------------|---------|
| All Categories | 0.638 | Baseline with all notes |
| **Nursing Only** | **0.638** | Most informative single category |
| Radiology Only | 0.545 | Moderate contribution |
| Exclude Nursing | 0.545 | Performance drops significantly |

---

## 7. Key Findings

### 7.1 Feature Engineering Impact

1. **BERT embeddings are highly effective**: +7.9% AUROC improvement
2. **Pattern annotations add value**: +2.6% AUROC over baseline
3. **Full fusion is best**: Combining all features achieves 0.844 AUROC

### 7.2 Disease-Specific Insights

1. **AKI has best predictability**: AUROC 0.82 with largest cohort
2. **Sepsis shows strong performance**: AUROC 0.81 despite higher mortality variance
3. **ARDS is challenging**: Small cohort (822) limits model performance

### 7.3 Text Feature Contributions

1. **ClinicalBERT (768→50 dim)**: Most impactful single text feature
2. **NER medical concepts (40 dim)**: Moderate improvement
3. **Nursing notes dominate**: Other note types add minimal value

---

## 8. Reproducibility Checklist

- [x] Data preprocessing scripts
- [x] Model training scripts
- [x] ClinicalBERT embedding extraction
- [x] NER concept extraction
- [x] Disease-stratified training
- [x] Enhanced Episode structure (74,829 episodes)
- [x] Results CSV files
- [x] Data Card
- [x] Model Card

---

**TIMELY-Bench v2.0 - Complete Benchmark Suite with Enhanced Features**

