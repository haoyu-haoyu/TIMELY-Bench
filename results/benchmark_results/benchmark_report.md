# Benchmark Results

## Overview

Experiments across:
- Time Windows: 6h, 12h, 24h, D0
- Tasks: Mortality, Prolonged LOS
- Cohorts: All, Sepsis, AKI
- Models: Logistic Regression, XGBoost

## Results


### Mortality

| Cohort | Model | 6h | 12h | 24h | D0 |
|--------|-------|-----|-----|-----|-----|
| all | LogisticRegression | 0.7833 | 0.8177 | 0.8517 | 0.7969 |
| all | XGBoost | 0.8052 | 0.8385 | 0.8679 | 0.8111 |
| sepsis | LogisticRegression | 0.7427 | 0.7749 | 0.8095 | 0.7551 |
| sepsis | XGBoost | 0.7491 | 0.7871 | 0.8207 | 0.7624 |
| aki | LogisticRegression | 0.7686 | 0.8002 | 0.8343 | 0.7802 |
| aki | XGBoost | 0.7831 | 0.8188 | 0.8489 | 0.7916 |

### Prolonged Los

| Cohort | Model | 6h | 12h | 24h | D0 |
|--------|-------|-----|-----|-----|-----|
| all | LogisticRegression | 0.7047 | 0.7462 | 0.7888 | 0.7311 |
| all | XGBoost | 0.7256 | 0.7620 | 0.8059 | 0.7432 |
| sepsis | LogisticRegression | 0.6757 | 0.7141 | 0.7568 | 0.6983 |
| sepsis | XGBoost | 0.6971 | 0.7336 | 0.7794 | 0.7134 |
| aki | LogisticRegression | 0.6839 | 0.7236 | 0.7653 | 0.7081 |
| aki | XGBoost | 0.7034 | 0.7408 | 0.7833 | 0.7216 |
