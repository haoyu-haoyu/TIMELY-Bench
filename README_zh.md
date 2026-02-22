# TIMELY-Bench

用于 ICU 多模态（结构化时序 + 临床笔记）时间对齐与融合的基准数据与评估框架。

[English](README.md) | 中文

## 当前状态

| 指标 | 数值 |
|------|------|
| Episodes（ICU stays） | **74,829** |
| 患者数（约） | ~50,000 |
| 时间窗口 | 6h, 12h, 24h（含 D0） |
| 生理特征（结构化） | 25 |
| 文本表示 | 标注统计特征、ClinicalBERT embedding（可选 MedCAT concepts） |

最近更新：2026 年 2 月 | 版本：2.0 Final

## 项目做什么

TIMELY-Bench 关注一个核心问题：临床笔记与结构化时序信号如何在时间上对齐，并在统一协议下公平比较不同融合策略（early/late）与不同时间窗口（6h/12h/24h + D0）。

它包含：
- 多窗口（6h/12h/24h + D0）结构化特征提取与可复现实验划分
- Episode JSON（结构化时序 + 对齐后的笔记 + pattern 检测 + 推理/标注统计特征）
- 条件图（Condition Graph）与典型时间演变模板（Physiology Templates）
- 轻量级基线（structured / text-only / early fusion / late fusion / temporal）
- 校准与跨窗口鲁棒性评估（含统计检验）

## 关键结果（24h，All cohort）

### 院内死亡（Mortality）

| 模型 | AUROC | AUPRC | ECE | Brier |
|------|-------|-------|-----|-------|
| Early Fusion XGBoost（Structured + ClinicalBERT embedding） | **0.885** | **0.584** | 0.0086 | 0.0740 |
| Late Fusion（tuned $\alpha$，ClinicalBERT） | 0.881 | 0.551 | 0.1078 | 0.0915 |
| Early Fusion XGBoost（Structured + 标注统计特征） | 0.873 | 0.557 | 0.0066 | 0.0770 |
| Late Fusion（tuned $\alpha$，标注统计特征） | 0.869 | 0.535 | 0.1813 | 0.1234 |
| XGBoost（Structured） | 0.868 | 0.541 | 0.1974 | 0.1327 |
| Logistic Regression（Structured） | 0.848 | 0.508 | 0.0083 | 0.0823 |
| Clinical GRU（Temporal） | 0.842 | 0.483 | 0.0336 | 0.0871 |
| Logistic Regression（Text-Only，ClinicalBERT embedding） | 0.832 | 0.444 | --- | --- |
| XGBoost（Text-Only，ClinicalBERT embedding） | 0.817 | 0.444 | 0.0089 | 0.0881 |
| XGBoost（Text-Only，标注统计特征） | 0.755 | 0.327 | 0.0062 | 0.0965 |
| Logistic Regression（Text-Only，MedCAT concepts） | 0.552 | 0.150 | --- | --- |
| XGBoost（Text-Only，MedCAT concepts） | 0.552 | 0.151 | --- | --- |

### 延长 ICU 住院（Prolonged LOS）

| 模型 | AUROC | AUPRC |
|------|-------|-------|
| Early Fusion XGBoost（Structured + ClinicalBERT embedding） | **0.835** | **0.509** |
| Late Fusion（tuned $\alpha$，ClinicalBERT） | 0.834 | 0.506 |
| Early Fusion XGBoost（Structured + 标注统计特征） | 0.818 | 0.468 |
| XGBoost（Structured） | 0.815 | 0.460 |
| Logistic Regression（Structured） | 0.797 | 0.422 |
| Late Fusion（tuned $\alpha$，标注统计特征） | 0.815 | 0.458 |
| XGBoost（Text-Only，ClinicalBERT embedding） | 0.800 | 0.456 |
| Logistic Regression（Text-Only，ClinicalBERT embedding） | 0.800 | 0.452 |
| XGBoost（Text-Only，标注统计特征） | 0.701 | 0.311 |
| Logistic Regression（Text-Only，MedCAT concepts） | 0.549 | 0.192 |
| XGBoost（Text-Only，MedCAT concepts） | 0.550 | 0.195 |

### 跨窗口鲁棒性（Cross-window Robustness）

Mortality AUROC（结构化基线，All cohort）：

| 模型 | 6h | 12h | 24h | CV (%) |
|------|----|-----|-----|--------|
| XGBoost | 0.805 | 0.839 | 0.868 | 3.05 |
| Logistic Regression | 0.783 | 0.818 | 0.852 | 3.13 |

统计检验：Friedman $\chi^2$=12.0，p=0.0025；两两 Wilcoxon 检验 p=0.0313（每组对比）。

## `final_release/` 中包含的产物

- `final_release/` 是一个轻量、可校验（checksummed）的交付包，包含关键 artefacts（图谱、模板、QC、CRES、证据等）。完整的 episode JSON 位于 `episodes/episodes_enhanced/`，为避免体积过大不在 `final_release/` 内重复打包。
- `condition_graphs/`：Sepsis/SIRS、AKI/KDIGO、Delirium/ICU、Stroke/Neuro 的 guideline-anchored 条件图（节点带 domain tag，如 `lab_marker`/`vital_sign`/`symptom`/`medication`/`multimorbidity`）
- `physiology_templates/`：典型时间演变模板（canonical trajectories / physiology templates）
- `llm_annotations/`：用于质检与评估的标注子集（例如约 900 条）
- `evidence/`, `qc/`, `cres/`：可复现实验与评估支架

## 重要名词说明（避免混淆）

- `Early Fusion (AnnotFeatures)`：structured 聚合特征与 annotation-derived 文本特征拼接后，训练单一表格模型（见 `results/fusion_baselines/`）。
- `Early Fusion (ClinicalBERT)`：structured 聚合特征与 stay-level ClinicalBERT 向量拼接后训练。
- `EarlyFusion_XGBoost`（部分 robustness/calibration 脚本中的命名）：历史命名下的结构化 XGBoost 基线，不代表 multimodal fusion。

## 目录结构（简要）

```
TIMELY-Bench_Final/
├── code/
│   ├── baselines/                  # 基线训练脚本
│   ├── evaluation/                 # 评估脚本（校准/鲁棒性/统计检验）
│   └── config.py
├── data/
│   └── processed/
│       ├── data_windows/           # 6h/12h/24h + D0 结构化特征
│       └── merge_output/
│           └── cohort_final.csv
├── final_release/                  # 可交付数据包
├── results/                        # 输出（standardized/robustness/calibration/...）
└── docs/                           # Data card / Model card / checklist
```

## 快速开始（本地或 CREATE）

```bash
cd TIMELY-Bench_Final

# Structured-only baselines
python code/baselines/train_tabular_baselines.py

# Text-only baselines（标注统计特征）
python code/baselines/train_text_only.py

# Text-only baselines（ClinicalBERT embedding）
python code/baselines/train_text_only_embeddings.py

# Fusion baselines（early concat + late weighted）
python code/baselines/train_fusion.py
```

评估：

```bash
python code/evaluation/run_calibration_evaluation.py
python code/evaluation/update_robustness_final.py
```

## 文档

- 数据卡：`docs/DATA_CARD.md`
- 对齐协议卡：`docs/ALIGNMENT_PROTOCOL_CARD.md`
- 模型卡：`docs/MODEL_CARD.md`
- 结果汇总：`results/standardized/results_summary.csv`
