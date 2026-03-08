# TIMELY-Bench

**用于 ICU 多模态（结构化时序 + 临床文本）预测的时间对齐基准（MIMIC-IV）**

[English](README.md) | 中文

## 发布快照（v2.0 Note-Centered）

| 项目 | 数值 |
|---|---|
| ICU stays | **74,829** |
| 结构化特征数 | **42** |
| 时序提取范围 | 入 ICU 后 **0-72h** |
| 文本提取范围 | 入 ICU 后 **0-48h** |
| 总笔记数 | **12,005,731** |
| 任务 | `mortality`、`prolonged_los` |
| 窗口 | `D0`、`W6`、`W12`、`W24`、`leaked`、`clean` |
| Canonical 核心实验 | **91 个 JSON** |

**最近更新：** 2026 年 3 月

## 相比旧版流程的更新

- 时间对齐从 admission-anchored 改为 **note-centered lookback**。
- 结构化特征从 **25 扩展到 42**。
- 增加了显式泄漏控制和 2x2 分解（`leaked` vs `clean`）。
- 新版 canonical 结果统一放在 `results/note_centered/`。

## 关键结果（Phase 4 修复后，canonical 91）

### 1）Structured-only 基线（AUROC）

| 任务 | 模型 | D0 | W6 | W12 | W24 |
|---|---:|---:|---:|---:|---:|
| mortality | LR | 0.8775 | 0.8663 | 0.8758 | 0.8839 |
| mortality | XGBoost | 0.9007 | 0.8863 | 0.8960 | 0.9042 |
| prolonged_los | LR | 0.8858 | 0.8641 | 0.8619 | 0.8646 |
| prolonged_los | XGBoost | 0.8972 | 0.8802 | 0.8814 | 0.8817 |

### 2）2x2 泄漏分解（Early Fusion XGBoost）

| 任务 | A 全泄漏 | B 仅结构化泄漏 | C 仅文本泄漏 | D clean |
|---|---:|---:|---:|---:|
| mortality | 0.9232 | 0.9231 | 0.9079 | 0.9079 |
| prolonged_los | 0.9368 | 0.9370 | 0.8856 | 0.8860 |

泄漏溢价（Leakage Premium）总结：
- Mortality：`A-D = +0.0154`
- Prolonged LOS：`A-D = +0.0508`
- 结构化泄漏贡献占绝对主导（约 99%-100%），在当前 note-level ClinicalBERT 设定下文本泄漏近似为 0。

### 3）Text-only 基线（AUROC）

| 任务 | 文本类型 | W24 | leaked | clean |
|---|---|---:|---:|---:|
| mortality | mean | 0.8502 | 0.8502 | 0.8501 |
| mortality | typed | 0.8390 | 0.8390 | 0.8388 |
| prolonged_los | mean | 0.8355 | 0.8355 | 0.8356 |
| prolonged_los | typed | 0.8230 | 0.8230 | 0.8234 |

### 4）Note-type 消融（mortality, W24, early fusion XGBoost）

| 条件 | AUROC | 相对 tabular 增量 |
|---|---:|---:|
| No text (tabular only) | 0.9042 | - |
| Nursing only | 0.9079 | +0.0037 |
| Radiology only | 0.9018 | -0.0024 |
| Lab only | 0.9037 | -0.0005 |
| All notes (typed pool) | 0.9073 | +0.0031 |
| All notes (mean pool) | 0.9079 | +0.0036 |

## 产物路径

- 核心结果：`results/note_centered/core_experiments/`
- 表格：`results/note_centered/tables/`
- 图：`results/note_centered/figures/`
- 分析结论：`results/note_centered/analysis/analysis_findings.md`
- 文档卡片：`docs/DATA_CARD.md`、`docs/MODEL_CARD.md`、`docs/ALIGNMENT_PROTOCOL_CARD.md`

## 快速开始

### 环境

```bash
conda create -n timely python=3.10
conda activate timely
pip install torch numpy pandas scikit-learn xgboost matplotlib seaborn scipy tqdm
```

### 训练基线

```bash
cd TIMELY-Bench_Final
python code/baselines/train_tabular_baselines.py
python code/baselines/train_text_only.py
python code/baselines/train_text_only_embeddings.py
python code/baselines/train_fusion.py
```

### 生成 Phase 5 分析产物

```bash
python code/analysis/generate_core_tables.py
python code/analysis/compare_old_vs_new.py
python code/analysis/answer_analysis_questions.py
MPLBACKEND=Agg python code/analysis/generate_figures.py
```

## 任务定义

| 任务 | 定义 |
|---|---|
| In-hospital mortality | 住院期间死亡 |
| Prolonged LOS | ICU 住院时长 > 7 天 |

## 许可与数据访问

本项目使用 MIMIC-IV 数据。原始数据提取需 PhysioNet credentialed access。
