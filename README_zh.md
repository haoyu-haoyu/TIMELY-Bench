# TIMELY-Bench

面向 ICU 多模态预测（结构化时序 + 临床文本）的时间对齐基准。

[English](README.md) | 中文

## 当前主线（v2.0 Note-Centered）

| 项目 | 数值 |
|---|---|
| ICU stays | 74,829 |
| 结构化特征数 | 42 |
| 主任务 | 院内死亡、延长住院 |
| 进展任务 | AKI Stage1→2+、脓毒症→感染性休克 |
| 实验总数 | 113（91 + 22） |
| 对齐窗口 | D0、W6、W12、W24、leaked、clean |

## 仓库包含内容

- 以笔记为中心（note-centered）的多模态对齐与训练流程。
- 2x2 泄漏分解（结构化泄漏 vs 文本泄漏）。
- 规范化结果目录：
  - 91 个主线实验：`results/note_centered/core_experiments/`
  - 22 个进展任务实验：`results/note_centered/progression_tasks/`
- 对应分析表格、图和阶段报告。

## 范围说明

- `results/note_centered/` 是当前 canonical note-centered benchmark 的正式结果目录。
- `results/cres/` 是补充性的临床推理评测轨道，不计入 113 个 canonical benchmark 实验。
- `results/llm_annotations/` 保存文本标注资产及其专属审计文件。
- `results/audit/` 是项目级 release audit 目录，可能同时引用主线 benchmark 和补充轨道。

## 当前核心发现

- 总泄漏溢价（A-D）：
  - Mortality：+0.0154
  - Prolonged LOS：+0.0508
- 结构化泄漏贡献约 99%+（跨任务一致）。
- 文本泄漏（note-level ClinicalBERT pooling）约为 0。
- 干净条件（Cell D）早融合 AUROC：
  - Mortality：0.9079
  - Prolonged LOS：0.8860
  - AKI progression：0.8714
  - Sepsis→Shock：0.9446

## 目录（主线）

```text
TIMELY-Bench_Final/
├── code/
│   ├── baselines/
│   │   ├── note_centered_common.py
│   │   ├── run_baselines.py
│   │   ├── run_single_experiment.sh
│   │   ├── train_fusion.py
│   │   └── train_progression_baselines.py
│   ├── data_processing/
│   ├── evaluation/
│   └── analysis/
├── data/
├── results/
│   ├── note_centered/
│   │   ├── core_experiments/
│   │   ├── progression_tasks/
│   │   ├── comparisons/
│   │   ├── tables/
│   │   ├── figures/
│   │   └── analysis/
│   ├── cres/
│   ├── llm_annotations/
│   └── audit/
├── docs/
└── archive/legacy_consolidated/
```

## 复现实验示例

```bash
cd TIMELY-Bench_Final

# 单个主线实验
bash code/baselines/run_single_experiment.sh early_fusion xgb mortality W24 original results/note_centered

# 进展任务（AKI / Sepsis-Shock）
python code/baselines/train_progression_baselines.py --task aki_progression --condition all
python code/baselines/train_progression_baselines.py --task sepsis_shock --condition all
```

## 文档

- 数据卡：`docs/DATA_CARD.md`
- 对齐协议卡：`docs/ALIGNMENT_PROTOCOL_CARD.md`
- 模型卡：`docs/MODEL_CARD.md`
- canonical 范围说明：`CANONICAL_SCOPE.md`
- 阶段报告 / release audit：`results/audit/`

## 旧版内容说明

旧 admission-anchored / medcat / readmission / long-horizon 轨道已归档到：

`archive/legacy_consolidated/root_cleanup/`

## 许可

本项目使用 MIMIC-IV，需 PhysioNet 凭证访问权限。
