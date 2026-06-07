# TIMELY-Bench

**基于 MIMIC-IV ICU 轨迹与临床文本的 anchor-bounded 临床时序推理基准。**

[English](README.md) | 中文

## 仓库范围

这个公开 GitHub 仓库包含两条复现主线：

| 主线 | 目的 | 主要产物 |
|---|---|---|
| **V2 note-centered leakage experiments** | 早期时间-文本对齐与 time-leakage 分解实验，覆盖 mortality 和 prolonged ICU LOS。 | `results/note_centered/`, `code/baselines/`, `code/analysis/` |
| **V3 TIMELY-Bench / CRES evaluation** | 论文主结果：4 个临床条件、168 小时 anchor-bounded 轨迹、结构化基线、9 个 frozen LLM provider、LLM-as-Judge。 | `results/v3/`, `results/cres_v3/`, `code/v3/`, `paper/npj_digital_medicine/` |

论文主结果主要来自 **V3 TIMELY-Bench / CRES evaluation**。V2 结果保留用于复现 time-leakage 实验和解释 V3 设计动机。

公开仓库不包含 raw MIMIC-IV 表、patient-level 派生文件、prompt JSONL、canonical response JSONL、per-instance scoring table 或 judge long-form rationale。详见 [DATA_ACCESS.md](DATA_ACCESS.md)、[PUBLIC_ARTIFACT_POLICY.md](PUBLIC_ARTIFACT_POLICY.md) 和 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## V3 Benchmark 快照

| 项目 | 数值 |
|---|---:|
| 数据来源 | MIMIC-IV ICU |
| source alignment ICU stays | 74,829 |
| 时间网格 | 168 小时 |
| 临床条件 | AKI, delirium, sepsis, stroke |
| Prompt instances per LLM provider | 53,070 |
| Frozen comparative LLM providers | 9 |
| Structured baseline tasks | eligible binary CRES tasks |

关键 frozen 结果：

- `results/cres_v3/phase65f_frozen_eval/phase65f_provider_metrics.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_per_task_dimension_metrics.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_condition_heatmap_data.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_stratified_metrics.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_temporal_degradation.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_formal_summary.md`

论文文件：

- `paper/npj_digital_medicine/timely_bench_npj_article.tex`
- `paper/npj_digital_medicine/timely_bench_npj_article.pdf`

## V2 Note-Centered 快照

| 项目 | 数值 |
|---|---:|
| ICU stays | 74,829 |
| 结构化特征数 | 42 |
| 时序范围 | 入 ICU 后 0-72h |
| 文本范围 | 入 ICU 后 0-48h |
| 总笔记数 | 12,005,731 |
| 任务 | `mortality`, `prolonged_los` |
| 窗口 | `D0`, `W6`, `W12`, `W24`, `leaked`, `clean` |

关键文件：

- `results/note_centered/leakage_premium_decomposition.csv`
- `results/note_centered/progression_tasks/cross_task_leakage_decomposition.csv`
- `results/note_centered/tables/`
- `results/note_centered/figures/`

## 快速开始

```bash
git clone https://github.com/haoyu-haoyu/TIMELY-Bench.git
cd TIMELY-Bench
python -m pip install -r requirements.txt
```

### 查看公开 aggregate 结果

公开结果和论文文件已经在 `results/` 和 `paper/` 下。查看这些 aggregate metrics 不需要 MIMIC-IV 权限。

### 重新生成 V2 轻量表格和图

```bash
python code/analysis/generate_core_tables.py
python code/analysis/compare_old_vs_new.py
python code/analysis/answer_analysis_questions.py
MPLBACKEND=Agg python code/analysis/generate_figures.py
```

### 从受控数据重建 V3/CRES

完整 V3 重建需要 PhysioNet credentialed MIMIC-IV access 和受控 patient-level 派生文件。CREATE/HPC 入口脚本为：

```bash
export PROJECT_ROOT=/path/to/TIMELY-Bench
export RESULTS_ROOT=${PROJECT_ROOT}/results/cres_v3

bash scripts/run_v3_full_source_refresh_create.sh
bash scripts/run_v3_create_pipeline.sh
sbatch scripts/run_phase6_cres_assembly_v3.sbatch
bash scripts/run_phase65f_frozen_eval_create.sh
```

这些 Slurm/CREATE 脚本是模板。迁移到其他集群时，需要设置：

- `PROJECT_ROOT`
- `RESULTS_ROOT`
- `VENV`
- `HF_HOME`
- provider-specific API key 或 env file

## Canonical 入口

| 目的 | 入口 |
|---|---|
| V2 aggregate 表格生成 | `code/analysis/generate_core_tables.py` |
| V2 leakage decomposition | `code/analysis/progression_leakage_analysis.py` |
| V3 source refresh | `scripts/run_v3_full_source_refresh_create.sh` |
| V3 state/representation build | `scripts/run_v3_create_pipeline.sh` |
| V3 CRES assembly | `scripts/run_phase6_cres_assembly_v3.sbatch` |
| V3 frozen scoring / judge packet | `scripts/run_phase65f_frozen_eval_create.sh` |

`code/v3/` 中部分 `pilot`、`repair`、`probe` 文件只用于 provenance，不是推荐的公开复现入口。详见 [code/v3/README.md](code/v3/README.md)。

## 数据访问

原始 MIMIC-IV 数据需通过 PhysioNet credentialed data access 获取。公开 GitHub 不包含 raw tables、note text、prompt JSONL、patient-level files、canonical response JSONL 或 judge rationales。受控复现边界详见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
