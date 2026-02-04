# MASTER AUDIT ADDENDUM

**审计附录 ID**: 20260128_addendum
**时间戳**: 2026-01-28T21:21:19 UTC
**项目路径**: /scratch/users/k25113331/TIMELY-Bench_Final

---

本附录补充 MASTER_AUDIT_REPORT (20260128_145841) 中的两项警告，
将审计总体判定从 **PASS_WITH_WARNINGS** 升级至 **PASS**。

---

## 补齐项 1: PROVENANCE.json

### 问题描述
原审计 Phase I (Delivery Compliance) 报告 PROVENANCE.json 缺失。

### 补齐措施
通过 `code/data_processing/generate_provenance.py` 自动生成 PROVENANCE.json，包含：

| 字段 | 值 |
|------|----|
| project_root | /cephfs/volumes/hpc_data_usr/k25113331/f7401904-95bd-4544-810d-29f0538f2c9c/TIMELY-Bench_Final |
| run_id | 20260128_211307 |
| python_version | 3.10.12 |
| requirements_hash | `5df1790817a06221...` |
| git_commit | None |
| baseline_inputs | alignment CSV, cohort_final.csv, episodes_enhanced |
| baseline_outputs | results_summary, permutation, late_fusion, QA gate |
| split_configuration | TEST_SIZE=0.2, N_FOLDS=5, RANDOM_STATE=42, GroupKFold |

**SHA256**: `65f5e252b11f0ae9e311ba17067c61a0d4e93f9ab2f336501643abcb6e2c4810`

### 验证
- [x] PROVENANCE.json 存在于 final_release/
- [x] 已纳入 manifest.json
- [x] Phase I 现在通过

---

## 补齐项 2: Subject Leakage 全量证据

### 问题描述
原审计 Phase H 仅采样 1000 episodes 进行泄漏检查。
需要全量 (full-population) 证据来确认零泄漏。

### 全量审计结果

**脚本**: `code/data_processing/audit_subject_leakage_full.py`
**覆盖范围**: 全部 74829 stays / 54551 unique subjects

#### Subject-Level Leakage (全量)

| 任务 | 有效样本 | Holdout交集 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|------|---------|------------|--------|--------|--------|--------|--------|
| mortality | 74829 | 0 | 0 | 0 | 0 | 0 | 0 |
| prolonged_los | 74829 | 0 | 0 | 0 | 0 | 0 | 0 |
| readmission_30d | 65899 | 0 | 0 | 0 | 0 | 0 | 0 |

**Global max intersection**: **0** (全部为零)
**Verdict**: **PASS**

#### Subject Multiplicity 分布

| 入院次数 | 患者数 |
|----------|--------|
| 1 | 42592 |
| 2 | 7896 |
| 3 | 2329 |
| 4 | 857 |
| 5 | 385 |
| 6 | 168 |
| 7 | 111 |
| 8 | 59 |
| 9 | 43 |
| 10 | 39 |
| 11 | 15 |
| 12 | 13 |
| 13 | 11 |
| 14 | 7 |
| 15 | 7 |
| 16 | 3 |
| 17 | 1 |
| 18 | 2 |
| 19 | 3 |
| 20 | 2 |
| 23 | 3 |
| 24 | 2 |
| 25 | 1 |
| 27 | 1 |
| 37 | 1 |

- 单次入院: 42592 (78.1%)
- 多次入院 (2+): 11959 (21.9%)
- 最大入院次数: 37

**结论**: GroupShuffleSplit + GroupKFold 以 subject_id 为 group 进行分组，
确保多次入院患者的所有 stays 落在同一分区，避免数据泄漏。
全量验证已确认所有 3 个任务 × 6 个分区组合的 subject 交集均为空集。

---

## 补齐项 3: Alignment 行数一致性核对

- **temporal_textual_alignment.csv**: 6,974,406 数据行
- **cohort_final.csv**: 74829 stays
- **episodes_enhanced/**: 74829 JSON 文件
- **一致性**: alignment 文件包含每个 stay 的多行时间序列记录，
  行数 (6,974,406) >> stay 数 (74829)，符合预期

---

## 升级判定

| 原始判定 | 补齐后判定 | 变更原因 |
|----------|-----------|---------|
| PASS_WITH_WARNINGS | **PASS** | PROVENANCE 补齐 + 全量泄漏证据 |

### 新增产物清单

```
final_release/
├── PROVENANCE.json                      (新增)
├── RELEASE_AUDIT_REPORT.md              (新增)
├── MASTER_AUDIT_ADDENDUM.md             (新增)
├── manifest.json                        (已更新)
└── evidence/
    ├── subject_leakage_full.json        (新增)
    └── subject_multiplicity.json        (新增)
```

---

*附录由 Claude Code 于 2026-01-28 生成*


---

## 补齐项 4: 路径口径统一 (Path Normalization)

**问题**: 审计产物中 `project_root` 出现两套路径——
`/scratch/users/k25113331/TIMELY-Bench_Final`（用户工作路径）和
`/cephfs/volumes/hpc_data_usr/k25113331/.../TIMELY-Bench_Final`（CephFS 挂载原始路径）。

**探测结果**: 两个路径的 `device:inode` 完全一致 (`52:9895663149215`)，
`manifest.json` 的 SHA256 相同 (`f50583f790e7f2db...`)。
`/scratch` 是 CephFS 的用户态符号链接/挂载点。

**处理**: 规范路径统一为 `/scratch/users/k25113331/TIMELY-Bench_Final`，
`/cephfs` 路径作为 `path_aliases` 记录在 `PROVENANCE.json` 中。
证据文件: `final_release/evidence/path_mapping_check.json`

---

## 补齐项 5: 代码树可复现凭证 (Code Tree Hash)

**问题**: `git_commit` 为 null（HPC scratch 无 git 仓库），缺少代码版本锚定。

**方案**: 对 `code/` 目录做确定性哈希：遍历全部源文件（排除 `__pycache__`/`*.pyc`），
按相对路径排序后逐文件 SHA256，再对拼接结果整体 SHA256。

- **code_tree_hash**: `3eb5d44bb72dc29d308318049ea8708dad4cb84d40a8ea4f4b250ff1f8998677`
- **文件数**: 111
- **scope**: 仅 `code/` 目录，不含 data/results/final_release/logs

即使无 git commit，`PROVENANCE.json` + `code_tree_hash` 可唯一定位代码版本。
证据文件: `final_release/evidence/code_tree_hash.json`

---

## 补齐项 6: Split 口径精确化 (Per-Task Splitter Specification)

**问题**: 原附录笼统记录 `split_configuration = GroupKFold`，
未区分各任务/管线的实际 splitter。

**精确化结果**:

| 任务 | Holdout Splitter | CV Splitter | Stratified? | groups |
|------|-----------------|-------------|-------------|--------|
| mortality | GroupShuffleSplit(test_size=0.2, rs=42) | GroupKFold(n_splits=5) | No | subject_id |
| prolonged_los | GroupShuffleSplit(test_size=0.2, rs=42) | GroupKFold(n_splits=5) | No | subject_id |
| readmission_30d | *无独立 holdout* | StratifiedGroupKFold(n_splits=5, shuffle=True, rs=42) | **Yes** | subject_id |

**readmission 说明**:
- `train_readmission_baselines.py` 使用 try/except 模式优先导入 `StratifiedGroupKFold`
- 当前环境 sklearn 1.7.2 支持，`HAS_STRATIFIED_GROUP_KFOLD=True`
- 实际训练走 `StratifiedGroupKFold` 分支（分层 + 分组），**非** `GroupKFold` 降级分支
- 训练脚本仅做 CV 评估，不使用 `GroupShuffleSplit` 做 holdout
- 泄漏审计中仍补充了 `GroupShuffleSplit` holdout 检查作为保守验证

**全量泄漏审计已使用真实 splitter 重新验证**，所有 intersection = 0。
证据文件: `final_release/evidence/subject_leakage_full.json`

---

*补强审计于 2026-01-28 21:36 UTC 完成 by Claude Code*
