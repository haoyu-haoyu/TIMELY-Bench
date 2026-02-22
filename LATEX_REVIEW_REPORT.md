# TIMELY-Bench 论文/报告审查报告

**日期**: 2026-02-04
**审查范围**: 所有报告文件 vs 最新实验数据
**结论**: ⚠️ 需要更新 — 报告数据有多处不一致

---

## 0. 文件格式说明

项目中 **没有 LaTeX (.tex) 文件**。所有论文/报告以 Markdown 格式存储：

| 文件 | 角色 | 行数 |
|------|------|------|
| `report/survey_draft.md` | D4 期刊风格论文草稿 | 241行 |
| `report/FINAL_REPORT.md` | 附录补充报告 (MedCAT) | 81行 |
| `report/POSTER.md` | Poster内容草稿 | 32行 |
| `documentation/FINAL_PROJECT_REPORT.md` | 完整项目报告 | 253行 |
| `report/references.bib` | 参考文献 | 124行 (14条) |

**建议**: 如需提交正式论文，应将 `survey_draft.md` 转换为 LaTeX 格式。

---

## 1. 数据不一致清单

### 1.1 `report/survey_draft.md` — D4 论文草稿

| 位置 | 当前值 | 实际最新值 | 严重级别 |
|------|--------|-----------|----------|
| §4.1 Mortality Best Model | "Full Fusion 0.844" | EarlyFusion_XGBoost **0.866** | 🔴 Critical |
| §4.1 LOS Best Model | "Full Fusion 0.844" | EarlyFusion_XGBoost **0.812** | 🔴 Critical |
| §4.1 Readmission | "XGBoost 0.632" | GradientBoosting **0.634** | 🟡 Minor |
| §4.2 ±6h AUROC | "0.777" | **0.805** (XGBoost test) | 🔴 Critical |
| §4.2 ±12h AUROC | "0.800" | **0.835** (XGBoost test) | 🔴 Critical |
| §4.2 ±24h AUROC | "0.833" | **0.865** (XGBoost test) | 🔴 Critical |
| Abstract Results | "0.844 for mortality" | **0.866** | 🔴 Critical |
| §2.1 Table TIMELY-Bench row | "Early+Late+GRU 0.844" | **Early+Late+GRU 0.866** | 🔴 Critical |
| §4.4 Baseline LOS | "0.739" | 需核实 | 🟡 Minor |
| §6 References | "[To be added]" | 14条已在bib中 | 🟡 Incomplete |
| Appendices A/B/C | 标题占位符 | 内容为空 | 🟡 Incomplete |

### 1.2 `documentation/FINAL_PROJECT_REPORT.md` — 项目报告

| 位置 | 问题 | 建议 |
|------|------|------|
| §5.1 TextOnly Bug描述 | 仍说"not included in final comparison" | 更新为"已修复(2026-02-04)，待重训练" |
| §5.2 Late Fusion | "α=0.96+0.04" | Bug修复后需重新计算权重 |
| §3.1 Best AUROC | 0.866 (正确) | ✅ 无需更新 |
| §3.2 LOS AUROC | 0.812 (正确) | ✅ 无需更新 |

### 1.3 `report/FINAL_REPORT.md` — MedCAT 补充报告

| 位置 | 问题 | 建议 |
|------|------|------|
| 整体 | 仅有MedCAT添加前后的对比 | 需要与最新窗口结果整合 |
| Mortality XGBoost Test | 0.8195 | 这是tabular-only数据，与fusion不同 — 注明模型类型 |
| 表格标题 | "Episode Stats + MedCAT" | 明确说明不含BERT/Concept特征 |

### 1.4 `report/POSTER.md` — Poster草稿

| 问题 | 说明 |
|------|------|
| 仅32行 | 内容极度不足，不可用于A0 Poster |
| 无可视化 | 无图表引用 |
| 数据过时 | 使用MedCAT阶段数据 |
| 缺少方法/结论 | 只有数字，无学术结构 |

### 1.5 `report/references.bib` — 参考文献

| 问题 | 说明 |
|------|------|
| 共14条引用 | 足够初稿使用 |
| 缺少关键引用 | Sepsis-3 (Singer 2016), KDIGO (2012), Berlin Definition (2012) |
| 部分条目不完整 | `mdpi2023scoping` 缺少author完整信息 |
| RETAIN引用年份 | 文件写2020但实际是2016 (NeurIPS) |

---

## 2. 结构性问题

### 2.1 `survey_draft.md` 论文结构评估

| 章节 | 完成度 | 问题 |
|------|--------|------|
| Abstract | 70% | 数字需更新，缺少Extension贡献描述 |
| 1. Introduction | 80% | 结构良好，Contributions需加Reasoning特征 |
| 2. Related Work | 85% | 表格完整，缺少2024-2025年新文献 |
| 3. Methods | 90% | 较完整，3.4.3 Reasoning部分可扩展 |
| 4. Results | 60% | ⚠️ 数字全部需更新到最新版本 |
| 5. Discussion | 75% | Key findings基本正确，需更新具体数字 |
| 6. Conclusion | 50% | 过于简短，缺少量化总结 |
| References | 0% | 仅占位符 "[To be added]" |
| Appendices | 0% | 空白 |

### 2.2 缺少的内容

1. **Cross-Window Robustness分析** — 论文未包含Friedman test/Wilcoxon检验结果
2. **校准分析** — ECE/Brier Score仅出现在项目报告中，论文未涉及
3. **Extension贡献** — Syndrome Detection F1数字、Disease Timeline统计
4. **可视化** — 论文无图表引用 (项目有heatmap/lineplot PNG文件)
5. **Data Leakage验证** — 重要的负面结果，应写入论文

---

## 3. 优先修复建议

### 🔴 Priority 1 — 必须立即修复 (影响论文正确性)

1. **更新 `survey_draft.md` §4 所有数字**
   - Mortality: 0.844 → 0.866 (EarlyFusion_XGBoost)
   - LOS: 0.844 → 0.812
   - 窗口比较: 更新三个窗口的AUROC
   - Abstract同步更新

2. **更新 `survey_draft.md` §2.1 对比表**
   - TIMELY-Bench行: 0.844 → 0.866

3. **修复 `documentation/FINAL_PROJECT_REPORT.md` §5.1**
   - 标记TextOnly bug已修复

### 🟡 Priority 2 — 应该修复 (影响论文完整性)

4. **补充References** — 从bib文件生成引用列表
5. **添加Calibration节** — 将ECE/Brier数据写入论文
6. **添加Statistical Tests节** — Friedman p<0.001
7. **补充Appendices** — 至少写A: Episode Schema

### 🟢 Priority 3 — 建议改进 (影响论文质量)

8. **转换为LaTeX格式** — 使用标准期刊模板
9. **添加图表** — 引用results/robustness/下的PNG
10. **扩写Discussion** — 加入Extension贡献讨论
11. **补充bib** — 添加Sepsis-3/KDIGO/Berlin引用

---

## 4. 快速修复参考

### survey_draft.md §4.1 应改为:

```markdown
| Task | Best Model | AUROC | AUPRC |
|------|------------|-------|-------|
| **Mortality** | EarlyFusion_XGBoost | **0.866** | 0.536 |
| **Prolonged LOS** | EarlyFusion_XGBoost | **0.812** | 0.463 |
| **30-Day Readmission** | GradientBoosting | 0.634 | 0.222 |
```

### survey_draft.md §4.2 应改为:

```markdown
| Window | AUROC | Δ vs ±24h |
|--------|-------|-----------|
| ±6h | 0.805 | -0.060 |
| ±12h | 0.835 | -0.030 |
| **±24h** | **0.865** | — |
```

### survey_draft.md Abstract 应改为:

> **Results**: Our best model (EarlyFusion XGBoost) achieves AUROC of 0.866 for mortality prediction and 0.812 for prolonged LOS prediction. The ±24h alignment window provides optimal performance across all models, with statistical significance confirmed by Friedman tests (p<0.001).

---

## 5. 总结

| 维度 | 评分 | 说明 |
|------|------|------|
| 数据准确性 | C | 多个关键数字过时 |
| 结构完整性 | B- | 框架完整但缺少校准/鲁棒性章节 |
| 引用质量 | C+ | 有bib但论文未引用，缺少关键文献 |
| Extension内容 | D | 几乎未写入论文 |
| 可提交性 | ❌ | 当前状态不可直接提交 |

**总体判断**: 论文草稿框架良好，但数字过时、Extension内容缺失。需要1-2轮集中更新才能达到提交标准。
