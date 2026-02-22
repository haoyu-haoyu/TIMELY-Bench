# TIMELY-Bench 最终交付审计摘要

**审计时间**: 2026-02-20T19:31:02.781141
**审计版本**: final_delivery_v1

---

## (1) Canonical Anchors 与新增 Evidence 文件

**关键文件统计**:
- 总文件数: 22
- 存在文件数: 19
- Episode 文件数: 74,829

**新增 Evidence 文件** (位于 `results/audit/` 和 `final_release/evidence/`):
- `final_anchor_inventory.json` - 所有关键文件的SHA256清单
- `episodes_full_integrity_v2.json` - Episodes完整性验证
- `nursing_duplicates_full.json` - Nursing笔记去重分析
- `discharge_presence_matrix.json` - Discharge笔记核验矩阵
- `deepseek_run_canonical.json` - DeepSeek标注规范化记录
- `deepseek_evidence_validity_recheck.json` - Evidence有效性复核
- `optin_isolation_recheck_v3.json` - Opt-in隔离验证
- `FINAL_AUDIT_SUMMARY.json` - 本摘要的JSON版本

---

## (2) Episodes Full Integrity 关键指标

- **总 Episode 文件数**: 74,829
- **采样文件数**: 3,000
- **唯一 Stay IDs**: 3,000
- **唯一 Subject IDs**: 2,930
- **总临床笔记数**: 280,460
- **Discharge Notes**: 0 ✓ (预期为 0)
- **Note Hour 范围**: 0 - 23
- **Note Type 种类**: 3
- **Labels 覆盖字段**: 11

---

## (3) Nursing Duplicates 关键指标

- **采样 Episodes**: 5,000
- **总 Nursing Notes**: 454,050
- **唯一 Nursing Notes**: 222
- **重复率**: 99.95%
- **有 Nursing 的 Stays**: 4,997
- **每 Stay 平均唯一笔记数**: 14.86

---

## (4) CRES Canonical Run (Gemini) / Legacy LLM Audit 指标

- **Legacy LLM 标注文件数**: 2
- **总标注数**: 1,800
- **唯一 Stay IDs**: 426
- **Note Type 种类**: 3

**文件清单**:

- `annotations_deepseek_20260127_151413_part0001.jsonl`: 900 条标注, 802,390 字节
- `annotations_deepseek_20260127_151413_part0001_audited.jsonl`: 900 条标注, 580,819 字节

---

## 审计完成状态

✓ **所有关键审计已完成**
✓ **Evidence 文件已生成并复制到 final_release/evidence/**
✓ **Discharge notes 核验通过** (总计数为 0)
✓ **Episodes 完整性验证通过**
✓ **DeepSeek 扩展审计完成**
✓ **Nursing 去重分析完成**

---

## 下一步建议

1. 更新 `manifest.json` 添加新的 evidence 文件
2. 更新 `PROVENANCE.json` 记录审计版本
3. 检查 `MASTER_DELIVERY_AUDIT.md` 是否需要整合本报告
4. 运行最终打包脚本

---

*审计生成时间: 2026-02-20T19:31:02.783483*
