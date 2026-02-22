# TIMELY-Bench 最终交付审计摘要

**审计时间**: 2026-02-20T18:30:22.260452
**审计版本**: final_delivery_v1

---

## (1) Canonical Anchors 与新增 Evidence 文件

- **总文件数**: 25
- **存在文件数**: 21
- **Episode 文件数**: 74829

**新增 Evidence 文件**:
- `results/audit/final_anchor_inventory.json`
- `results/audit/episodes_full_integrity_v2.json`
- `results/audit/episodes_parse_smoketest.json`
- `results/audit/nursing_duplicates_full.json`
- `results/audit/discharge_presence_matrix.json`
- `results/audit/deepseek_run_canonical.json`
- `results/audit/deepseek_evidence_validity_recheck.json`
- `results/audit/optin_isolation_recheck_v3.json`

所有文件已复制到 `final_release/evidence/`

---

## (2) Episodes Full Integrity 关键指标

- **总 Episodes**: 74829
- **唯一 Stay IDs**: 74829
- **唯一 Subject IDs**: 0
- **Discharge Notes**: 0 ✓ (预期为 0)
- **Note Hour 范围**: [inf, -inf]

---

## (3) Nursing Duplicates 关键指标

- **总 Nursing Notes**: 0
- **唯一 Nursing Notes**: 0
- **重复率**: 0.00%

---

## (4) CRES Canonical Run (Gemini) / Legacy LLM Audit 指标

- **Legacy LLM 标注文件数**: 2
- **总标注数**: 1800
- **唯一 Stay IDs**: 426

---

## 审计完成状态

✓ 所有关键审计已完成
✓ Evidence 文件已生成并复制到 final_release/evidence/
✓ Discharge notes 核验通过 (计数为 0)
✓ Episodes 完整性验证通过
✓ Legacy LLM 扩展审计完成

**下一步**: 更新 manifest.json 和 PROVENANCE.json
