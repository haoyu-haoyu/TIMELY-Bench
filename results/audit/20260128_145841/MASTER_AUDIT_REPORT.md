# TIMELY-Bench 端到端审计报告

**审计 ID**: 20260128_145841  
**时间戳**: 2026-01-28T15:31:31  
**项目路径**: /scratch/users/k25113331/TIMELY-Bench_Final

---

## 总体判定: **PASS_WITH_WARNINGS**

✅ 所有关键数据完整性检查通过  
✅ 无信息泄漏、时间穿越或病人级泄漏  
⚠️ 存在少量警告（非关键）

---

## 各阶段检查结果

| 阶段 | 检查项 | 状态 | 说明 |
|------|--------|------|------|
| A | 环境设置 | ⚠️ | 元数据记录（无pass字段） |
| B | 文件清单 | ✅ PASS | 关键文件完整 |
| C (对齐) | 时间窗口 | ✅ PASS | 无discharge笔记，hour∈[0,24) |
| C (Episode) | Episode窗口 | ✅ PASS | discharge=0, hour_lt_0=0, hour_gte_24=0 |
| D | 标签完整性 | ✅ PASS | mortality冲突=0，标签有效 |
| E | 模式一致性 | ✅ PASS | 时序小时范围[0,23]，无违规 |
| F | 聚合/缺失 | ✅ PASS | NaN=0, Inf=0, 检查93765值 |
| G | 文本管道 | ✅ PASS | 未来笔记=0, discharge摘要=0 |
| **H** | **训练泄漏** | ✅ **PASS** | **GroupKFold已使用，无泄漏风险** |
| I | 交付符合性 | ⚠️ | PROVENANCE.json缺失 |

---

## 关键发现

### ✅ 无关键问题

1. **训练/评估泄漏检查 (Phase H) - 通过**
   - GroupKFold: 已使用
   - 多次入院患者: 仅3位（996位中）
   - 潜在泄漏风险: 无

2. **时间窗口约束 (Phase C) - 通过**
   - Discharge笔记: 0
   - 负小时数: 0
   - ≥24小时: 0

3. **标签完整性 (Phase D) - 通过**
   - Mortality率: 10.5%
   - Readmission率: 18.2%
   - 冲突: 0

4. **数据质量 (Phase E, F, G) - 通过**
   - NaN/Inf值: 0
   - 未来数据泄漏: 0

---

## 警告（非关键）

1. **Phase A**: ENVIRONMENT.json是元数据文件，无pass字段（正常）
2. **Phase I**: PROVENANCE.json缺失（可选交付物）

---

## 详细统计

### 数据规模
- 对齐文件: 6,974,407 行
- Episode文件: 74,830 个
- 时序值检查: 93,765 个
- 临床笔记检查: 46,233 条

### 标签分布（样本）
- Mortality: 210/2000 (10.5%)
- Readmission: 364/2000 (18.2%)

---

## 结论

**项目符合 TIMELY-Bench 基线可复现要求。**

关键验证：
1. ✅ 无病人级数据泄漏（GroupKFold正确使用）
2. ✅ 无时间穿越（所有数据在24小时观察窗口内）
3. ✅ 无特征构建错误（无NaN/Inf，时序一致）
4. ⚠️ PROVENANCE.json可选文件缺失，不影响科学有效性

---

## 审计文件清单

```
results/audit/20260128_145841/
├── ENVIRONMENT.json
├── FILE_INVENTORY.json
├── ALIGNMENT_WINDOW_CHECK.json
├── EPISODE_WINDOW_CHECK.json
├── LABEL_INTEGRITY_CHECK.json
├── PATTERN_FEATURE_CONSISTENCY.json
├── AGGREGATION_MISSINGNESS_AUDIT.json
├── TEXT_PIPELINE_AUDIT.json
├── TRAINING_LEAKAGE_AUDIT.json
├── DELIVERY_COMPLIANCE_CHECK.json
├── MASTER_AUDIT_REPORT.json
└── MASTER_AUDIT_REPORT.md
```

---

*报告生成于 2026-01-28 by Claude Code*
