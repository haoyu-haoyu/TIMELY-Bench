# TextOnly 模型 Bug 修复报告

**日期**: 2026-02-04
**严重级别**: 🔴 Critical
**影响范围**: TextOnly模型、Fusion模型、DL校准评估

---

## 1. Bug 描述

### 问题
在提取临床笔记文本长度特征时，代码使用了错误的字段名 `text`，而Episode JSON实际数据字段名为 `text_full`。

### 根因
Episode JSON 的 notes 结构中，文本内容存储在 `text_full` 字段：
```json
{
  "note_id": "...",
  "note_type": "...",
  "note_category": "...",
  "chart_hour": 12.5,
  "chart_time": "2180-01-01 12:30:00",
  "text_full": "Patient is alert and oriented...",
  "text_relevant": "...",
  "text_length": 1234,
  "has_llm_features": true
}
```

但代码错误使用了 `n.get('text', '')`，该字段不存在，导致始终返回空字符串。

### 影响
- `total_text_length` 特征始终为 0
- `avg_text_length` 特征始终为 0
- TextOnly模型无法学习到文本长度相关信息
- Fusion模型的文本分支同样受影响

---

## 2. 修复方案

### 修复策略
使用三级回退（fallback）策略，兼容不同版本的Episode结构：

```python
# 修复前 (BUG):
n.get('text', '')

# 修复后 (FIXED):
n.get('text_full') or n.get('text_relevant') or n.get('text', '')
```

### 修复文件清单

| 文件 | 行号 | 状态 |
|------|------|------|
| `code/baselines/train_text_only.py` | L54 | ✅ 已修复 |
| `code/baselines/train_fusion.py` | L170 | ✅ 已修复 |
| `code/evaluation/run_dl_calibration_hpc.py` | L267 | ✅ 已修复 |

---

## 3. 验证步骤

### 3.1 本地验证
```bash
# 快速验证修复生效
python -c "
import json, glob
files = sorted(glob.glob('episodes/episodes_enhanced/*.json'))[:5]
for f in files:
    ep = json.load(open(f))
    notes = ep.get('clinical_text', {}).get('notes', [])
    for n in notes[:1]:
        text = n.get('text_full') or n.get('text_relevant') or n.get('text', '')
        print(f'{f}: text_length={len(text)}')
"
```

### 3.2 CREATE HPC 重训练
修复后需在 CREATE HPC 上重新训练 TextOnly 和 Fusion 模型。
详见 `scripts/rerun_textonly_on_create.sh`。

---

## 4. 预期影响

| 模型 | 修复前 AUROC | 预期修复后 AUROC |
|------|-------------|-----------------|
| TextOnly | ~0.50 (随机) | 0.60-0.70 |
| EarlyFusion | 0.866 | 0.866+ (小幅提升) |
| DL Calibration | 受影响 | 校准更准确 |

**说明**: TextOnly模型应有显著提升；Fusion模型因文本长度仅占特征的一小部分，改善幅度较小。

---

## 5. 修复验证通过标准

- [ ] `total_text_length` 特征不再全为 0
- [ ] TextOnly 模型 AUROC > 0.55
- [ ] Fusion 模型结果与修复前持平或提升
- [ ] 无新增回归问题
