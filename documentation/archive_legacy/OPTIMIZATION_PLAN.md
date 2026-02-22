# TIMELY-Bench 下一步优化计划

**创建日期**: 2024-12-24
**基于**: 项目深度分析报告

---

## 执行摘要

根据项目分析，我们识别出以下核心问题和优化方向：

| 问题 | 严重度 | 解决方案 |
|------|--------|----------|
| Readmission预测性能极差 (AUROC 0.62) | 🔴 高 | 移除或重新定义任务 |
| 文本融合无收益 (<1%提升) | 🔴 高 | 增强文本特征 + 改进融合策略 |
| ARDS样本不足 (822例, 1.1%) | 🟡 中 | 合并疾病类别或过采样 |
| GRU时序模型未完成 | 🟡 中 | 完成实现并评估 |

---

## 阶段一：数据质量优化 (第1-2天)

### 任务 1.1: 处理Readmission任务问题

**问题**: AUROC仅0.62，无临床意义

**方案选择**:
- [ ] 方案A: 移除Readmission任务
- [ ] 方案B: 改为"7天ICU再入院"预测
- [ ] 方案C: 保留但标注为"实验性"

**执行步骤**:
1. 分析Readmission的特征重要性
2. 决定保留/移除/修改
3. 更新数据集和文档

### 任务 1.2: 扩大LLM标注规模

**当前状态**: 100样本已标注 (63% SUPPORTIVE)
**目标**: 标注全部500样本

**执行步骤**:
1. 运行annotate_patterns.py标注剩余400样本
2. 统计完整标注结果
3. 分析不同Pattern的标注分布

### 任务 1.3: 处理全量数据对齐

**当前状态**: 仅处理5,000患者 (采样)
**目标**: 处理全部74,812患者

**执行步骤**:
1. 优化内存使用，分批处理
2. 运行全量时序-文本对齐
3. 生成完整的对齐数据集

---

## 阶段二：增强文本特征 (第3-5天)

### 任务 2.1: 改进文本特征提取

**当前问题**: 仅5个二值特征，信息量不足

**增强方案**:
```python
# 当前特征 (5个)
["pneumonia", "edema", "infection", "consolidation", "pleural_effusion"]

# 扩展特征 (20+个)
新增类别:
- 生命体征描述: fever_mentioned, tachycardia_mentioned, hypotension_mentioned
- 实验室异常: elevated_creatinine, abnormal_wbc, elevated_lactate
- 器官功能: respiratory_failure, renal_failure, cardiac_dysfunction
- 干预措施: intubated, vasopressors, dialysis
- 临床状态: sepsis_documented, shock_documented, altered_mental_status
```

### 任务 2.2: 添加临床概念嵌入

**方案**: 使用ClinicalBERT生成文本向量

**执行步骤**:
1. 安装transformers库
2. 加载Bio_ClinicalBERT模型
3. 为每个笔记生成768维向量
4. 降维到50-100维用于融合

### 任务 2.3: 关键词匹配优化

**目标**: 降低UNRELATED比例从20%到<10%

**优化策略**:
1. 扩展Pattern-关键词映射
2. 添加同义词和缩写
3. 使用模糊匹配

---

## 阶段三：改进融合策略 (第6-8天)

### 任务 3.1: 完成GRU时序模型

**文件**: `train_temporal_gru_v2.py`

**执行步骤**:
1. 检查现有GRU代码
2. 补充缺失部分
3. 训练并评估
4. 对比XGBoost baseline

### 任务 3.2: 实现Attention融合

**架构设计**:
```
时序特征 (T×D_ts) → GRU → 时序表示 (H_ts)
                            ↓
文本特征 (D_txt) → MLP → 文本表示 (H_txt) → Cross-Attention → 融合表示 → 预测
                            ↑
                     Query: H_ts
```

### 任务 3.3: 多模态Transformer (可选)

**架构**:
- 使用预训练的临床Transformer
- 输入: [CLS] + 时序tokens + [SEP] + 文本tokens
- 输出: 分类头

---

## 阶段四：特征重要性分析 (第9-10天)

### 任务 4.1: SHAP分析

**目标**: 识别最重要的时序特征和Pattern

**执行步骤**:
1. 为XGBoost模型计算SHAP值
2. 生成特征重要性排名
3. 分析Pattern对预测的贡献

### 任务 4.2: Pattern关联分析

**目标**: 理解Pattern之间的共现关系

**执行步骤**:
1. 计算Pattern共现矩阵
2. 分析Pattern与结局的关联
3. 识别高价值Pattern组合

---

## 阶段五：模型优化与评估 (第11-14天)

### 任务 5.1: 超参数调优

**方法**: Optuna贝叶斯优化

**调优目标**:
- XGBoost: max_depth, learning_rate, n_estimators
- GRU: hidden_size, n_layers, dropout
- 融合模型: attention_heads, fusion_dim

### 任务 5.2: 集成学习

**策略**:
```python
ensemble_prediction = (
    0.4 * xgboost_prob +
    0.3 * gru_prob +
    0.3 * fusion_prob
)
```

### 任务 5.3: 最终评估

**评估内容**:
1. 5折交叉验证
2. 独立测试集
3. 按疾病亚组分析
4. 校准曲线

---

## 执行时间表

```
第1天:  任务1.1 - 处理Readmission问题
第2天:  任务1.2 + 1.3 - 扩大标注 + 全量对齐
第3天:  任务2.1 - 增强文本特征设计
第4天:  任务2.2 - ClinicalBERT嵌入
第5天:  任务2.3 - 关键词优化
第6天:  任务3.1 - 完成GRU模型
第7天:  任务3.2 - Attention融合
第8天:  任务3.2续 - 调试融合模型
第9天:  任务4.1 - SHAP分析
第10天: 任务4.2 - Pattern关联分析
第11天: 任务5.1 - 超参数调优
第12天: 任务5.2 - 集成学习
第13天: 任务5.3 - 最终评估
第14天: 文档整理 + 结果可视化
```

---

## 预期成果

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| Mortality AUROC | 0.861 | 0.88+ |
| LOS AUROC | 0.798 | 0.82+ |
| 融合收益 | <1% | 3-5% |
| SUPPORTIVE标注 | 63% | 70%+ |
| 文本特征数 | 5 | 20+ |
| 对齐覆盖率 | 36.5% | 60%+ |

---

## 立即开始

建议从以下任务开始：

1. **任务1.2**: 扩大LLM标注规模 (快速见效)
2. **任务2.1**: 增强文本特征 (核心改进)
3. **任务3.1**: 完成GRU模型 (深度学习基线)

请确认您想先执行哪个任务，我将开始实施。
