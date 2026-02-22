# TIMELY-Bench v2.0 GRU 模型修复说明

## 修复概览

本次修复为 TIMELY-Bench v2.0 项目的 GRU 时序模型代码添加了多项增强功能，提升了代码的健壮性、可维护性和训练效果。

## 修复内容

### 1. 统一配置管理 (`config.py`)

创建了集中式配置文件，统一管理所有路径和超参数：

#### 主要功能：
- **路径配置**：所有数据文件和输出目录的路径统一管理
- **模型超参数**：GRU 层数、隐藏维度、Dropout等
- **训练配置**：批次大小、学习率、训练轮数等
- **Early Stopping配置**：耐心值、最小改善幅度
- **学习率调度器配置**：耐心值、衰减因子、最小学习率

#### 使用方法：
```python
from config import HIDDEN_DIM, EPOCHS, LR, ensure_directories

# 确保所有目录存在
ensure_directories()

# 验证关键文件
files_ok, missing = validate_files()
```

#### 配置验证：
```bash
python config.py
```

### 2. Early Stopping 机制

添加了智能早停机制，防止过拟合并节省训练时间。

#### 特性：
- **自动停止**：验证集性能不再提升时自动停止训练
- **最佳模型追踪**：记录最佳验证性能对应的epoch
- **可配置参数**：
  - `patience`: 容忍多少个epoch无改善（默认10）
  - `min_delta`: 最小改善幅度（默认1e-4）
  - `verbose`: 是否打印详细信息

#### 实现细节：
```python
early_stopping = EarlyStopping(
    patience=10,
    min_delta=1e-4,
    verbose=True
)

# 在训练循环中
if early_stopping(val_auroc, epoch):
    print(f"Early Stopping at epoch {epoch}")
    break
```

### 3. 学习率调度器

使用 PyTorch 的 `ReduceLROnPlateau` 自动调整学习率。

#### 功能：
- **自适应降低学习率**：验证性能停滞时降低学习率
- **配置参数**：
  - `patience`: 5个epoch无改善后降低学习率
  - `factor`: 学习率乘以0.5（减半）
  - `min_lr`: 最小学习率1e-6

#### 使用：
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5,
    patience=5, min_lr=1e-6, verbose=True
)

# 每个epoch后
scheduler.step(val_auroc)
```

### 4. 训练日志记录

完整记录训练过程的所有指标。

#### 记录内容：
- 每个epoch的训练损失
- 验证集损失
- 验证集 AUROC
- 验证集 AUPRC
- 当前学习率

#### 保存格式：
- JSON 文件，便于后续分析
- 每个fold单独保存
- 文件名包含时间戳

#### 示例：
```python
logger = TrainingLogger(LOG_DIR)

# 每个epoch
logger.log_epoch(epoch, train_loss, val_loss, val_auroc, val_auprc, lr)

# 训练结束
log_file = logger.save(fold=1, filename_suffix='20241224')
```

### 5. 结果保存功能

多格式保存训练结果，方便分析和报告。

#### CSV 格式：
保存每个fold的详细结果：
- fold编号
- 验证集AUROC
- 验证集AUPRC
- 最佳epoch
- 总训练轮数

#### JSON 格式：
保存完整的配置和结果：
```json
{
  "timestamp": "2024-12-24T...",
  "config": {
    "hidden_dim": 64,
    "num_layers": 2,
    "learning_rate": 0.001,
    ...
  },
  "cross_validation": {
    "mean_auroc": 0.8234,
    "std_auroc": 0.0156,
    "fold_details": [...]
  },
  "test": {
    "test_auroc": 0.8156,
    "test_auprc": 0.7823
  }
}
```

### 6. 错误处理机制

添加了完整的异常处理和用户友好的错误信息。

#### 功能：
- **文件检查**：训练前验证所有必要文件是否存在
- **Try-Catch包装**：所有关键步骤都有异常捕获
- **详细错误信息**：出错时提供traceback
- **优雅退出**：捕获键盘中断（Ctrl+C）

#### 示例输出：
```
❌ 数据加载失败: FileNotFoundError: timeseries.csv not found
部分必要文件缺失，请检查路径配置！
```

### 7. 文件存在性验证

训练前自动检查所有必要文件。

#### 检查内容：
- 时序数据文件
- 笔记时间文件
- LLM特征文件
- 标签文件
- 键值文件

#### 输出示例：
```
📁 检查数据文件...
   ✅ 时序数据: timeseries.csv (100.0 MB)
   ✅ 笔记时间: note_time.csv (88.7 MB)
   ✅ LLM特征: llm_features_deepseek.csv (1.6 MB)
   ✅ 标签文件: processed_labels.csv (0.1 MB)
   ✅ 键值文件: processed_keys.csv (0.8 MB)
```

## 使用指南

### 快速开始

#### 1. 验证配置
```bash
python config.py
```

#### 2. 运行测试
```bash
python test_gru_fixes.py
```

#### 3. 开始训练
```bash
python train_temporal_gru_v2.py
```

### 自定义配置

编辑 `config.py` 文件修改配置：

```python
# 模型配置
HIDDEN_DIM = 128  # 增加隐藏维度
NUM_LAYERS = 3    # 增加GRU层数

# 训练配置
EPOCHS = 100              # 增加最大训练轮数
BATCH_SIZE = 128          # 减小批次大小
LR = 0.0005              # 降低学习率

# Early Stopping
EARLY_STOPPING_PATIENCE = 15  # 更耐心的早停
```

### 输出文件结构

训练后会生成以下文件结构：

```
Output_temporal_gru/
├── training_results.csv          # CSV格式结果
├── training_results.json         # JSON格式结果
├── models/                       # 保存的模型
│   ├── best_model_fold1.pt
│   ├── best_model_fold2.pt
│   └── ...
└── logs/                         # 训练日志
    ├── training_log_fold0_20241224.json
    ├── training_log_fold1_20241224.json
    └── ...
```

## 训练流程

### 完整流程图

```
1. 文件验证
   ↓
2. 数据加载和预处理
   ↓
3. 分离测试集（20%）
   ↓
4. 5折交叉验证
   │
   ├─→ Fold 1
   │    ├── 标准化
   │    ├── 训练（Early Stopping + LR Scheduler）
   │    ├── 验证
   │    └── 保存最佳模型
   │
   ├─→ Fold 2
   │    └── ...
   │
   └─→ Fold 5
        └── ...
   ↓
5. 在独立测试集上评估
   ↓
6. 保存结果（CSV + JSON）
```

### 训练监控

训练过程中会实时显示：

```
Fold 1/5
开始训练...
   Epoch   0/50: Train Loss=0.6234, Val Loss=0.5891, Val AUROC=0.7234, Val AUPRC=0.6891, LR=0.001000
   Epoch   5/50: Train Loss=0.4567, Val Loss=0.4321, Val AUROC=0.7891, Val AUPRC=0.7456, LR=0.001000
      ✓ Validation improved by 0.06570
   Epoch  10/50: Train Loss=0.3456, Val Loss=0.3789, Val AUROC=0.8123, Val AUPRC=0.7789, LR=0.000500
      EarlyStopping counter: 1/10

   ⏹️  Early Stopping at epoch 23
   📌 Best epoch was 15 with AUROC=0.8234

   💾 训练日志已保存: Output_temporal_gru/logs/training_log_fold0_20241224.json
   💾 模型已保存: Output_temporal_gru/models/best_model_fold1.pt

   ✅ Fold 1 完成:
      最佳 AUROC: 0.8234 (Epoch 15)
      最佳 AUPRC: 0.7891
```

## 性能优化建议

### 1. 调整 Early Stopping

如果模型过早停止：
```python
EARLY_STOPPING_PATIENCE = 15  # 增加耐心值
EARLY_STOPPING_MIN_DELTA = 1e-5  # 降低最小改善幅度
```

### 2. 调整学习率策略

更激进的学习率衰减：
```python
LR_SCHEDULER_PATIENCE = 3     # 更快降低学习率
LR_SCHEDULER_FACTOR = 0.3     # 更大的衰减
```

### 3. 增加模型容量

```python
HIDDEN_DIM = 128
NUM_LAYERS = 3
```

### 4. 数据增强

可以在数据加载函数中添加：
- 时间步随机丢弃
- 特征噪声注入
- 样本权重平衡

## 故障排查

### 常见问题

#### 1. 文件未找到
```
❌ 时序数据: timeseries.csv (不存在)
```
**解决**：检查 `config.py` 中的路径配置

#### 2. 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**：降低 `BATCH_SIZE`

#### 3. 训练太慢
```python
# 减少数据量测试
EPOCHS = 20
N_FOLDS = 3
```

#### 4. 验证性能不稳定
- 增加 `EARLY_STOPPING_PATIENCE`
- 检查数据质量
- 尝试不同的学习率

## 技术规格

### 系统要求
- Python 3.7+
- PyTorch 1.8+
- NumPy, Pandas, Scikit-learn

### 计算资源
- 推荐：GPU（CUDA或MPS）
- 最小内存：8GB RAM
- 存储空间：约500MB用于模型和日志

### 训练时间估计
- 每个fold：10-30分钟（取决于硬件）
- 完整5折交叉验证：1-2.5小时
- 使用Early Stopping通常可节省30-50%时间

## 版本历史

### v2.0 (当前版本)
- ✅ 添加统一配置管理
- ✅ 实现 Early Stopping
- ✅ 集成学习率调度器
- ✅ 完善错误处理
- ✅ 多格式结果保存
- ✅ 详细训练日志

### v1.0 (原始版本)
- 基础 GRU 模型
- 简单交叉验证
- 基本结果输出

## 贡献与支持

如有问题或建议，请查看：
- 项目文档：`documentation/` 目录
- 测试脚本：`test_gru_fixes.py`
- 配置文件：`config.py`

## 许可证

遵循 TIMELY-Bench 项目原有许可证。
