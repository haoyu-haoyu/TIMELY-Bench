# 快速参考指南

## 一行命令

### 验证配置
```bash
python config.py
```

### 运行测试
```bash
python test_gru_fixes.py
```

### 开始训练
```bash
python train_temporal_gru_v2.py
```

## 关键配置参数

### config.py

```python
# 模型架构
HIDDEN_DIM = 64          # GRU隐藏维度
NUM_LAYERS = 2           # GRU层数
DROPOUT = 0.2            # Dropout比例

# 训练设置
BATCH_SIZE = 256         # 批次大小
EPOCHS = 50              # 最大训练轮数
LR = 0.001               # 初始学习率

# Early Stopping
EARLY_STOPPING_PATIENCE = 10      # 耐心值（epoch）
EARLY_STOPPING_MIN_DELTA = 1e-4   # 最小改善幅度

# 学习率调度器
LR_SCHEDULER_PATIENCE = 5         # 耐心值（epoch）
LR_SCHEDULER_FACTOR = 0.5         # 学习率衰减因子
LR_SCHEDULER_MIN_LR = 1e-6        # 最小学习率

# 数据划分
TEST_SIZE = 0.2          # 测试集比例
N_FOLDS = 5              # 交叉验证折数
```

## 修复内容清单

- [x] 创建 `config.py` 统一配置管理
- [x] 添加 `EarlyStopping` 类
- [x] 添加 `TrainingLogger` 类
- [x] 集成 `ReduceLROnPlateau` 学习率调度器
- [x] 实现 CSV/JSON 结果保存
- [x] 添加文件验证函数 `check_files()`
- [x] 完善异常处理和错误信息
- [x] 创建测试脚本 `test_gru_fixes.py`

## 输出文件

```
Output_temporal_gru/
├── training_results.csv    # 交叉验证结果
├── training_results.json   # 完整配置和结果
├── models/                 # 保存的最佳模型
│   └── best_model_fold*.pt
└── logs/                   # 训练过程日志
    └── training_log_fold*_*.json
```

## 常用调整

### 更快训练（测试用）
```python
EPOCHS = 20
N_FOLDS = 3
BATCH_SIZE = 512
```

### 更好性能（生产用）
```python
HIDDEN_DIM = 128
NUM_LAYERS = 3
EARLY_STOPPING_PATIENCE = 15
```

### 更小模型（资源受限）
```python
HIDDEN_DIM = 32
BATCH_SIZE = 128
```

## 核心类使用

### Early Stopping
```python
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
if early_stopping(val_metric, epoch):
    break  # 停止训练
```

### 训练日志
```python
logger = TrainingLogger(LOG_DIR)
logger.log_epoch(epoch, train_loss, val_loss, val_auroc, val_auprc, lr)
log_file = logger.save(fold=1)
```

### 学习率调度
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
scheduler.step(val_auroc)
```

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| 文件未找到 | 检查 `config.py` 路径配置 |
| 内存不足 | 降低 `BATCH_SIZE` |
| 过拟合 | 增加 `DROPOUT` 或减少 `HIDDEN_DIM` |
| 训练太慢 | 增加 `BATCH_SIZE` 或减少 `EPOCHS` |
| 早停太早 | 增加 `EARLY_STOPPING_PATIENCE` |

## 性能监控

训练时关注：
- **Val AUROC**: 主要评估指标
- **Val AUPRC**: 类别不平衡时的重要指标
- **EarlyStopping counter**: 监控是否即将停止
- **Learning Rate**: 观察学习率衰减

## 下一步

1. 运行测试验证代码：`python test_gru_fixes.py`
2. 查看完整文档：`GRU_FIX_README.md`
3. 开始训练：`python train_temporal_gru_v2.py`
4. 分析结果：检查 `Output_temporal_gru/` 目录

## 支持

- 详细文档：`GRU_FIX_README.md`
- 配置说明：`config.py`
- 测试脚本：`test_gru_fixes.py`
