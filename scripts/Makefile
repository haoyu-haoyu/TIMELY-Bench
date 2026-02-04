# TIMELY-Bench v2.0 Makefile
# ============================
# 完整的可复现自动化流程

.PHONY: all install data baselines fusion gru eval verify clean help

# Python 解释器
PYTHON = python3

# 目录
CODE_DIR = code
DATA_DIR = data/processed
RESULTS_DIR = results
EPISODES_DIR = episodes

# ==============================================================================
# 主要目标
# ==============================================================================

# 默认：运行所有基线和融合实验
all: baselines fusion gru eval
	@echo "✅ All experiments completed!"

# 完整流程（从数据处理到评估）
full: data baselines fusion gru eval
	@echo "✅ Full pipeline completed!"

# ==============================================================================
# 环境设置
# ==============================================================================

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# ==============================================================================
# 数据处理
# ==============================================================================

data: splits patterns episodes
	@echo "✅ Data processing completed!"

# 生成固定的数据划分
splits:
	@echo "📊 Generating predefined splits..."
	$(PYTHON) $(CODE_DIR)/data_processing/generate_predefined_splits.py

# 模式检测
patterns:
	@echo "🔍 Running pattern detection..."
	$(PYTHON) $(CODE_DIR)/data_processing/pattern_detector.py

# 构建 Episodes（如需重建）
episodes:
	@echo "📁 Building episodes..."
	$(PYTHON) $(CODE_DIR)/data_processing/batch_build_all_episodes.py

# ==============================================================================
# 模型训练
# ==============================================================================

# 表格基线 (XGBoost, LR)
baselines:
	@echo "🚀 Running tabular baselines..."
	$(PYTHON) $(CODE_DIR)/baselines/train_tabular_baselines.py
	@echo "✅ Tabular baselines completed"

# 仅文本基线
text-only:
	@echo "📝 Running text-only baselines..."
	$(PYTHON) $(CODE_DIR)/baselines/train_text_only.py

# 融合实验 (Early + Late)
fusion:
	@echo "🔀 Running fusion experiments..."
	$(PYTHON) $(CODE_DIR)/baselines/train_fusion.py
	@echo "✅ Fusion experiments completed"

# GRU 时序模型
gru:
	@echo "🧠 Running GRU models..."
	$(PYTHON) $(CODE_DIR)/baselines/train_temporal_gru_v2.py
	@echo "✅ GRU models completed"

# 带 Delta 特征的训练
delta:
	@echo "📈 Running delta features experiments..."
	$(PYTHON) $(CODE_DIR)/baselines/train_with_delta_features.py
	@echo "✅ Delta features completed"

# 增强推理特征训练
reasoning:
	@echo "💡 Running enhanced reasoning experiments..."
	$(PYTHON) $(CODE_DIR)/baselines/train_enhanced_reasoning.py

# 对齐窗口对比
aligner:
	@echo "⏱️ Running alignment window comparison..."
	$(PYTHON) $(CODE_DIR)/baselines/train_aligner_comparison.py

# ==============================================================================
# 评估
# ==============================================================================

eval: calibration ablation
	@echo "✅ Evaluation completed!"

# 校准评估
calibration:
	@echo "📏 Running calibration evaluation..."
	$(PYTHON) $(CODE_DIR)/baselines/eval_calibration.py

# 笔记消融实验
ablation:
	@echo "🔬 Running note ablation..."
	$(PYTHON) $(CODE_DIR)/baselines/eval_note_ablation.py

# ==============================================================================
# 验证
# ==============================================================================

verify:
	@echo "🔍 Verifying data and results..."
	@echo ""
	@echo "--- Predefined Splits ---"
	$(PYTHON) -c "import pandas as pd; \
		df = pd.read_csv('$(DATA_DIR)/predefined_splits.csv'); \
		print(df['split'].value_counts())"
	@echo ""
	@echo "--- Episodes Count ---"
	@ls $(EPISODES_DIR)/episodes_enhanced/*.json 2>/dev/null | wc -l | xargs echo "Enhanced episodes:"
	@ls $(EPISODES_DIR)/episodes_all/*.json 2>/dev/null | wc -l | xargs echo "All episodes:"
	@echo ""
	@echo "--- Results Files ---"
	@ls -la $(RESULTS_DIR)/*/  2>/dev/null | head -20
	@echo "✅ Verification completed"

# 检查数据泄漏
check-leakage:
	@echo "🔒 Checking for data leakage..."
	$(PYTHON) -c "import pandas as pd; \
		splits = pd.read_csv('$(DATA_DIR)/predefined_splits.csv'); \
		train_ids = set(splits[splits['split']=='train']['stay_id']); \
		test_ids = set(splits[splits['split']=='test']['stay_id']); \
		overlap = train_ids & test_ids; \
		print(f'Train-Test overlap: {len(overlap)}'); \
		assert len(overlap) == 0, 'Data leakage detected!'"
	@echo "✅ No data leakage detected"

# ==============================================================================
# 清理
# ==============================================================================

clean:
	@echo "🧹 Cleaning results..."
	rm -rf $(RESULTS_DIR)/*/*.csv
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	@echo "✅ Cleaned"

clean-all: clean
	rm -rf $(EPISODES_DIR)/episodes_all/*.json
	rm -rf $(EPISODES_DIR)/episodes_enhanced/*.json
	@echo "⚠️ All data cleaned (episodes removed)"

# ==============================================================================
# 帮助
# ==============================================================================

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              TIMELY-Bench v2.0 Makefile                          ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  SETUP                                                           ║"
	@echo "║    make install      - Install Python dependencies               ║"
	@echo "║                                                                  ║"
	@echo "║  DATA                                                            ║"
	@echo "║    make data         - Process all data (splits, patterns)       ║"
	@echo "║    make splits       - Generate predefined train/val/test splits ║"
	@echo "║    make patterns     - Run pattern detection                     ║"
	@echo "║                                                                  ║"
	@echo "║  TRAINING                                                        ║"
	@echo "║    make baselines    - Run tabular baselines (XGBoost, LR)       ║"
	@echo "║    make text-only    - Run text-only baselines                   ║"
	@echo "║    make fusion       - Run fusion experiments                    ║"
	@echo "║    make gru          - Run GRU temporal models                   ║"
	@echo "║    make delta        - Run with delta features                   ║"
	@echo "║    make reasoning    - Run enhanced reasoning features           ║"
	@echo "║    make aligner      - Run alignment window comparison           ║"
	@echo "║                                                                  ║"
	@echo "║  EVALUATION                                                      ║"
	@echo "║    make eval         - Run all evaluations                       ║"
	@echo "║    make calibration  - Evaluate model calibration                ║"
	@echo "║    make ablation     - Run note ablation study                   ║"
	@echo "║                                                                  ║"
	@echo "║  VERIFICATION                                                    ║"
	@echo "║    make verify       - Verify data and results                   ║"
	@echo "║    make check-leakage - Check for data leakage                   ║"
	@echo "║                                                                  ║"
	@echo "║  PIPELINES                                                       ║"
	@echo "║    make all          - Run baselines + fusion + gru + eval       ║"
	@echo "║    make full         - Full pipeline (data + training + eval)    ║"
	@echo "║                                                                  ║"
	@echo "║  CLEANUP                                                         ║"
	@echo "║    make clean        - Clean result files                        ║"
	@echo "║    make clean-all    - Clean all data and results                ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
