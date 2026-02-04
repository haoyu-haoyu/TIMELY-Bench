"""
TIMELY-Bench v2.0 Snakemake Pipeline
====================================

可复现的自动化工作流

使用方法:
    # 安装Snakemake
    pip install snakemake

    # 运行完整pipeline
    snakemake --cores 4 all

    # 只运行基线实验
    snakemake --cores 4 baselines

    # 干运行（查看将执行的步骤）
    snakemake -n all
"""

# 配置
configfile: "config.yaml"

# 默认目标
rule all:
    input:
        "results/benchmark_results/fusion_results.csv",
        "results/benchmark_results/benchmark_results_full.csv"

# ==========================================
# Step 1: 数据分割
# ==========================================
rule generate_splits:
    input:
        cohort="data/processed/merge_output/cohort_final.csv"
    output:
        train="data/splits/train.csv",
        val="data/splits/val.csv",
        test="data/splits/test.csv"
    shell:
        "python code/data_processing/generate_data_splits.py"

# ==========================================
# Step 2: XGBoost 基线
# ==========================================
rule run_xgboost_baselines:
    input:
        features_6h="data/processed/data_windows/window_6h/features_aggregated.csv",
        features_12h="data/processed/data_windows/window_12h/features_aggregated.csv",
        features_24h="data/processed/data_windows/window_24h/features_aggregated.csv",
        cohort="data/processed/merge_output/cohort_final.csv"
    output:
        "results/benchmark_results/benchmark_results_full.csv"
    shell:
        "python code/baselines/run_xgboost_baselines.py"

# ==========================================
# Step 3: Fusion 基线
# ==========================================
rule run_fusion_baselines:
    input:
        features_24h="data/processed/data_windows/window_24h/features_aggregated.csv",
        llm_features="data/llm_features/llm_features_deepseek.csv",
        cohort="data/processed/merge_output/cohort_final.csv"
    output:
        "results/benchmark_results/fusion_results.csv"
    shell:
        "python code/baselines/run_fusion_baselines.py"

# ==========================================
# Step 4: GRU 时序模型 (可选)
# ==========================================
rule run_gru_baselines:
    input:
        temporal="data/processed/data_windows/window_24h/features_temporal.npy",
        cohort="data/processed/merge_output/cohort_final.csv"
    output:
        "results/benchmark_results/gru_results.csv"
    shell:
        "python code/baselines/run_temporal_gru.py"

# ==========================================
# 汇总规则
# ==========================================
rule baselines:
    input:
        "results/benchmark_results/benchmark_results_full.csv",
        "results/benchmark_results/fusion_results.csv"

rule clean:
    shell:
        "rm -rf results/benchmark_results/*.csv"
