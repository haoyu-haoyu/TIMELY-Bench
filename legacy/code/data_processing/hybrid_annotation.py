"""
Hybrid Annotation Strategy
混合标注策略：规则全量 + LLM核心验证

策略：
1. 先用 rule_based_annotation() 为全量对齐数据打标签 (7.4万条)
2. 针对核心Episode (3,000个Stay ID)，用LLM进行二次验证
3. 合并结果，生成最终的 annotated_samples_all.csv

优点：
- 全量数据有标注覆盖
- 核心数据有LLM质量保证
- 断点续传支持
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
from typing import Optional, Set

from config import (
    TEMPORAL_ALIGNMENT_DIR, PROCESSED_DIR, ROOT_DIR,
    MERGE_OUTPUT_DIR
)

from annotate_patterns import (
    annotate_samples,
    annotate_samples_rule_based,
    rule_based_annotation,
    DEEPSEEK_API_KEY,
    OPENAI_API_KEY
)

# 配置
ALIGNMENT_FILE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment.csv'
OUTPUT_DIR = PROCESSED_DIR / 'pattern_annotations'
CORE_EPISODES_FILE = ROOT_DIR / 'episodes_core' / 'core_episode_selection.csv'
COHORT_FILE = MERGE_OUTPUT_DIR / 'cohort_final.csv'


def load_core_stay_ids() -> Set[int]:
    """加载核心Episode的Stay IDs"""

    # 优先使用core_episode_selection.csv
    if CORE_EPISODES_FILE.exists():
        core_df = pd.read_csv(CORE_EPISODES_FILE)
        stay_ids = set(core_df['stay_id'].astype(int).tolist())
        print(f"Loaded {len(stay_ids)} core Stay IDs from {CORE_EPISODES_FILE}")
        return stay_ids

    # 备选：使用cohort中有完整数据的前3000个患者
    if COHORT_FILE.exists():
        cohort_df = pd.read_csv(COHORT_FILE)

        # 优先选择有sepsis或aki的患者（更临床相关）
        priority = cohort_df[
            (cohort_df['has_sepsis_final'] == 1) |
            (cohort_df['has_aki_final'] == 1)
        ]

        if len(priority) >= 3000:
            stay_ids = set(priority['stay_id'].head(3000).astype(int).tolist())
        else:
            # 补充其他患者
            remaining = cohort_df[~cohort_df['stay_id'].isin(priority['stay_id'])]
            needed = 3000 - len(priority)
            additional = remaining['stay_id'].head(needed).astype(int).tolist()
            stay_ids = set(priority['stay_id'].astype(int).tolist()) | set(additional)

        print(f"Generated {len(stay_ids)} core Stay IDs from cohort")
        return stay_ids

    print("Warning: No core Stay IDs source found!")
    return set()


def run_rule_based_full_annotation(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: 规则标注全量数据
    """
    print("\n" + "=" * 70)
    print("STEP 1: Rule-based Annotation for ALL Alignments")
    print("=" * 70)

    # 检查缓存
    cache_path = OUTPUT_DIR / 'annotated_samples_rules_full.csv'
    if cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        print(f"Cache found: {len(cached_df)} rule-based annotations")

        # 检查是否需要补充新的对齐
        cached_keys = set(
            cached_df.apply(
                lambda r: f"{r['stay_id']}_{r['pattern_name']}_{r.get('pattern_hour', 0)}_{r.get('note_type', '')}",
                axis=1
            )
        )

        alignment_df['_key'] = alignment_df.apply(
            lambda r: f"{r['stay_id']}_{r['pattern_name']}_{r.get('pattern_hour', 0)}_{r.get('note_type', '')}",
            axis=1
        )

        new_alignments = alignment_df[~alignment_df['_key'].isin(cached_keys)]

        if len(new_alignments) == 0:
            print("All alignments already annotated with rules!")
            return cached_df

        print(f"Found {len(new_alignments)} new alignments to annotate")
        alignment_df = new_alignments.drop(columns=['_key'])

    # 执行规则标注
    print(f"\nAnnotating {len(alignment_df)} alignments with rules...")

    results = []
    total = len(alignment_df)

    for i, (_, row) in enumerate(alignment_df.iterrows()):
        if (i + 1) % 5000 == 0:
            print(f"   Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

        annotation = rule_based_annotation(row)

        result = {
            'stay_id': row['stay_id'],
            'pattern_name': row['pattern_name'],
            'pattern_hour': row.get('pattern_hour', 0),
            'pattern_value': row.get('pattern_value'),
            'pattern_severity': row.get('pattern_severity', 'moderate'),
            'note_type': row.get('note_type', 'unknown'),
            'alignment_quality': row.get('alignment_quality', 'unknown'),
            'note_text': str(row.get('note_text_relevant', ''))[:300],
            'annotation_category': annotation['category'],
            'annotation_confidence': annotation['confidence'],
            'annotation_reasoning': annotation['reasoning'],
            'annotation_source': 'rule'  # 标记来源
        }
        results.append(result)

    new_df = pd.DataFrame(results)

    # 合并缓存
    if cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        new_df = pd.concat([cached_df, new_df], ignore_index=True)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(cache_path, index=False)
    print(f"\nSaved {len(new_df)} rule-based annotations: {cache_path}")

    # 统计
    print("\n[Rule-based Annotation Statistics]")
    category_counts = new_df['annotation_category'].value_counts()
    for cat, count in category_counts.items():
        pct = count / len(new_df) * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")

    return new_df


def run_llm_core_verification(
    alignment_df: pd.DataFrame,
    core_stay_ids: Set[int],
    rule_annotations: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 2: LLM验证核心数据
    """
    print("\n" + "=" * 70)
    print("STEP 2: LLM Verification for Core Episodes")
    print("=" * 70)

    # 筛选核心Episode的对齐
    core_alignments = alignment_df[alignment_df['stay_id'].isin(core_stay_ids)]
    print(f"Core alignments: {len(core_alignments)} (from {len(core_stay_ids)} Stay IDs)")

    if len(core_alignments) == 0:
        print("No core alignments found!")
        return pd.DataFrame()

    # 确保有足够的样本覆盖核心Stay IDs
    core_coverage = core_alignments['stay_id'].nunique()
    print(f"Core Stay ID coverage: {core_coverage}/{len(core_stay_ids)}")

    # 检查API可用性
    if not DEEPSEEK_API_KEY and not OPENAI_API_KEY:
        print("\nNo API keys available, skipping LLM verification")
        return pd.DataFrame()

    api_type = 'deepseek' if DEEPSEEK_API_KEY else 'openai'
    print(f"\nUsing {api_type} API for verification")

    # 调用LLM标注（带断点续传）
    llm_annotations = annotate_samples(
        core_alignments,
        str(OUTPUT_DIR),
        api_type=api_type,
        max_samples=None,  # 无限制
        max_workers=10,    # 增加并发
        use_cache=True     # 断点续传
    )

    if len(llm_annotations) > 0:
        llm_annotations['annotation_source'] = 'llm'  # 标记来源

    print(f"\nLLM verified {len(llm_annotations)} core annotations")

    return llm_annotations


def merge_annotations(
    rule_annotations: pd.DataFrame,
    llm_annotations: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 3: 合并标注结果

    策略：LLM标注优先（更准确），规则标注作为补充
    """
    print("\n" + "=" * 70)
    print("STEP 3: Merging Annotations")
    print("=" * 70)

    if len(llm_annotations) == 0:
        print("No LLM annotations, using rule-based only")
        final_df = rule_annotations.copy()
    else:
        # 构建LLM标注的唯一键
        llm_keys = set(
            llm_annotations.apply(
                lambda r: f"{r['stay_id']}_{r['pattern_name']}_{r.get('pattern_hour', 0)}_{r.get('note_type', '')}",
                axis=1
            )
        )

        # 从规则标注中移除已有LLM标注的条目
        rule_annotations['_key'] = rule_annotations.apply(
            lambda r: f"{r['stay_id']}_{r['pattern_name']}_{r.get('pattern_hour', 0)}_{r.get('note_type', '')}",
            axis=1
        )

        rule_only = rule_annotations[~rule_annotations['_key'].isin(llm_keys)]
        rule_only = rule_only.drop(columns=['_key'])

        # 合并
        final_df = pd.concat([llm_annotations, rule_only], ignore_index=True)

        print(f"LLM annotations: {len(llm_annotations)}")
        print(f"Rule-only annotations: {len(rule_only)}")

    print(f"Total merged: {len(final_df)}")

    # 保存最终结果
    output_path = OUTPUT_DIR / 'annotated_samples_all.csv'
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved final annotations: {output_path}")

    return final_df


def print_final_statistics(final_df: pd.DataFrame, core_stay_ids: Set[int]):
    """打印最终统计"""

    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    print(f"\nTotal annotations: {len(final_df)}")
    print(f"Unique Stay IDs: {final_df['stay_id'].nunique()}")
    print(f"Unique patterns: {final_df['pattern_name'].nunique()}")

    # 来源统计
    if 'annotation_source' in final_df.columns:
        print("\n[By Annotation Source]")
        source_counts = final_df['annotation_source'].value_counts()
        for src, count in source_counts.items():
            pct = count / len(final_df) * 100
            print(f"   {src}: {count} ({pct:.1f}%)")

    # 类别统计
    print("\n[By Category]")
    category_counts = final_df['annotation_category'].value_counts()
    for cat, count in category_counts.items():
        pct = count / len(final_df) * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")

    # 核心覆盖率
    core_annotations = final_df[final_df['stay_id'].isin(core_stay_ids)]
    core_covered = core_annotations['stay_id'].nunique()
    print(f"\n[Core Episode Coverage]")
    print(f"   Core Stay IDs covered: {core_covered}/{len(core_stay_ids)} ({core_covered/len(core_stay_ids)*100:.1f}%)")
    print(f"   Core annotations: {len(core_annotations)}")

    # LLM验证的核心数据
    if 'annotation_source' in final_df.columns:
        llm_core = core_annotations[core_annotations['annotation_source'] == 'llm']
        print(f"   LLM-verified core: {len(llm_core)}")

    # 按笔记类型
    if 'note_type' in final_df.columns:
        print("\n[By Note Type]")
        for nt, count in final_df['note_type'].value_counts().items():
            pct = count / len(final_df) * 100
            print(f"   {nt}: {count} ({pct:.1f}%)")


def main():
    print("=" * 70)
    print("Hybrid Annotation Pipeline")
    print("Rule-based (Full) + LLM (Core Verification)")
    print("=" * 70)

    # 加载对齐数据
    if not ALIGNMENT_FILE.exists():
        print(f"Alignment file not found: {ALIGNMENT_FILE}")
        return

    alignment_df = pd.read_csv(ALIGNMENT_FILE)
    print(f"\nLoaded {len(alignment_df)} alignments")
    print(f"Unique Stay IDs: {alignment_df['stay_id'].nunique()}")

    # 加载核心Stay IDs
    core_stay_ids = load_core_stay_ids()

    # Step 1: 规则标注全量
    rule_annotations = run_rule_based_full_annotation(alignment_df)

    # Step 2: LLM验证核心
    llm_annotations = run_llm_core_verification(
        alignment_df, core_stay_ids, rule_annotations
    )

    # Step 3: 合并结果
    final_df = merge_annotations(rule_annotations, llm_annotations)

    # 打印统计
    print_final_statistics(final_df, core_stay_ids)

    print("\n" + "=" * 70)
    print("Hybrid Annotation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
