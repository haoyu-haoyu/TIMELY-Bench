"""
扩展LLM标注：标注剩余的样本
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import os

from config import TEMPORAL_ALIGNMENT_DIR, PROCESSED_DIR

from annotate_patterns import (
    annotate_samples,
    annotate_samples_rule_based,
    DEEPSEEK_API_KEY,
    OPENAI_API_KEY
)

ALIGNMENT_SAMPLES_FILE = TEMPORAL_ALIGNMENT_DIR / 'llm_annotation_samples.csv'
ANNOTATED_FILE = PROCESSED_DIR / 'pattern_annotations' / 'annotated_samples_deepseek.csv'
OUTPUT_DIR = PROCESSED_DIR / 'pattern_annotations'

def main():
    print("Expanding LLM Annotation")
    print("=" * 60)

    # 加载所有样本
    if not os.path.exists(ALIGNMENT_SAMPLES_FILE):
        print(f"Samples file not found: {ALIGNMENT_SAMPLES_FILE}")
        return

    all_samples = pd.read_csv(ALIGNMENT_SAMPLES_FILE)
    print(f"Total samples: {len(all_samples)}")

    # 加载已标注的样本
    if os.path.exists(ANNOTATED_FILE):
        annotated = pd.read_csv(ANNOTATED_FILE)
        print(f"Already annotated: {len(annotated)}")

        # 获取已标注的stay_id + pattern组合
        annotated_keys = set(
            annotated.apply(
                lambda r: f"{r['stay_id']}_{r['pattern_name']}_{r.get('pattern_hour', 0)}",
                axis=1
            )
        )

        # 找到未标注的样本
        all_samples['_key'] = all_samples.apply(
            lambda r: f"{r['stay_id']}_{r['pattern_name']}_{r.get('pattern_hour', 0)}",
            axis=1
        )
        remaining = all_samples[~all_samples['_key'].isin(annotated_keys)].drop(columns=['_key'])
        print(f"Remaining to annotate: {len(remaining)}")
    else:
        annotated = pd.DataFrame()
        remaining = all_samples
        print(f"No previous annotations found, starting fresh")

    if len(remaining) == 0:
        print("All samples already annotated!")
        return

    # 标注剩余样本
    print(f"\nAnnotating {len(remaining)} remaining samples...")

    if DEEPSEEK_API_KEY:
        print("Using DeepSeek API")
        new_annotations = annotate_samples(
            remaining,
            OUTPUT_DIR,
            api_type='deepseek',
            max_samples=None,  # 无限制，处理全部样本
            max_workers=5
        )
    elif OPENAI_API_KEY:
        print("Using OpenAI API")
        new_annotations = annotate_samples(
            remaining,
            OUTPUT_DIR,
            api_type='openai',
            max_samples=None,  # 无限制，处理全部样本
            max_workers=5
        )
    else:
        print("No API keys found, using rule-based annotation")
        new_annotations = annotate_samples_rule_based(remaining, OUTPUT_DIR)

    # 合并新旧标注
    if len(annotated) > 0 and len(new_annotations) > 0:
        all_annotations = pd.concat([annotated, new_annotations], ignore_index=True)

        # 保存合并后的结果
        merged_path = os.path.join(OUTPUT_DIR, 'annotated_samples_all.csv')
        all_annotations.to_csv(merged_path, index=False)
        print(f"\nSaved merged annotations: {merged_path}")
        print(f"   Total annotations: {len(all_annotations)}")

        # 打印最终统计
        print("\n" + "=" * 60)
        print("FINAL ANNOTATION STATISTICS")
        print("=" * 60)

        category_counts = all_annotations['annotation_category'].value_counts()
        total = len(all_annotations)
        for cat, count in category_counts.items():
            pct = count / total * 100
            print(f"   {cat}: {count} ({pct:.1f}%)")

        # 按笔记类型统计
        if 'note_type' in all_annotations.columns:
            print("\n[By Note Type]")
            for note_type in all_annotations['note_type'].unique():
                type_df = all_annotations[all_annotations['note_type'] == note_type]
                print(f"\n   [{note_type.upper()}] ({len(type_df)} samples)")
                type_cats = type_df['annotation_category'].value_counts()
                for cat, count in type_cats.items():
                    pct = count / len(type_df) * 100
                    print(f"      {cat}: {count} ({pct:.1f}%)")

    print("\nLLM Annotation Expansion Complete!")

if __name__ == "__main__":
    main()
