"""
从 LLM 标注提取规则并扩展到全部数据
基于 DeepSeek 标注的 key_evidence 构建规则系统

策略：
1. 从 LLM 标注的 pattern-evidence-category 关系中学习规则
2. 对每个 pattern，提取其 SUPPORTIVE/CONTRADICTORY 的典型 evidence
3. 用规则匹配扩展到未标注数据
"""

import pandas as pd
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# 路径配置
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from config import PROCESSED_DIR, TEMPORAL_ALIGNMENT_DIR

# 输入输出
LLM_ANNOTATIONS_FILE = PROCESSED_DIR / 'pattern_annotations' / 'annotated_samples_deepseek.csv'
ALIGNMENT_FILE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment_core3000.csv'
OUTPUT_FILE = PROCESSED_DIR / 'pattern_annotations' / 'expanded_annotations_rules.csv'

# ==========================================
# 1. 提取 Pattern-Evidence 规则
# ==========================================

def extract_rules_from_llm(llm_df: pd.DataFrame) -> Dict:
    """从 LLM 标注中提取 pattern-evidence 规则"""
    
    rules = defaultdict(lambda: {
        'supportive_keywords': Counter(),
        'contradictory_keywords': Counter(),
        'supportive_patterns': [],
        'contradictory_patterns': []
    })
    
    for _, row in llm_df.iterrows():
        pattern = row.get('pattern_name', '')
        category = row.get('annotation_category', '')
        evidence_str = row.get('annotation_key_evidence', '[]')
        
        if not pattern or not category or category not in ['SUPPORTIVE', 'CONTRADICTORY']:
            continue
        
        # 解析 key_evidence
        try:
            evidences = json.loads(evidence_str) if evidence_str else []
        except:
            evidences = []
        
        # 提取关键词
        for ev in evidences:
            if ev and len(ev) > 3:
                ev_lower = ev.lower().strip()
                if category == 'SUPPORTIVE':
                    rules[pattern]['supportive_keywords'][ev_lower] += 1
                else:
                    rules[pattern]['contradictory_keywords'][ev_lower] += 1
    
    # 筛选高频规则（至少出现2次）
    filtered_rules = {}
    for pattern, rule_data in rules.items():
        filtered_rules[pattern] = {
            'supportive': [kw for kw, cnt in rule_data['supportive_keywords'].items() if cnt >= 2],
            'contradictory': [kw for kw, cnt in rule_data['contradictory_keywords'].items() if cnt >= 2]
        }
    
    return filtered_rules


# ==========================================
# 2. 规则匹配
# ==========================================

def apply_rules(alignment_df: pd.DataFrame, rules: Dict, llm_annotations: pd.DataFrame) -> pd.DataFrame:
    """应用规则到对齐数据"""
    
    # 创建 LLM 标注的索引用于优先使用
    llm_index = set()
    for _, row in llm_annotations.iterrows():
        key = (row.get('stay_id'), row.get('pattern_name'), str(row.get('note_type', '')))
        llm_index.add(key)
    
    results = []
    
    for idx, row in alignment_df.iterrows():
        stay_id = row.get('stay_id')
        pattern = row.get('pattern_name', '')
        note_type = str(row.get('note_type', ''))
        note_text = str(row.get('note_text_relevant', '')).lower()
        
        key = (stay_id, pattern, note_type)
        
        # 如果已有 LLM 标注，跳过
        if key in llm_index:
            continue
        
        # 应用规则
        category = 'UNRELATED'
        confidence = 0.5
        matched_rule = ''
        
        if pattern in rules:
            rule = rules[pattern]
            
            # 检查 SUPPORTIVE 规则
            for kw in rule.get('supportive', []):
                if kw in note_text:
                    category = 'SUPPORTIVE'
                    confidence = 0.7
                    matched_rule = kw
                    break
            
            # 检查 CONTRADICTORY 规则（如果还没匹配到 SUPPORTIVE）
            if category == 'UNRELATED':
                for kw in rule.get('contradictory', []):
                    if kw in note_text:
                        category = 'CONTRADICTORY'
                        confidence = 0.7
                        matched_rule = kw
                        break
        
        results.append({
            'stay_id': stay_id,
            'pattern_name': pattern,
            'note_type': note_type,
            'annotation_category': category,
            'annotation_confidence': confidence,
            'annotation_reasoning': f'Rule matched: {matched_rule}' if matched_rule else 'No rule matched',
            'annotation_source': 'rule_expansion'
        })
        
        if idx % 50000 == 0:
            print(f"   Processed {idx:,}/{len(alignment_df):,}...")
    
    return pd.DataFrame(results)


# ==========================================
# 3. 主流程
# ==========================================

def main():
    print("=" * 60)
    print("规则提取与扩展")
    print("=" * 60)
    
    # 加载 LLM 标注
    print("\n1. 加载 LLM 标注...")
    llm_df = pd.read_csv(LLM_ANNOTATIONS_FILE)
    print(f"   LLM 标注: {len(llm_df):,} 条")
    
    # 提取规则
    print("\n2. 提取规则...")
    rules = extract_rules_from_llm(llm_df)
    
    total_rules = sum(len(r['supportive']) + len(r['contradictory']) for r in rules.values())
    print(f"   提取 {len(rules)} 个 pattern 的 {total_rules} 条规则")
    
    # 显示部分规则示例
    print("\n   规则示例:")
    for pattern in list(rules.keys())[:5]:
        r = rules[pattern]
        print(f"   {pattern}:")
        print(f"      SUPPORTIVE: {r['supportive'][:3]}")
        print(f"      CONTRADICTORY: {r['contradictory'][:3]}")
    
    # 加载对齐数据
    print("\n3. 加载对齐数据...")
    alignment_df = pd.read_csv(ALIGNMENT_FILE, low_memory=False)
    print(f"   对齐数据: {len(alignment_df):,} 条")
    
    # 应用规则
    print("\n4. 应用规则扩展...")
    expanded_df = apply_rules(alignment_df, rules, llm_df)
    print(f"   扩展标注: {len(expanded_df):,} 条")
    
    # 统计
    print("\n5. 扩展结果统计:")
    print(expanded_df['annotation_category'].value_counts())
    
    # 合并 LLM 标注和规则扩展
    print("\n6. 合并数据...")
    llm_df['annotation_source'] = 'llm_deepseek'
    combined = pd.concat([
        llm_df[['stay_id', 'pattern_name', 'note_type', 'annotation_category', 
                'annotation_confidence', 'annotation_reasoning', 'annotation_source']],
        expanded_df
    ], ignore_index=True)
    
    # 保存
    print(f"\n7. 保存到 {OUTPUT_FILE}...")
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"   总计: {len(combined):,} 条标注")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
