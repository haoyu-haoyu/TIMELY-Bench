"""
全量智能规则匹配脚本
在 EC2 上使用完整的 47GB 对齐数据运行

策略：
1. 加载完整的 temporal_textual_alignment.csv (47GB)
2. 应用智能规则匹配
3. 生成覆盖所有 74,829 stay_id 的标注
"""

import pandas as pd
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# 路径配置
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent

ALIGNMENT_FILE = PROJECT_ROOT / 'data' / 'processed' / 'temporal_alignment' / 'temporal_textual_alignment.csv'
LLM_ANNOTATION_FILE = PROJECT_ROOT / 'data' / 'processed' / 'pattern_annotations' / 'annotated_samples_deepseek.csv'
OUTPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / 'pattern_annotations' / 'smart_annotations_full.csv'

# ==========================================
# 医学术语同义词库
# ==========================================

MEDICAL_SYNONYMS = {
    'tachycardia': ['tachycardia', 'tachy', 'tachycardic', 'rapid heart rate', 
                    'elevated hr', 'hr elevated', 'heart rate elevated',
                    'pulse rapid', 'fast heart rate', 'racing heart'],
    'bradycardia': ['bradycardia', 'brady', 'bradycardic', 'slow heart rate',
                    'low heart rate', 'hr low'],
    'hypotension': ['hypotension', 'hypotensive', 'low blood pressure', 'low bp',
                    'bp low', 'sbp low', 'map low', 'pressure low'],
    'hypertension': ['hypertension', 'hypertensive', 'high blood pressure', 
                     'elevated bp', 'bp elevated', 'htn'],
    'respiratory_distress': ['respiratory distress', 'resp distress', 'sob', 
                             'shortness of breath', 'dyspnea', 'labored breathing',
                             'work of breathing', 'wob', 'increased wob',
                             'hypoxic', 'hypoxia', 'desaturation', 'desat'],
    'tachypnea': ['tachypnea', 'tachypneic', 'rapid breathing', 'rr elevated',
                  'respiratory rate elevated', 'fast breathing'],
    'altered_consciousness': ['altered mental status', 'ams', 'confusion',
                              'confused', 'lethargy', 'lethargic', 'obtunded',
                              'unresponsive', 'somnolent', 'drowsy', 'gcs',
                              'decreased loc', 'altered loc', 'encephalopathy'],
    'oliguria': ['oliguria', 'oliguric', 'low urine output', 'poor urine output',
                 'decreased urine', 'uop low', 'minimal urine', 'anuria',
                 'no urine', 'urine output decreased'],
    'aki': ['aki', 'acute kidney injury', 'renal failure', 'creatinine elevated',
            'cr elevated', 'renal dysfunction', 'kidney dysfunction'],
    'fever': ['fever', 'febrile', 'temp elevated', 'temperature elevated',
              'hyperthermia', 'pyrexia', 'high temperature'],
    'hypothermia': ['hypothermia', 'hypothermic', 'low temperature',
                    'temp low', 'cold'],
    'anemia': ['anemia', 'anemic', 'low hemoglobin', 'hgb low', 'hb low',
               'low hematocrit', 'hct low'],
    'leukocytosis': ['leukocytosis', 'elevated wbc', 'wbc elevated', 'high wbc'],
    'hyperkalemia': ['hyperkalemia', 'potassium elevated', 'k elevated',
                     'high potassium', 'elevated k'],
    'acidosis': ['acidosis', 'acidotic', 'low ph', 'ph low', 'metabolic acidosis',
                 'respiratory acidosis', 'lactate elevated', 'high lactate'],
    'atrial_fibrillation': ['atrial fibrillation', 'afib', 'a-fib', 'af',
                            'irregular rhythm', 'irregularly irregular'],
    '_negative': ['no ', 'not ', 'without ', 'denies ', 'negative for ',
                  'absent ', 'none ', 'resolved ', 'improved ', 'normal ']
}

# 数值范围规则
NUMERIC_RULES = {
    'tachycardia': {'field': 'hr|heart rate|pulse', 'threshold': 100, 'direction': '>'},
    'severe_tachycardia': {'field': 'hr|heart rate|pulse', 'threshold': 120, 'direction': '>'},
    'bradycardia': {'field': 'hr|heart rate|pulse', 'threshold': 60, 'direction': '<'},
    'hypotension': {'field': 'sbp|systolic|bp', 'threshold': 90, 'direction': '<'},
    'fever': {'field': 'temp|temperature', 'threshold': 38.0, 'direction': '>'},
    'hypothermia': {'field': 'temp|temperature', 'threshold': 36.0, 'direction': '<'},
    'tachypnea': {'field': 'rr|resp rate|respiratory rate', 'threshold': 20, 'direction': '>'},
    'spo2_low': {'field': 'spo2|oxygen sat|o2 sat', 'threshold': 92, 'direction': '<'},
}


class SmartRuleMatcher:
    """智能规则匹配器"""
    
    def __init__(self):
        self.synonyms = MEDICAL_SYNONYMS
        self.numeric_rules = NUMERIC_RULES
        self.learned_rules = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """预编译正则表达式"""
        self.compiled_patterns = {}
        
        for pattern, synonyms in self.synonyms.items():
            if pattern.startswith('_'):
                continue
            regex_parts = [re.escape(syn) for syn in synonyms]
            self.compiled_patterns[pattern] = re.compile(
                r'(?:' + '|'.join(regex_parts) + r')',
                re.IGNORECASE
            )
    
    def learn_from_llm(self, llm_df: pd.DataFrame):
        """从 LLM 标注中学习规则"""
        print("   从 LLM 标注学习规则...")
        
        pattern_evidence = defaultdict(lambda: {
            'supportive': Counter(),
            'contradictory': Counter()
        })
        
        for _, row in llm_df.iterrows():
            pattern = row.get('pattern_name', '')
            category = row.get('annotation_category', '')
            evidence_str = row.get('annotation_key_evidence', '[]')
            
            if not pattern or category not in ['SUPPORTIVE', 'CONTRADICTORY']:
                continue
            
            try:
                evidences = json.loads(evidence_str) if evidence_str else []
            except:
                evidences = []
            
            for ev in evidences:
                if ev and len(ev) > 5:
                    ev_lower = ev.lower().strip()
                    if len(ev_lower) > 5:
                        pattern_evidence[pattern][category.lower()][ev_lower] += 1
        
        for pattern, evidence in pattern_evidence.items():
            self.learned_rules[pattern] = {
                'supportive': [e for e, c in evidence['supportive'].items() if c >= 2],
                'contradictory': [e for e, c in evidence['contradictory'].items() if c >= 2]
            }
        
        total = sum(len(r['supportive']) + len(r['contradictory']) 
                    for r in self.learned_rules.values())
        print(f"   学习到 {len(self.learned_rules)} 个 pattern 的 {total} 条规则")
    
    def match(self, pattern: str, note_text: str) -> Tuple[str, float, str]:
        """智能匹配"""
        text_lower = note_text.lower()
        
        has_negation = any(neg in text_lower for neg in self.synonyms['_negative'])
        
        # 同义词匹配
        if pattern in self.compiled_patterns:
            match = self.compiled_patterns[pattern].search(text_lower)
            if match:
                matched_term = match.group()
                start_pos = max(0, match.start() - 30)
                context_before = text_lower[start_pos:match.start()]
                
                if any(neg in context_before for neg in self.synonyms['_negative']):
                    return ('CONTRADICTORY', 0.75, f'Negated: {matched_term}')
                else:
                    return ('SUPPORTIVE', 0.8, f'Synonym: {matched_term}')
        
        # 数值匹配
        if pattern in self.numeric_rules:
            rule = self.numeric_rules[pattern]
            field_pattern = rule['field']
            number_regex = rf'(?:{field_pattern})\s*[:=]?\s*(\d+(?:\.\d+)?)'
            
            matches = re.findall(number_regex, text_lower, re.IGNORECASE)
            for value_str in matches:
                try:
                    value = float(value_str)
                    threshold = rule['threshold']
                    direction = rule['direction']
                    
                    if direction == '>' and value > threshold:
                        return ('SUPPORTIVE', 0.85, f'Numeric: {value}>{threshold}')
                    elif direction == '<' and value < threshold:
                        return ('SUPPORTIVE', 0.85, f'Numeric: {value}<{threshold}')
                except:
                    pass
        
        # LLM 学习的规则
        if pattern in self.learned_rules:
            rules = self.learned_rules[pattern]
            
            for evidence in rules.get('supportive', []):
                if evidence in text_lower:
                    if has_negation and any(neg + evidence[:10] in text_lower 
                                           for neg in self.synonyms['_negative']):
                        continue
                    return ('SUPPORTIVE', 0.7, f'LLM: {evidence[:30]}')
            
            for evidence in rules.get('contradictory', []):
                if evidence in text_lower:
                    return ('CONTRADICTORY', 0.7, f'LLM neg: {evidence[:30]}')
        
        return ('UNRELATED', 0.5, 'No match')


def main():
    print("=" * 60)
    print("全量智能规则匹配 (47GB 对齐数据)")
    print("=" * 60)
    
    # 初始化匹配器
    matcher = SmartRuleMatcher()
    
    # 加载 LLM 标注
    print("\n1. 加载 LLM 标注...")
    if LLM_ANNOTATION_FILE.exists():
        llm_df = pd.read_csv(LLM_ANNOTATION_FILE)
        print(f"   LLM 标注: {len(llm_df):,} 条")
        matcher.learn_from_llm(llm_df)
    else:
        print("   LLM 标注文件不存在，跳过")
    
    # 加载对齐数据（分块处理以节省内存）
    print("\n2. 处理对齐数据（分块读取）...")
    
    results = []
    stats = Counter()
    chunk_size = 1000000  # 每次读取 100 万行
    total_processed = 0
    
    for chunk in tqdm(pd.read_csv(ALIGNMENT_FILE, chunksize=chunk_size, low_memory=False),
                      desc="Processing chunks"):
        for _, row in chunk.iterrows():
            stay_id = row.get('stay_id')
            pattern = row.get('pattern_name', '')
            note_text = str(row.get('note_text_relevant', ''))
            
            category, confidence, reason = matcher.match(pattern, note_text)
            stats[category] += 1
            
            results.append({
                'stay_id': stay_id,
                'pattern_name': pattern,
                'note_type': row.get('note_type', ''),
                'annotation_category': category,
                'annotation_confidence': confidence,
                'annotation_reasoning': reason,
                'annotation_source': 'smart_rules_full'
            })
        
        total_processed += len(chunk)
        
        # 定期打印进度
        if total_processed % 5000000 == 0:
            print(f"   已处理: {total_processed:,}")
            print(f"   当前统计: {dict(stats)}")
    
    # 创建结果 DataFrame
    print("\n3. 保存结果...")
    result_df = pd.DataFrame(results)
    
    # 合并 LLM 标注（优先使用）
    if LLM_ANNOTATION_FILE.exists():
        llm_df['annotation_source'] = 'llm_deepseek'
        result_df = pd.concat([
            llm_df[['stay_id', 'pattern_name', 'note_type', 'annotation_category', 
                    'annotation_confidence', 'annotation_reasoning', 'annotation_source']],
            result_df
        ], ignore_index=True)
    
    result_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n4. 统计汇总:")
    print(f"   总条目: {len(result_df):,}")
    print(f"   唯一 stay_id: {result_df['stay_id'].nunique():,}")
    print(result_df['annotation_category'].value_counts())
    
    supportive_pct = stats['SUPPORTIVE'] / sum(stats.values()) * 100 if stats else 0
    print(f"\n   SUPPORTIVE 占比: {supportive_pct:.1f}%")


if __name__ == "__main__":
    main()
