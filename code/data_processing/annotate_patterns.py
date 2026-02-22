"""
LLM-based Pattern Annotation (Multi-Note Version)
使用LLM对pattern-text alignments进行标注

支持多种笔记类型的智能标注：
- 根据笔记类型调整提示词
- 针对不同笔记类型优化关键词匹配
- 添加alignment_quality感知

标注类别：
- SUPPORTIVE: 文本明确支持/确认模式
- CONTRADICTORY: 文本与模式矛盾
- AMBIGUOUS: 文本提及但不明确
- UNRELATED: 文本未提及相关内容

支持的LLM后端：
- DeepSeek API
- OpenAI API
- 本地Ollama
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
import os
import time
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests

from config import TEMPORAL_ALIGNMENT_DIR, PROCESSED_DIR

# ==========================================
# 配置
# ==========================================
# Debug-only sampling file for quick manual/LLM inspection.
# Canonical release input is results/llm_annotations/llm_annotation_set.csv.
ALIGNMENT_DEBUG_SAMPLES_FILE = TEMPORAL_ALIGNMENT_DIR / 'llm_annotation_debug_samples.csv'
LEGACY_ALIGNMENT_SAMPLES_FILE = TEMPORAL_ALIGNMENT_DIR / 'llm_annotation_samples.csv'
OUTPUT_DIR = PROCESSED_DIR / 'pattern_annotations'

# API配置（根据实际情况修改）
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# ==========================================
# 1. 标注类别定义
# ==========================================

ANNOTATION_CATEGORIES = {
    "SUPPORTIVE": "The clinical note explicitly mentions, confirms, or documents the detected physiological pattern",
    "CONTRADICTORY": "The clinical note explicitly contradicts or negates the detected pattern",
    "AMBIGUOUS": "The clinical note mentions related concepts but doesn't clearly confirm or deny the pattern",
    "UNRELATED": "The clinical note doesn't discuss anything relevant to this physiological pattern"
}

# ==========================================
# 2. 多笔记类型感知的Prompt模板
# ==========================================

# 基础提示词模板
ANNOTATION_PROMPT_BASE = """You are a clinical expert annotating the relationship between detected physiological patterns and clinical notes.

## DETECTED PATTERN
- Pattern Name: {pattern_name}
- Pattern Value: {pattern_value}
- Severity: {pattern_severity}
- Disease Context: {pattern_disease}
- Pattern Description: {pattern_description}

## NOTE TYPE: {note_type}
{note_type_guidance}

## CLINICAL NOTE EXCERPT
{note_text}

## TASK
Classify the relationship between the detected pattern and the clinical note into ONE of these categories:

1. **SUPPORTIVE** - The note explicitly mentions or confirms the pattern
   Example: "Patient tachycardic to 110 bpm" confirms tachycardia pattern
   Example: "Creatinine elevated to 2.1" confirms creatinine_elevated pattern

2. **CONTRADICTORY** - The note explicitly contradicts the pattern
   Example: "Patient afebrile" contradicts fever pattern
   Example: "Normal renal function" contradicts AKI pattern

3. **AMBIGUOUS** - The note mentions related concepts but is unclear
   Example: "Temperature within normal limits" is ambiguous for fever
   Example: "Monitor renal function" is ambiguous for AKI

4. **UNRELATED** - The note doesn't discuss this pattern at all
   Example: A note about chest X-ray findings when the pattern is hyperkalemia

## RESPONSE FORMAT
Respond with ONLY a JSON object:
{{"category": "SUPPORTIVE|CONTRADICTORY|AMBIGUOUS|UNRELATED", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

# 笔记类型特定的指导
NOTE_TYPE_GUIDANCE = {
    'discharge': """This is a DISCHARGE SUMMARY - a comprehensive retrospective document written at patient discharge.
- Contains detailed documentation of the entire hospital stay
- May describe the pattern in the context of overall clinical course
- Look for explicit mentions of the physiological condition being treated
- Higher chance of finding supportive documentation for significant patterns""",

    'nursing': """This is a NURSING NOTE - real-time clinical assessment by nursing staff.
- Contains immediate observations and vital sign interpretations
- May use abbreviations (HR, BP, RR, T, SpO2)
- Look for direct observations of the patient's current state
- Very reliable for vital sign patterns (tachycardia, fever, hypotension)""",

    'lab_comment': """This is a LAB COMMENT - notes from laboratory analysis.
- Contains abnormal value flags and laboratory observations
- May mention hemolysis, sample quality, or critical values
- Highly reliable for lab-based patterns (creatinine, potassium, lactate)
- Look for "abnormal", "critical", or comparison to reference ranges""",

    'radiology': """This is a RADIOLOGY NOTE - imaging interpretation by radiologists.
- Contains findings from X-rays, CT scans, or other imaging
- Best for respiratory patterns (infiltrates, edema, consolidation)
- May not directly mention physiological values
- Look for imaging findings that correlate with the pattern"""
}

# 默认指导（向后兼容）
DEFAULT_NOTE_GUIDANCE = """This clinical note may contain relevant information about the detected pattern.
Look for explicit mentions or implicit indications of the physiological condition."""

# ==========================================
# 3. Pattern描述映射
# ==========================================

PATTERN_DESCRIPTIONS = {
    # Sepsis
    "fever": "Body temperature > 38.3°C indicating infection/inflammation",
    "hypothermia": "Body temperature < 36°C indicating severe illness",
    "tachycardia": "Heart rate > 90 bpm indicating stress/compensation",
    "tachypnea": "Respiratory rate > 20/min indicating respiratory distress",
    "hypotension": "Systolic blood pressure < 90 mmHg indicating shock",
    "map_low": "Mean arterial pressure < 70 mmHg indicating poor perfusion",
    "hypoxemia": "Oxygen saturation < 94% indicating oxygenation failure",
    "lactate_elevated": "Lactate > 2 mmol/L indicating tissue hypoperfusion",
    "thrombocytopenia": "Platelet count < 150,000 indicating coagulopathy",
    "leukocytosis": "White blood cell count > 12,000 indicating infection",
    "leukopenia": "White blood cell count < 4,000 indicating immunosuppression",
    
    # AKI
    "creatinine_elevated": "Creatinine > 1.2 mg/dL indicating renal dysfunction",
    "creatinine_severe": "Creatinine ≥ 4.0 mg/dL indicating severe AKI (Stage 3)",
    "creatinine_rise_acute": "Creatinine rise ≥ 0.3 mg/dL in 48h indicating acute kidney injury",
    "bun_elevated": "BUN > 20 mg/dL indicating renal dysfunction",
    "bun_severe": "BUN > 40 mg/dL indicating severe renal dysfunction",
    "oliguria": "Urine output < 500 mL/day indicating renal failure",
    "hyperkalemia": "Potassium > 5.5 mEq/L indicating electrolyte disturbance",
    "metabolic_acidosis": "Bicarbonate < 22 mEq/L indicating acid-base disturbance",
    
    # ARDS
    "hypoxemia_mild": "PaO2/FiO2 ratio < 300 indicating mild ARDS",
    "hypoxemia_moderate": "PaO2/FiO2 ratio < 200 indicating moderate ARDS",
    "hypoxemia_severe": "PaO2/FiO2 ratio ≤ 100 indicating severe ARDS",
    "spo2_low": "SpO2 < 90% indicating severe oxygen desaturation",
    "respiratory_distress": "Respiratory rate > 30/min indicating respiratory failure",
    
    # Critical
    "bradycardia": "Heart rate < 60 bpm indicating conduction abnormality",
    "severe_tachycardia": "Heart rate > 120 bpm indicating severe stress",
    "hypertensive_crisis": "Systolic BP > 180 mmHg indicating hypertensive emergency",
    "anemia": "Hemoglobin < 10 g/dL indicating blood loss or production failure",
    "severe_anemia": "Hemoglobin < 7 g/dL requiring transfusion",
    "altered_consciousness": "GCS < 14 indicating neurological impairment",
    "coma": "GCS ≤ 8 indicating severe neurological impairment",
}

# ==========================================
# 4. LLM调用函数
# ==========================================

def call_deepseek_api(prompt: str, max_retries: int = 3) -> Optional[str]:
    """调用DeepSeek API"""
    
    if not DEEPSEEK_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"   API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    
    return None

def call_openai_api(prompt: str, max_retries: int = 3) -> Optional[str]:
    """调用OpenAI API"""
    
    if not OPENAI_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENAI_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"   API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    
    return None

def parse_llm_response(response: str) -> Dict:
    """解析LLM响应"""
    
    if not response:
        return {"category": "UNRELATED", "confidence": 0.0, "reasoning": "API error"}
    
    # 尝试提取JSON
    try:
        # 移除markdown代码块
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        result = json.loads(response.strip())
        
        # 验证category
        category = result.get('category', 'UNRELATED').upper()
        if category not in ANNOTATION_CATEGORIES:
            category = 'UNRELATED'
        
        return {
            "category": category,
            "confidence": float(result.get('confidence', 0.5)),
            "reasoning": result.get('reasoning', '')
        }
    except:
        # 尝试简单解析
        response_upper = response.upper()
        if 'SUPPORTIVE' in response_upper:
            return {"category": "SUPPORTIVE", "confidence": 0.7, "reasoning": response[:100]}
        elif 'CONTRADICTORY' in response_upper:
            return {"category": "CONTRADICTORY", "confidence": 0.7, "reasoning": response[:100]}
        elif 'AMBIGUOUS' in response_upper:
            return {"category": "AMBIGUOUS", "confidence": 0.7, "reasoning": response[:100]}
        else:
            return {"category": "UNRELATED", "confidence": 0.5, "reasoning": response[:100]}

# ==========================================
# 5. 批量标注（多线程版本）
# ==========================================

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 线程锁用于进度打印
_print_lock = threading.Lock()
_counter = {'processed': 0}

def process_single_sample(row_data: Tuple[int, pd.Series], api_type: str, total: int) -> Dict:
    """处理单个样本（供多线程调用）- 支持多笔记类型"""
    idx, row = row_data

    # 获取pattern描述
    pattern_name = row['pattern_name']
    pattern_desc = PATTERN_DESCRIPTIONS.get(pattern_name, f"Pattern: {pattern_name}")

    # 获取笔记类型和对应的指导
    note_type = row.get('note_type', 'unknown')
    note_type_guidance = NOTE_TYPE_GUIDANCE.get(note_type, DEFAULT_NOTE_GUIDANCE)

    # 构建prompt - 使用新的多笔记类型感知模板
    prompt = ANNOTATION_PROMPT_BASE.format(
        pattern_name=pattern_name,
        pattern_value=row.get('pattern_value', 'N/A'),
        pattern_severity=row.get('pattern_severity', 'N/A'),
        pattern_disease=row.get('pattern_disease', 'N/A'),
        pattern_description=pattern_desc,
        note_type=note_type.upper(),
        note_type_guidance=note_type_guidance,
        note_text=str(row.get('note_text_relevant', ''))[:500]
    )
    
    # 调用API
    if api_type == 'deepseek':
        response = call_deepseek_api(prompt)
    elif api_type == 'openai':
        response = call_openai_api(prompt)
    else:
        response = None
    
    # 解析响应
    annotation = parse_llm_response(response)
    
    # 更新进度
    with _print_lock:
        _counter['processed'] += 1
        if _counter['processed'] % 20 == 0:
            print(f"   Processed {_counter['processed']}/{total}...")
    
    # 返回结果 - 包含笔记类型和对齐质量
    return {
        'stay_id': row['stay_id'],
        'pattern_name': pattern_name,
        'pattern_hour': row.get('pattern_hour'),
        'pattern_value': row.get('pattern_value'),
        'pattern_severity': row.get('pattern_severity'),
        'note_type': note_type,  # 新增
        'alignment_quality': row.get('alignment_quality', 'unknown'),  # 新增
        'note_text': str(row.get('note_text_relevant', ''))[:300],
        'annotation_category': annotation['category'],
        'annotation_confidence': annotation['confidence'],
        'annotation_reasoning': annotation['reasoning']
    }

def annotate_samples(
    samples_df: pd.DataFrame,
    output_dir: str,
    api_type: str = 'deepseek',
    max_samples: Optional[int] = None,
    max_workers: int = 5,  # 并发线程数
    use_cache: bool = True  # 支持断点续传
) -> pd.DataFrame:
    """
    批量标注样本（多线程版本，支持断点续传）

    Args:
        use_cache: 如果为True，会检查已标注的样本并跳过
    """

    os.makedirs(output_dir, exist_ok=True)

    # 断点续传：加载已有的标注结果
    cache_path = os.path.join(output_dir, f'annotated_samples_{api_type}.csv')
    annotated_keys = set()

    if use_cache and os.path.exists(cache_path):
        existing_df = pd.read_csv(cache_path)
        # 构建已标注样本的唯一键
        annotated_keys = set(
            existing_df.apply(
                lambda r: f"{r['stay_id']}_{r['pattern_name']}_{r.get('pattern_hour', 0)}_{r.get('note_type', '')}",
                axis=1
            )
        )
        print(f"Cache found: {len(annotated_keys)} samples already annotated")
    else:
        existing_df = pd.DataFrame()

    # 过滤掉已标注的样本
    if annotated_keys:
        samples_df = samples_df.copy()
        samples_df['_key'] = samples_df.apply(
            lambda r: f"{r['stay_id']}_{r['pattern_name']}_{r.get('pattern_hour', 0)}_{r.get('note_type', '')}",
            axis=1
        )
        samples_df = samples_df[~samples_df['_key'].isin(annotated_keys)]
        samples_df = samples_df.drop(columns=['_key'])
        print(f"Remaining samples to annotate: {len(samples_df)}")

    if len(samples_df) == 0:
        print("All samples already annotated!")
        return existing_df

    if max_samples:
        samples_df = samples_df.head(max_samples)

    print(f"\nAnnotating {len(samples_df)} samples using {api_type}...")
    print(f"   Using {max_workers} concurrent workers")
    
    # 重置计数器
    _counter['processed'] = 0
    
    # 准备数据
    row_data_list = list(samples_df.iterrows())
    total = len(row_data_list)
    
    results = []
    failed_samples = []
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_sample, row_data, api_type, total): row_data[0]
            for row_data in row_data_list
        }
        
        # 收集结果
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"   Error processing sample {idx}: {e}")
                failed_samples.append(idx)
    
    # 按原始顺序排序
    results_df = pd.DataFrame(results)

    # 合并新结果与已有缓存
    if len(existing_df) > 0 and len(results_df) > 0:
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
        print(f"Merged with cache: {len(existing_df)} + {len(results)} = {len(results_df)} total")
    elif len(existing_df) > 0:
        results_df = existing_df

    # 保存结果
    output_path = os.path.join(output_dir, f'annotated_samples_{api_type}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(results_df)} total samples)")
    
    # 保存失败的样本
    if failed_samples:
        failed_path = os.path.join(output_dir, f'failed_samples_{api_type}.txt')
        with open(failed_path, 'w') as f:
            for idx in failed_samples:
                f.write(f"{idx}\n")
        print(f"{len(failed_samples)} failed samples saved to: {failed_path}")

    # 统计
    print("\nAnnotation Statistics:")
    category_counts = results_df['annotation_category'].value_counts()
    for cat, count in category_counts.items():
        pct = count / len(results_df) * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")

    # 按笔记类型统计
    if 'note_type' in results_df.columns:
        print("\nStatistics by Note Type:")
        for note_type in results_df['note_type'].unique():
            type_df = results_df[results_df['note_type'] == note_type]
            print(f"\n   [{note_type.upper()}] ({len(type_df)} samples)")
            type_cats = type_df['annotation_category'].value_counts()
            for cat, count in type_cats.items():
                pct = count / len(type_df) * 100
                print(f"      {cat}: {count} ({pct:.1f}%)")

    return results_df

# ==========================================
# 6. 规则基础标注（备用方案）- 支持多笔记类型
# ==========================================

# 针对不同笔记类型的关键词增强
NOTE_TYPE_KEYWORDS_ENHANCED = {
    'discharge': {
        'fever': ['fever', 'febrile', 'temperature elevated', 'hyperthermia', 'infectious', 'septic'],
        'tachycardia': ['tachycardia', 'tachycardic', 'rapid heart', 'heart rate elevated', 'hr 1'],
        'hypotension': ['hypotension', 'hypotensive', 'shock', 'pressors', 'vasopressors', 'low bp'],
        'creatinine': ['creatinine', 'renal', 'aki', 'acute kidney injury', 'kidney', 'dialysis'],
        'lactate': ['lactate', 'lactic acidosis', 'elevated lactate', 'hypoperfusion'],
        'anemia': ['anemia', 'transfusion', 'prbc', 'blood loss', 'hemorrhage'],
    },
    'nursing': {
        'fever': ['temp', 'temperature', 'febrile', 'chills', 't:'],
        'tachycardia': ['hr', 'heart rate', 'pulse', 'tachycardic', 'rapid'],
        'hypotension': ['bp', 'blood pressure', 'sbp', 'map', 'hypotensive', 'pressors'],
        'altered_consciousness': ['confused', 'oriented', 'gcs', 'responsive', 'alert', 'lethargic'],
        'oliguria': ['urine output', 'uop', 'foley', 'void', 'oliguric'],
    },
    'lab_comment': {
        'creatinine': ['creatinine', 'elevated', 'abnormal', 'critical', 'rising'],
        'hyperkalemia': ['potassium', 'k+', 'elevated', 'hemolysis', 'critical'],
        'lactate': ['lactate', 'elevated', 'critical', 'abnormal'],
        'anemia': ['hemoglobin', 'hgb', 'hematocrit', 'low', 'critical', 'transfuse'],
    },
    'radiology': {
        'hypoxemia': ['infiltrate', 'opacity', 'consolidation', 'edema', 'ards'],
        'respiratory': ['bilateral', 'infiltrates', 'diffuse', 'pneumonia', 'effusion'],
    }
}

def rule_based_annotation(row) -> Dict:
    """基于规则的简单标注（当API不可用时使用）- 支持笔记类型感知"""

    note_text = str(row.get('note_text_relevant', '')).lower()
    pattern_name = row['pattern_name'].lower()
    note_type = row.get('note_type', 'unknown')

    if not note_text or len(note_text) < 10:
        return {"category": "UNRELATED", "confidence": 0.9, "reasoning": "No relevant text"}

    # 优先使用笔记类型特定的关键词
    type_keywords = NOTE_TYPE_KEYWORDS_ENHANCED.get(note_type, {})

    # 检查确认性关键词
    confirm_patterns = {
        'fever': ['fever', 'febrile', 'temp elevated', 'hyperthermia'],
        'tachycardia': ['tachycardia', 'tachycardic', 'rapid heart', 'hr elevated'],
        'hypotension': ['hypotension', 'hypotensive', 'low bp', 'shock'],
        'hypoxemia': ['hypoxia', 'hypoxemic', 'desaturation', 'low oxygen'],
        'creatinine': ['creatinine elevated', 'renal dysfunction', 'aki', 'kidney injury'],
        'anemia': ['anemia', 'anemic', 'low hemoglobin', 'transfusion'],
        'oliguria': ['oliguria', 'low urine', 'decreased output'],
    }

    # 检查否定性关键词
    negate_patterns = {
        'fever': ['afebrile', 'no fever', 'normothermic'],
        'tachycardia': ['normal heart rate', 'no tachycardia', 'nsr'],
        'hypotension': ['normotensive', 'stable bp', 'hemodynamically stable'],
        'hypoxemia': ['adequate oxygenation', 'normal saturation', 'room air'],
    }

    # 先检查笔记类型特定的关键词（更高可信度）
    for pattern_key in type_keywords:
        if pattern_key in pattern_name:
            for kw in type_keywords[pattern_key]:
                if kw in note_text:
                    return {"category": "SUPPORTIVE", "confidence": 0.85, "reasoning": f"Found type-specific keyword: {kw}"}

    # 匹配通用确认性关键词
    for pattern_key, keywords in confirm_patterns.items():
        if pattern_key in pattern_name:
            for kw in keywords:
                if kw in note_text:
                    return {"category": "SUPPORTIVE", "confidence": 0.8, "reasoning": f"Found keyword: {kw}"}

    # 匹配否定性关键词
    for pattern_key, keywords in negate_patterns.items():
        if pattern_key in pattern_name:
            for kw in keywords:
                if kw in note_text:
                    return {"category": "CONTRADICTORY", "confidence": 0.8, "reasoning": f"Found negation: {kw}"}

    # 默认
    if len(note_text) > 50:
        return {"category": "AMBIGUOUS", "confidence": 0.5, "reasoning": "Related text but unclear"}
    else:
        return {"category": "UNRELATED", "confidence": 0.6, "reasoning": "No clear match"}

def annotate_samples_rule_based(samples_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """使用规则进行标注 - 支持多笔记类型"""

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRule-based annotation for {len(samples_df)} samples...")

    results = []

    for _, row in samples_df.iterrows():
        annotation = rule_based_annotation(row)
        note_type = row.get('note_type', 'unknown')

        result = {
            'stay_id': row['stay_id'],
            'pattern_name': row['pattern_name'],
            'pattern_hour': row.get('pattern_hour'),
            'pattern_value': row.get('pattern_value'),
            'pattern_severity': row.get('pattern_severity'),
            'note_type': note_type,  # 新增
            'alignment_quality': row.get('alignment_quality', 'unknown'),  # 新增
            'note_text': str(row.get('note_text_relevant', ''))[:300],
            'annotation_category': annotation['category'],
            'annotation_confidence': annotation['confidence'],
            'annotation_reasoning': annotation['reasoning']
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    # 保存
    output_path = os.path.join(output_dir, 'annotated_samples_rules.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # 统计
    print("\nAnnotation Statistics:")
    category_counts = results_df['annotation_category'].value_counts()
    for cat, count in category_counts.items():
        pct = count / len(results_df) * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")

    # 按笔记类型统计
    if 'note_type' in results_df.columns:
        print("\nStatistics by Note Type:")
        for note_type in results_df['note_type'].unique():
            type_df = results_df[results_df['note_type'] == note_type]
            print(f"\n   [{note_type.upper()}] ({len(type_df)} samples)")
            type_cats = type_df['annotation_category'].value_counts()
            for cat, count in type_cats.items():
                pct = count / len(type_df) * 100
                print(f"      {cat}: {count} ({pct:.1f}%)")

    return results_df

# ==========================================
# Main
# ==========================================

def main():
    print("Pattern Annotation Pipeline")
    print("=" * 60)
    
    # 加载样本
    samples_path = ALIGNMENT_DEBUG_SAMPLES_FILE
    if not samples_path.exists() and LEGACY_ALIGNMENT_SAMPLES_FILE.exists():
        samples_path = LEGACY_ALIGNMENT_SAMPLES_FILE
        print(f"[WARN] Using legacy debug sample file: {samples_path}")

    if not samples_path.exists():
        print(f"Samples file not found: {ALIGNMENT_DEBUG_SAMPLES_FILE}")
        return
    
    samples_df = pd.read_csv(samples_path)
    print(f"Loaded {len(samples_df)} samples from {samples_path.name}")
    
    # 检查API可用性
    if DEEPSEEK_API_KEY:
        print("\nDeepSeek API key found")
        annotate_samples(samples_df, OUTPUT_DIR, api_type='deepseek', max_samples=None)
    elif OPENAI_API_KEY:
        print("\nOpenAI API key found")
        annotate_samples(samples_df, OUTPUT_DIR, api_type='openai', max_samples=None)
    else:
        print("\nNo API keys found, using rule-based annotation")
        annotate_samples_rule_based(samples_df, OUTPUT_DIR)
    
    print("\nAnnotation Complete!")

if __name__ == "__main__":
    main()
