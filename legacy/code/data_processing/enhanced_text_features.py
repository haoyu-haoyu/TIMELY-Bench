"""
增强版文本特征提取器
从5个二值特征扩展到20+个临床特征

特征类别：
1. 生命体征描述 (6个)
2. 实验室异常 (6个)
3. 器官功能状态 (4个)
4. 干预措施 (4个)
5. 临床综合征 (4个)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
import os

from config import TEMPORAL_ALIGNMENT_DIR

# ==========================================
# 增强版临床特征定义
# ==========================================

ENHANCED_TEXT_FEATURES = {
    # === 生命体征描述 (6个) ===
    'fever_mentioned': {
        'keywords': ['fever', 'febrile', 'temperature elevated', 'hyperthermia', 'temp >38', 'temp >100'],
        'negations': ['afebrile', 'no fever', 'normothermic']
    },
    'tachycardia_mentioned': {
        'keywords': ['tachycardia', 'tachycardic', 'heart rate elevated', 'hr >100', 'rapid heart'],
        'negations': ['no tachycardia', 'heart rate normal', 'nsr']
    },
    'hypotension_mentioned': {
        'keywords': ['hypotension', 'hypotensive', 'low blood pressure', 'sbp <90', 'shock', 'pressors'],
        'negations': ['normotensive', 'hemodynamically stable', 'no hypotension']
    },
    'bradycardia_mentioned': {
        'keywords': ['bradycardia', 'bradycardic', 'slow heart rate', 'hr <60'],
        'negations': ['no bradycardia']
    },
    'tachypnea_mentioned': {
        'keywords': ['tachypnea', 'tachypneic', 'respiratory rate elevated', 'rr >20', 'rapid breathing'],
        'negations': ['no tachypnea', 'breathing normal']
    },
    'hypoxia_mentioned': {
        'keywords': ['hypoxia', 'hypoxic', 'hypoxemia', 'desaturation', 'spo2 <90', 'low oxygen'],
        'negations': ['no hypoxia', 'adequate oxygenation', 'room air']
    },

    # === 实验室异常 (6个) ===
    'elevated_creatinine': {
        'keywords': ['creatinine elevated', 'elevated creatinine', 'creatinine rise', 'cr elevated', 'rising cr'],
        'negations': ['creatinine normal', 'cr stable', 'normal renal']
    },
    'abnormal_wbc': {
        'keywords': ['leukocytosis', 'leukopenia', 'wbc elevated', 'elevated wbc', 'low wbc', 'bandemia'],
        'negations': ['wbc normal', 'normal white count']
    },
    'elevated_lactate': {
        'keywords': ['lactate elevated', 'elevated lactate', 'lactic acidosis', 'high lactate'],
        'negations': ['lactate normal', 'lactate cleared']
    },
    'elevated_bilirubin': {
        'keywords': ['bilirubin elevated', 'hyperbilirubinemia', 'jaundice', 'icterus'],
        'negations': ['bilirubin normal', 'no jaundice']
    },
    'hyperkalemia': {
        'keywords': ['hyperkalemia', 'potassium elevated', 'elevated potassium', 'high k+', 'k+ >5'],
        'negations': ['potassium normal', 'k+ normal']
    },
    'thrombocytopenia': {
        'keywords': ['thrombocytopenia', 'low platelets', 'platelets low', 'plt <100'],
        'negations': ['platelets normal', 'plt normal']
    },

    # === 器官功能状态 (4个) ===
    'respiratory_failure': {
        'keywords': ['respiratory failure', 'ards', 'acute respiratory distress', 'intubated', 'ventilator'],
        'negations': ['no respiratory failure', 'extubated']
    },
    'renal_failure': {
        'keywords': ['renal failure', 'aki', 'acute kidney injury', 'dialysis', 'crrt', 'kidney failure'],
        'negations': ['no aki', 'renal function normal']
    },
    'cardiac_dysfunction': {
        'keywords': ['heart failure', 'cardiomyopathy', 'low ef', 'chf', 'cardiac arrest'],
        'negations': ['ef normal', 'no heart failure']
    },
    'hepatic_dysfunction': {
        'keywords': ['liver failure', 'hepatic failure', 'transaminitis', 'elevated ast', 'elevated alt'],
        'negations': ['liver function normal', 'lfts normal']
    },

    # === 干预措施 (4个) ===
    'on_vasopressors': {
        'keywords': ['vasopressor', 'norepinephrine', 'levophed', 'vasopressin', 'phenylephrine', 'dopamine'],
        'negations': ['off pressors', 'weaned off vasopressor']
    },
    'on_mechanical_ventilation': {
        'keywords': ['mechanical ventilation', 'intubated', 'ventilator', 'peep', 'fio2'],
        'negations': ['extubated', 'room air', 'off ventilator']
    },
    'received_transfusion': {
        'keywords': ['transfusion', 'prbc', 'packed red blood cells', 'blood products', 'ffp'],
        'negations': []
    },
    'on_dialysis': {
        'keywords': ['dialysis', 'hemodialysis', 'crrt', 'cvvh', 'renal replacement'],
        'negations': ['no dialysis', 'off dialysis']
    },

    # === 临床综合征 (4个) ===
    'sepsis_documented': {
        'keywords': ['sepsis', 'septic', 'severe sepsis', 'septic shock', 'sirs'],
        'negations': ['no sepsis', 'sepsis ruled out']
    },
    'shock_documented': {
        'keywords': ['shock', 'cardiogenic shock', 'distributive shock', 'hypovolemic'],
        'negations': ['no shock', 'hemodynamically stable']
    },
    'altered_mental_status': {
        'keywords': ['altered mental status', 'ams', 'confusion', 'delirium', 'encephalopathy', 'obtunded'],
        'negations': ['alert', 'oriented', 'no ams', 'mental status normal']
    },
    'coagulopathy': {
        'keywords': ['coagulopathy', 'dic', 'elevated inr', 'bleeding', 'hemorrhage'],
        'negations': ['no coagulopathy', 'inr normal']
    },

    # === 原有特征 (5个，保持兼容) ===
    'pneumonia': {
        'keywords': ['pneumonia', 'pna', 'consolidation', 'infiltrate'],
        'negations': ['no pneumonia', 'clear lungs']
    },
    'edema': {
        'keywords': ['edema', 'pulmonary edema', 'fluid overload', 'volume overload'],
        'negations': ['no edema', 'euvolemic']
    },
    'infection': {
        'keywords': ['infection', 'infectious', 'bacteremia', 'uti', 'cellulitis'],
        'negations': ['no infection', 'cultures negative']
    },
    'consolidation': {
        'keywords': ['consolidation', 'lobar consolidation', 'dense opacity'],
        'negations': ['no consolidation']
    },
    'pleural_effusion': {
        'keywords': ['pleural effusion', 'effusion', 'fluid in chest'],
        'negations': ['no effusion', 'no pleural fluid']
    }
}


def extract_text_features(text: str, return_details: bool = False) -> Dict:
    """
    从临床文本中提取增强版特征

    Args:
        text: 临床笔记文本
        return_details: 是否返回匹配详情

    Returns:
        特征字典 {feature_name: 0/1}
    """
    if pd.isna(text) or not text:
        # 返回全0特征
        return {name: 0 for name in ENHANCED_TEXT_FEATURES}

    text_lower = str(text).lower()
    features = {}
    details = {}

    for feature_name, config in ENHANCED_TEXT_FEATURES.items():
        keywords = config['keywords']
        negations = config.get('negations', [])

        # 检查是否有否定关键词
        has_negation = any(neg.lower() in text_lower for neg in negations)

        # 检查是否有肯定关键词
        has_keyword = any(kw.lower() in text_lower for kw in keywords)

        # 如果有否定词，特征为0；否则看是否有关键词
        if has_negation:
            features[feature_name] = 0
            if return_details:
                details[feature_name] = 'negated'
        elif has_keyword:
            features[feature_name] = 1
            if return_details:
                # 找到匹配的关键词
                matched = [kw for kw in keywords if kw.lower() in text_lower]
                details[feature_name] = matched[0] if matched else 'matched'
        else:
            features[feature_name] = 0
            if return_details:
                details[feature_name] = 'not_found'

    if return_details:
        return features, details
    return features


def extract_features_for_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    为DataFrame中的所有文本提取特征

    Args:
        df: 输入DataFrame
        text_column: 文本列名

    Returns:
        添加了特征列的DataFrame
    """
    print(f"Extracting enhanced text features from {len(df)} samples...")

    # 提取特征
    features_list = []
    for _, row in df.iterrows():
        text = row.get(text_column, '')
        features = extract_text_features(text)
        features_list.append(features)

    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)

    # 合并
    result = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # 打印统计
    print(f"\n   [Feature Statistics]")
    total = len(result)
    for col in features_df.columns:
        positive = features_df[col].sum()
        pct = positive / total * 100
        if pct > 0:  # 只打印有正样本的特征
            print(f"   {col}: {positive} ({pct:.1f}%)")

    return result


def aggregate_patient_text_features(
    alignment_df: pd.DataFrame,
    patient_id_col: str = 'stay_id'
) -> pd.DataFrame:
    """
    将多条对齐记录聚合为每个患者一行的特征

    聚合策略：
    - 对于每个二值特征，使用 max (任一笔记提及即为1)
    """
    print(f"Aggregating text features by patient...")

    # 先提取特征
    df_with_features = extract_features_for_dataframe(
        alignment_df,
        text_column='note_text_relevant'
    )

    # 获取特征列
    feature_cols = list(ENHANCED_TEXT_FEATURES.keys())

    # 按患者聚合
    agg_dict = {col: 'max' for col in feature_cols if col in df_with_features.columns}

    patient_features = df_with_features.groupby(patient_id_col).agg(agg_dict).reset_index()

    print(f"   Aggregated to {len(patient_features)} patients")

    return patient_features


# ==========================================
# 主函数
# ==========================================

def main():
    print("Enhanced Text Feature Extractor")
    print("=" * 60)
    print(f"Total features defined: {len(ENHANCED_TEXT_FEATURES)}")
    print()

    # 列出所有特征
    print("[Feature Categories]")
    categories = {
        '生命体征描述': ['fever_mentioned', 'tachycardia_mentioned', 'hypotension_mentioned',
                    'bradycardia_mentioned', 'tachypnea_mentioned', 'hypoxia_mentioned'],
        '实验室异常': ['elevated_creatinine', 'abnormal_wbc', 'elevated_lactate',
                   'elevated_bilirubin', 'hyperkalemia', 'thrombocytopenia'],
        '器官功能状态': ['respiratory_failure', 'renal_failure', 'cardiac_dysfunction', 'hepatic_dysfunction'],
        '干预措施': ['on_vasopressors', 'on_mechanical_ventilation', 'received_transfusion', 'on_dialysis'],
        '临床综合征': ['sepsis_documented', 'shock_documented', 'altered_mental_status', 'coagulopathy'],
        '原有特征': ['pneumonia', 'edema', 'infection', 'consolidation', 'pleural_effusion']
    }

    for cat_name, features in categories.items():
        print(f"\n{cat_name} ({len(features)}个):")
        for f in features:
            print(f"   - {f}")

    # 测试提取
    print("\n" + "=" * 60)
    print("[Test Extraction]")

    test_texts = [
        "Patient presented with fever and tachycardia, started on vasopressors for septic shock.",
        "Labs notable for elevated creatinine, AKI stage 2, started on CRRT.",
        "Chest X-ray shows bilateral infiltrates, intubated for respiratory failure.",
        "Patient is afebrile, hemodynamically stable, no signs of infection."
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:60]}...")
        features, details = extract_text_features(text, return_details=True)
        positive_features = [k for k, v in features.items() if v == 1]
        print(f"   Positive features: {positive_features}")

    # 处理实际数据
    print("\n" + "=" * 60)
    print("[Process Alignment Data]")

    alignment_file = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment.csv'
    if alignment_file.exists():
        alignment_df = pd.read_csv(alignment_file)
        print(f"Loaded {len(alignment_df)} alignments")

        # 只处理有相关文本的样本
        has_text = alignment_df[alignment_df['note_text_relevant'].str.len() > 10].copy()
        print(f"Samples with relevant text: {len(has_text)}")

        # 提取特征
        df_with_features = extract_features_for_dataframe(has_text, 'note_text_relevant')

        # 保存
        output_path = TEMPORAL_ALIGNMENT_DIR / 'alignment_with_enhanced_features.csv'
        df_with_features.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

        # 聚合到患者级别
        patient_features = aggregate_patient_text_features(has_text)
        patient_output = TEMPORAL_ALIGNMENT_DIR / 'patient_text_features.csv'
        patient_features.to_csv(patient_output, index=False)
        print(f"Saved: {patient_output}")

        # 打印聚合后的统计
        print("\n[Patient-Level Feature Statistics]")
        feature_cols = list(ENHANCED_TEXT_FEATURES.keys())
        for col in feature_cols:
            if col in patient_features.columns:
                positive = patient_features[col].sum()
                total = len(patient_features)
                pct = positive / total * 100
                if pct > 0:
                    print(f"   {col}: {positive}/{total} patients ({pct:.1f}%)")
    else:
        print(f"Alignment file not found: {alignment_file}")

    print("\nEnhanced Text Feature Extraction Complete!")


if __name__ == "__main__":
    main()
