"""
MedCAT UMLS 概念提取
从临床笔记中提取医学术语（UMLS CUI）

使用方法：先下载预训练模型，然后运行此脚本
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# 配置
EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')
OUTPUT_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/text_concepts')
BATCH_SIZE = 100

# 简化版：使用 spacy NER 作为替代方案
# 因为完整的 MedCAT UMLS 模型需要单独下载 (~1-5GB)
USE_SPACY_NER = True
NEGATION_TERMS = ['no', 'without', 'denies', 'denied', 'negative for', 'rule out']


def load_spacy_model():
    """加载 spacy 模型进行实体识别"""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp


def extract_medical_entities_spacy(text, nlp):
    """使用 spacy 提取医学实体"""
    if not text or len(text) < 10:
        return []
    
    # 截断长文本
    text = text[:5000]
    
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        window = text[max(0, ent.start_char - 50):ent.start_char].lower()
        negated = any(term in window for term in NEGATION_TERMS)
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'negated': negated
        })
    
    return entities


def extract_notes_from_episode(episode_path):
    """从 Episode 文件中提取笔记文本"""
    try:
        with open(episode_path) as f:
            ep = json.load(f)
        
        stay_id = ep.get('stay_id')
        clinical_text = ep.get('clinical_text', {})
        notes = clinical_text.get('notes', [])
        
        texts = []
        for note in notes:
            if isinstance(note, dict):
                text = note.get('text_full') or note.get('text_relevant') or note.get('text', '')
            else:
                text = str(note)
            if text and len(text) > 10:
                texts.append(text)
        
        return stay_id, texts
    except Exception as e:
        return None, []


def aggregate_entities(all_entities):
    """聚合实体为特征向量"""
    if not all_entities:
        return {}
    
    # 统计实体类型频率
    label_counts = Counter()
    entity_texts = []
    
    for ent in all_entities:
        if ent.get('negated'):
            continue
        label_counts[ent['label']] += 1
        entity_texts.append(ent['text'].lower())
    
    # 统计常见医学关键词
    medical_keywords = {
        'fever': 0, 'infection': 0, 'sepsis': 0, 'pneumonia': 0,
        'heart': 0, 'cardiac': 0, 'blood': 0, 'pressure': 0,
        'kidney': 0, 'renal': 0, 'liver': 0, 'lung': 0,
        'pain': 0, 'failure': 0, 'acute': 0, 'chronic': 0,
        'diabetes': 0, 'hypertension': 0, 'oxygen': 0, 'breathing': 0
    }
    
    full_text = ' '.join(entity_texts)
    for keyword in medical_keywords:
        medical_keywords[keyword] = full_text.count(keyword)
    
    features = {
        'n_entities': len(all_entities),
        **{f'ner_{k}': v for k, v in label_counts.items()},
        **{f'med_{k}': v for k, v in medical_keywords.items()}
    }
    
    return features


def main():
    print("=" * 60)
    print("MedCAT/spaCy 概念提取")
    print("=" * 60)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("加载 spaCy 模型...")
    nlp = load_spacy_model()
    print("模型加载完成")
    
    # 获取所有 Episode 文件
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    print(f"找到 {len(episode_files):,} 个 Episode 文件")
    
    # 存储结果
    all_results = []
    
    for ep_file in tqdm(episode_files, desc="提取概念"):
        stay_id, texts = extract_notes_from_episode(ep_file)
        
        if not stay_id or not texts:
            continue
        
        # 提取所有笔记的实体
        all_entities = []
        for text in texts[:50]:  # 限制每个 Episode 最多处理 50 条笔记
            entities = extract_medical_entities_spacy(text, nlp)
            all_entities.extend(entities)
        
        # 聚合特征
        features = aggregate_entities(all_entities)
        features['stay_id'] = stay_id
        all_results.append(features)
    
    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 填充缺失列
        df = df.fillna(0)
        
        # 保存
        df.to_csv(OUTPUT_DIR / 'spacy_concepts.csv', index=False)
        
        print("\n" + "=" * 60)
        print(f"完成！")
        print(f"  提取了 {len(df):,} 个 stay_id 的概念特征")
        print(f"  特征维度: {df.shape[1]} 列")
        print(f"  保存到: {OUTPUT_DIR}")
        
        # 显示概念统计
        print("\n概念统计:")
        if 'n_entities' in df.columns:
            print(f"  平均实体数: {df['n_entities'].mean():.1f}")
        
        # 显示常见医学关键词
        med_cols = [c for c in df.columns if c.startswith('med_')]
        if med_cols:
            print("\n常见医学关键词频率:")
            for col in med_cols[:10]:
                keyword = col.replace('med_', '')
                total = df[col].sum()
                print(f"  {keyword}: {total:,.0f}")
    else:
        print("警告：未提取到任何概念")


if __name__ == "__main__":
    main()
