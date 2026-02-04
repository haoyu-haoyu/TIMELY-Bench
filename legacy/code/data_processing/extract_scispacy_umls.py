"""
SciSpacy UMLS 概念提取

使用 SciSpacy 的 UMLS 实体链接器从临床文本提取 UMLS 概念
这是 MedCAT 的替代方案，无需 UMLS 许可证
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 配置
EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')
OUTPUT_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/umls_concepts')
COHORTS_FILE = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/cohorts/cohort_with_conditions.csv')


def install_scispacy():
    """安装 SciSpacy 及其模型"""
    import subprocess
    
    print("安装 SciSpacy...")
    subprocess.run(['pip', 'install', 'scispacy', '-q'], check=True)
    
    # 安装医学 NER 模型
    print("安装医学 NER 模型 (en_core_sci_sm)...")
    subprocess.run(['pip', 'install', 
        'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz', 
        '-q'], check=True)
    
    # 安装 UMLS 实体链接器
    print("安装 UMLS 实体链接器...")
    subprocess.run(['pip', 'install', 
        'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz',
        '-q'], check=True)
    
    print("安装完成")


def load_model():
    """加载 SciSpacy 模型"""
    import spacy
    
    try:
        nlp = spacy.load("en_core_sci_sm")
        print("加载 en_core_sci_sm 模型")
    except:
        nlp = spacy.load("en_ner_bc5cdr_md")
        print("加载 en_ner_bc5cdr_md 模型")
    
    # 尝试添加 UMLS 链接器
    try:
        from scispacy.linking import EntityLinker
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        print("已添加 UMLS 实体链接器")
    except Exception as e:
        print(f"无法添加 UMLS 链接器: {e}")
    
    return nlp


def extract_notes_from_episode(stay_id):
    """从 Episode 提取临床文本"""
    ep_file = EPISODES_DIR / f'TIMELY_v2_{stay_id}.json'
    if not ep_file.exists():
        return []
    
    try:
        with open(ep_file) as f:
            ep = json.load(f)
        
        texts = []
        notes = ep.get('clinical_text', {}).get('notes', [])
        
        for note in notes:
            if isinstance(note, dict):
                text = note.get('text_full') or note.get('text_relevant', '')
            else:
                text = str(note)
            if text and len(text.strip()) > 20:
                texts.append(text)
        
        return texts
    except:
        return []


def extract_umls_concepts(nlp, texts):
    """提取 UMLS 概念"""
    all_entities = []
    
    for text in texts[:30]:  # 每个 Episode 最多 30 条笔记
        text = text[:8000]  # 限制文本长度
        
        try:
            doc = nlp(text)
            
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'cui': None,
                    'umls_name': None,
                    'semantic_types': []
                }
                
                # 如果有 UMLS 链接器
                if hasattr(ent, '_') and hasattr(ent._, 'kb_ents'):
                    if ent._.kb_ents:
                        top_link = ent._.kb_ents[0]
                        entity_info['cui'] = top_link[0]
                        entity_info['confidence'] = top_link[1]
                
                all_entities.append(entity_info)
        except:
            continue
    
    return all_entities


def aggregate_to_features(entities):
    """聚合实体为特征"""
    if not entities:
        return {
            'n_entities': 0,
            'n_unique_cuis': 0,
            'n_disease': 0,
            'n_chemical': 0,
        }
    
    # 实体类型统计
    label_counter = Counter(e['label'] for e in entities)
    cui_set = set(e['cui'] for e in entities if e.get('cui'))
    
    # 基于 BC5CDR 标签 (DISEASE, CHEMICAL)
    features = {
        'n_entities': len(entities),
        'n_unique_cuis': len(cui_set),
        'n_disease': label_counter.get('DISEASE', 0),
        'n_chemical': label_counter.get('CHEMICAL', 0),
    }
    
    # 添加其他标签
    for label, count in label_counter.items():
        if label not in ['DISEASE', 'CHEMICAL']:
            features[f'n_{label.lower()}'] = count
    
    # 医学关键词检测 (基于实体文本)
    entity_texts = ' '.join(e['text'].lower() for e in entities)
    keywords = [
        'infection', 'fever', 'sepsis', 'pain', 'failure',
        'cardiac', 'respiratory', 'renal', 'hepatic', 'diabetes',
        'hypertension', 'pneumonia', 'hemorrhage', 'shock'
    ]
    for kw in keywords:
        features[f'kw_{kw}'] = 1 if kw in entity_texts else 0
    
    return features


def main():
    print("=" * 60)
    print("SciSpacy UMLS 概念提取")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 安装/加载模型
    try:
        import scispacy
    except ImportError:
        install_scispacy()
    
    nlp = load_model()
    
    # 获取 stay_ids
    if COHORTS_FILE.exists():
        cohort = pd.read_csv(COHORTS_FILE)
        stay_ids = cohort['stay_id'].tolist()
    else:
        stay_ids = [int(f.stem.split('_')[-1]) for f in EPISODES_DIR.glob('*.json')]
    
    print(f"总 Episodes: {len(stay_ids):,}")
    
    # 批量处理
    results = []
    for stay_id in tqdm(stay_ids, desc="提取 UMLS 概念"):
        texts = extract_notes_from_episode(stay_id)
        if not texts:
            results.append({'stay_id': stay_id, 'n_entities': 0})
            continue
        
        entities = extract_umls_concepts(nlp, texts)
        features = aggregate_to_features(entities)
        features['stay_id'] = stay_id
        results.append(features)
    
    # 保存
    df = pd.DataFrame(results).fillna(0)
    df.to_csv(OUTPUT_DIR / 'umls_concepts.csv', index=False)
    
    print(f"\n完成！")
    print(f"  样本数: {len(df):,}")
    print(f"  特征维度: {df.shape[1]}")
    print(f"  保存到: {OUTPUT_DIR}")
    
    print(f"\n概念统计:")
    if 'n_entities' in df.columns:
        print(f"  平均实体数: {df['n_entities'].mean():.1f}")
    if 'n_disease' in df.columns:
        print(f"  平均疾病数: {df['n_disease'].mean():.1f}")
    if 'n_chemical' in df.columns:
        print(f"  平均药物/化学物质数: {df['n_chemical'].mean():.1f}")


if __name__ == "__main__":
    main()
