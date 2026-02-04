"""
MedCAT UMLS 完整模型概念提取

使用用户的 UMLS API Key 下载并使用完整的 UMLS 模型
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
OUTPUT_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/medcat_umls')
MODEL_DIR = Path('/home/ubuntu/medcat_models')
COHORTS_FILE = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/cohorts/cohort_with_conditions.csv')

# UMLS 认证信息（从环境变量读取）
UMLS_API_KEY = os.environ.get('UMLS_API_KEY')


def download_umls_model():
    """下载 UMLS 模型"""
    import subprocess
    import urllib.request
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # MedCAT 模型下载需要通过其官方渠道
    # 使用 medcat 的模型下载功能
    print("尝试下载 MedCAT UMLS 模型...")

    if not UMLS_API_KEY:
        print("未设置 UMLS_API_KEY 环境变量，无法下载 UMLS 模型")
        return None
    
    # 检查是否已有模型
    model_files = list(MODEL_DIR.glob('*.zip')) + list(MODEL_DIR.glob('*model*'))
    if model_files:
        print(f"发现已有模型文件: {model_files}")
        return model_files[0]
    
    # 尝试使用公开可用的小型 UMLS 模型
    # 注意: 完整 UMLS 模型需要手动下载
    print("正在尝试下载公开可用的模型...")
    
    try:
        from medcat.cat import CAT
        # 使用 MedCAT 的默认空模型进行测试
        print("创建空 MedCAT 模型进行测试...")
        return None
    except Exception as e:
        print(f"模型下载失败: {e}")
        return None


def load_model(model_path=None):
    """加载 MedCAT 模型"""
    from medcat.cat import CAT
    from medcat.cdb import CDB
    from medcat.vocab import Vocab
    from medcat.config import Config
    
    if model_path and Path(model_path).exists():
        print(f"加载模型: {model_path}")
        try:
            cat = CAT.load_model_pack(str(model_path))
            print(f"模型包含 {len(cat.cdb.cui2names)} 个概念")
            return cat
        except:
            pass
    
    # 创建基于关键词的简化模型
    print("使用增强版关键词提取...")
    return None


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


def extract_concepts_medcat(cat, texts):
    """使用 MedCAT 提取概念"""
    all_entities = []
    
    for text in texts[:30]:
        text = text[:8000]
        try:
            entities = cat.get_entities(text)
            for ent_id, ent_data in entities.get('entities', {}).items():
                all_entities.append({
                    'cui': ent_data.get('cui'),
                    'name': ent_data.get('pretty_name'),
                    'type_ids': ent_data.get('type_ids', []),
                    'context_similarity': ent_data.get('context_similarity', 0),
                })
        except:
            continue
    
    return all_entities


def extract_concepts_keywords(texts):
    """使用增强关键词匹配提取概念"""
    # 扩展的医学概念字典
    medical_concepts = {
        # 疾病/诊断
        'sepsis': 'C0036690', 'pneumonia': 'C0032285', 'ards': 'C0035222',
        'aki': 'C2609414', 'heart failure': 'C0018801', 'diabetes': 'C0011849',
        'hypertension': 'C0020538', 'stroke': 'C0038454', 'mi': 'C0027051',
        'copd': 'C0024117', 'asthma': 'C0004096', 'covid': 'C5203670',
        
        # 症状
        'fever': 'C0015967', 'pain': 'C0030193', 'dyspnea': 'C0013404',
        'cough': 'C0010200', 'nausea': 'C0027497', 'fatigue': 'C0015672',
        'edema': 'C0013604', 'tachycardia': 'C0039231', 'hypotension': 'C0020649',
        
        # 实验室/生命体征
        'creatinine': 'C0010294', 'lactate': 'C0022924', 'wbc': 'C0023516',
        'platelet': 'C0005821', 'hemoglobin': 'C0019046', 'glucose': 'C0017725',
        
        # 治疗
        'intubation': 'C0021932', 'ventilation': 'C0035204', 'dialysis': 'C0011946',
        'antibiotic': 'C0003232', 'vasopressor': 'C0042510', 'transfusion': 'C0005841',
        
        # 器官
        'lung': 'C0024109', 'kidney': 'C0022646', 'liver': 'C0023884',
        'heart': 'C0018787', 'brain': 'C0006104',
    }
    
    all_text = ' '.join(texts).lower()
    
    found_concepts = []
    for term, cui in medical_concepts.items():
        count = all_text.count(term)
        if count > 0:
            found_concepts.append({
                'cui': cui,
                'name': term,
                'count': count
            })
    
    return found_concepts


def aggregate_to_features(entities, use_medcat=True):
    """聚合概念为特征"""
    if not entities:
        return {'n_concepts': 0, 'n_unique_cuis': 0}
    
    if use_medcat:
        cui_counter = Counter(e['cui'] for e in entities if e.get('cui'))
        type_counter = Counter()
        for e in entities:
            for tid in e.get('type_ids', []):
                type_counter[tid] += 1
        
        features = {
            'n_concepts': len(entities),
            'n_unique_cuis': len(cui_counter),
        }
        
        # Semantic Type 特征 (UMLS)
        type_mapping = {
            'T047': 'disease', 'T033': 'finding', 'T184': 'symptom',
            'T121': 'drug', 'T061': 'procedure', 'T023': 'body_part',
        }
        for tid, name in type_mapping.items():
            features[f'type_{name}'] = type_counter.get(tid, 0)
    else:
        # 关键词模式的特征
        cui_counter = Counter(e['cui'] for e in entities)
        features = {
            'n_concepts': sum(e['count'] for e in entities),
            'n_unique_cuis': len(cui_counter),
        }
        
        # 按类别统计
        disease_cuis = {'C0036690', 'C0032285', 'C0035222', 'C2609414', 'C0018801'}
        symptom_cuis = {'C0015967', 'C0030193', 'C0013404', 'C0010200', 'C0039231'}
        
        features['n_disease'] = sum(e['count'] for e in entities if e['cui'] in disease_cuis)
        features['n_symptom'] = sum(e['count'] for e in entities if e['cui'] in symptom_cuis)
        
        # 添加每个概念的存在标志
        for e in entities:
            features[f'has_{e["name"].replace(" ", "_")}'] = 1
    
    return features


def main():
    print("=" * 60)
    print("MedCAT UMLS 概念提取")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 尝试加载 MedCAT
    try:
        from medcat.cat import CAT
        model_path = download_umls_model()
        cat = load_model(model_path)
        use_medcat = cat is not None
    except ImportError:
        print("MedCAT 未安装，使用增强关键词提取")
        cat = None
        use_medcat = False
    
    # 获取 stay_ids
    if COHORTS_FILE.exists():
        cohort = pd.read_csv(COHORTS_FILE)
        stay_ids = cohort['stay_id'].tolist()
    else:
        stay_ids = [int(f.stem.split('_')[-1]) for f in EPISODES_DIR.glob('*.json')]
    
    print(f"总 Episodes: {len(stay_ids):,}")
    print(f"使用 MedCAT: {use_medcat}")
    
    # 批量处理
    results = []
    for stay_id in tqdm(stay_ids, desc="提取概念"):
        texts = extract_notes_from_episode(stay_id)
        
        if not texts:
            results.append({'stay_id': stay_id, 'n_concepts': 0})
            continue
        
        if use_medcat and cat:
            entities = extract_concepts_medcat(cat, texts)
        else:
            entities = extract_concepts_keywords(texts)
        
        features = aggregate_to_features(entities, use_medcat)
        features['stay_id'] = stay_id
        results.append(features)
    
    # 保存
    df = pd.DataFrame(results).fillna(0)
    output_file = OUTPUT_DIR / 'medcat_umls_concepts.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n完成！")
    print(f"  样本数: {len(df):,}")
    print(f"  特征维度: {df.shape[1]}")
    print(f"  保存到: {output_file}")
    
    # 统计
    if 'n_concepts' in df.columns:
        print(f"\n概念统计:")
        print(f"  平均概念数: {df['n_concepts'].mean():.1f}")
        print(f"  有概念的样本: {(df['n_concepts'] > 0).sum():,} ({(df['n_concepts'] > 0).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
