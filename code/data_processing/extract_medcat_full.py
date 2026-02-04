"""
MedCAT 医学概念提取 (使用 MedMentions 公开模型)

MedMentions 模型包含约 35,000 个 UMLS 概念，无需 UMLS 许可证
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
OUTPUT_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/medcat_concepts')
MODEL_DIR = Path('/home/ubuntu/medcat_models')


def install_medcat():
    """安装 MedCAT"""
    import subprocess
    print("安装 MedCAT...")
    subprocess.run(['pip', 'install', 'medcat', '-q'], check=True)
    print("MedCAT 安装完成")


def download_model():
    """下载 MedMentions 公开模型"""
    import subprocess
    import urllib.request
    import zipfile
    
    # MedMentions 公开模型 URL
    model_url = "https://medcat.rosalind.kcl.ac.uk/media/medmen_wstatus_2021_oct.zip"
    model_name = "medmen_wstatus_2021_oct"
    model_path = MODEL_DIR / model_name
    zip_path = MODEL_DIR / f"{model_name}.zip"
    
    if model_path.exists():
        print(f"模型已存在: {model_path}")
        return str(model_path)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"下载 MedMentions 模型 (~500MB)...")
    urllib.request.urlretrieve(model_url, str(zip_path))
    
    print("解压模型...")
    with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
        zip_ref.extractall(str(MODEL_DIR))
    
    os.remove(str(zip_path))
    print(f"模型下载完成: {model_path}")
    return str(model_path)


def load_model(model_path):
    """加载 MedCAT 模型"""
    from medcat.cat import CAT
    
    print(f"加载 MedCAT 模型...")
    cat = CAT.load_model_pack(model_path)
    print(f"模型加载完成，包含 {len(cat.cdb.cui2names)} 个概念")
    return cat


def extract_notes_from_episode(stay_id):
    """从 Episode 提取临床文本"""
    ep_file = EPISODES_DIR / f'TIMELY_v2_{stay_id}.json'
    if not ep_file.exists():
        return []
    
    try:
        with open(ep_file) as f:
            ep = json.load(f)
        
        texts = []
        clinical_text = ep.get('clinical_text', {})
        notes = clinical_text.get('notes', [])
        
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


def extract_concepts(cat, texts):
    """使用 MedCAT 提取概念"""
    all_entities = []
    
    for text in texts[:50]:  # 限制每个 Episode 最多 50 条笔记
        try:
            text = text[:10000]  # 限制文本长度
            entities = cat.get_entities(text)
            
            for ent_id, ent_data in entities.get('entities', {}).items():
                if ent_data.get('negated'):
                    continue
                all_entities.append({
                    'cui': ent_data.get('cui'),
                    'name': ent_data.get('pretty_name'),
                    'type_ids': ent_data.get('type_ids', []),
                    'context_similarity': ent_data.get('context_similarity', 0),
                })
        except:
            continue
    
    return all_entities


def aggregate_to_features(entities):
    """将实体聚合为特征"""
    if not entities:
        return {'n_concepts': 0, 'n_unique_cuis': 0}
    
    # 统计 CUI 频率
    cui_counter = Counter(e['cui'] for e in entities if e.get('cui'))
    
    # 提取概念类型
    type_counter = Counter()
    for e in entities:
        for tid in e.get('type_ids', []):
            type_counter[tid] += 1
    
    # 常见医学概念类型映射
    # UMLS Semantic Type IDs
    type_mapping = {
        'T047': 'disease',      # Disease or Syndrome
        'T033': 'finding',      # Finding
        'T184': 'symptom',      # Sign or Symptom
        'T121': 'drug',         # Pharmacologic Substance
        'T061': 'procedure',    # Therapeutic or Preventive Procedure
        'T023': 'body_part',    # Body Part
        'T046': 'pathologic',   # Pathologic Function
        'T037': 'injury',       # Injury or Poisoning
    }
    
    features = {
        'n_concepts': len(entities),
        'n_unique_cuis': len(cui_counter),
        'avg_similarity': np.mean([e.get('context_similarity', 0) for e in entities]) if entities else 0,
    }
    
    for type_id, type_name in type_mapping.items():
        features[f'type_{type_name}'] = type_counter.get(type_id, 0)
    
    return features


def main():
    print("=" * 60)
    print("MedCAT 医学概念提取 (MedMentions 模型)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 安装 MedCAT
    try:
        from medcat.cat import CAT
    except ImportError:
        install_medcat()
        from medcat.cat import CAT
    
    # 下载/加载模型
    try:
        model_path = download_model()
        cat = load_model(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查网络连接或手动下载模型")
        return
    
    # 获取 stay_ids
    cohort_file = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/cohorts/cohort_with_conditions.csv')
    if cohort_file.exists():
        cohort = pd.read_csv(cohort_file)
        stay_ids = cohort['stay_id'].tolist()
    else:
        stay_ids = [int(f.stem.split('_')[-1]) for f in EPISODES_DIR.glob('*.json')]
    
    print(f"总 Episodes: {len(stay_ids):,}")
    
    # 批量处理
    results = []
    for stay_id in tqdm(stay_ids, desc="提取 MedCAT 概念"):
        texts = extract_notes_from_episode(stay_id)
        if not texts:
            continue
        
        entities = extract_concepts(cat, texts)
        features = aggregate_to_features(entities)
        features['stay_id'] = stay_id
        results.append(features)
    
    # 保存结果
    df = pd.DataFrame(results).fillna(0)
    df.to_csv(OUTPUT_DIR / 'medcat_concepts.csv', index=False)
    
    print(f"\n完成！")
    print(f"  样本数: {len(df):,}")
    print(f"  特征维度: {df.shape[1]}")
    print(f"  保存到: {OUTPUT_DIR}")
    
    # 统计
    print(f"\n概念统计:")
    print(f"  平均概念数: {df['n_concepts'].mean():.1f}")
    print(f"  平均唯一 CUI 数: {df['n_unique_cuis'].mean():.1f}")


if __name__ == "__main__":
    main()
