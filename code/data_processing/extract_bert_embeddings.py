"""
ClinicalBERT 嵌入提取
从临床笔记中提取 768 维嵌入向量

使用模型: emilyalsentzer/Bio_ClinicalBERT
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ROOT_DIR

# 配置
EPISODES_DIR = ROOT_DIR / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = ROOT_DIR / 'data' / 'processed' / 'text_embeddings'
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    """加载 ClinicalBERT 模型"""
    print(f"加载模型: {MODEL_NAME}")
    print(f"使用设备: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    
    return tokenizer, model


def get_cls_embedding(texts, tokenizer, model):
    """获取文本的 CLS 嵌入"""
    if not texts:
        return None
    
    # 分词
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取 CLS token 嵌入
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return cls_embeddings


def extract_notes_from_episode(episode_path):
    """从 Episode 文件中提取笔记文本"""
    try:
        with open(episode_path) as f:
            ep = json.load(f)
        
        stay_id = ep.get('stay_id')
        clinical_text = ep.get('clinical_text', {})
        notes = clinical_text.get('notes', [])
        
        # 提取笔记文本
        texts = []
        for note in notes:
            if isinstance(note, dict):
                # 尝试多个可能的键名
                note_type = note.get('note_type')
                if note_type == 'discharge':
                    continue
                chart_hour = note.get('chart_hour')
                try:
                    chart_hour = float(chart_hour) if chart_hour is not None else None
                except (TypeError, ValueError):
                    chart_hour = None
                if chart_hour is not None and (chart_hour < 0 or chart_hour >= 24):
                    continue

                text = note.get('text_full') or note.get('text_relevant') or note.get('text', '')
            else:
                text = str(note)
            if text and len(text) > 10:
                texts.append(text[:2000])  # 截断长文本
        
        return stay_id, texts
    except Exception as e:
        return None, []


def process_batch(texts_batch, tokenizer, model):
    """处理一批文本"""
    all_texts = []
    batch_info = []
    
    for stay_id, texts in texts_batch:
        if texts:
            start_idx = len(all_texts)
            all_texts.extend(texts)
            end_idx = len(all_texts)
            batch_info.append((stay_id, start_idx, end_idx))
    
    if not all_texts:
        return {}
    
    # 分批处理大量文本
    all_embeddings = []
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i:i+BATCH_SIZE]
        embeddings = get_cls_embedding(batch_texts, tokenizer, model)
        if embeddings is not None:
            all_embeddings.append(embeddings)
    
    if not all_embeddings:
        return {}
    
    all_embeddings = np.vstack(all_embeddings)
    
    # 聚合每个 stay_id 的嵌入 (mean pooling)
    results = {}
    for stay_id, start_idx, end_idx in batch_info:
        stay_embeddings = all_embeddings[start_idx:end_idx]
        mean_embedding = stay_embeddings.mean(axis=0)
        results[stay_id] = mean_embedding
    
    return results


def main():
    print("=" * 60)
    print("ClinicalBERT 嵌入提取")
    print("=" * 60)

    if not EPISODES_DIR.exists():
        raise FileNotFoundError(f"Episodes dir not found: {EPISODES_DIR}")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    tokenizer, model = load_model()
    
    # 获取所有 Episode 文件
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    print(f"找到 {len(episode_files):,} 个 Episode 文件")
    
    # 存储结果
    all_embeddings = []
    all_stay_ids = []
    
    # 批处理
    batch = []
    batch_size = 10  # 每次处理 10 个 Episode
    
    for ep_file in tqdm(episode_files, desc="提取嵌入"):
        stay_id, texts = extract_notes_from_episode(ep_file)
        
        if stay_id and texts:
            batch.append((stay_id, texts))
        
        if len(batch) >= batch_size:
            results = process_batch(batch, tokenizer, model)
            for sid, emb in results.items():
                all_stay_ids.append(sid)
                all_embeddings.append(emb)
            batch = []
    
    # 处理剩余的
    if batch:
        results = process_batch(batch, tokenizer, model)
        for sid, emb in results.items():
            all_stay_ids.append(sid)
            all_embeddings.append(emb)
    
    # 保存结果
    if all_embeddings:
        embeddings_array = np.array(all_embeddings)
        stay_ids_df = pd.DataFrame({'stay_id': all_stay_ids})
        
        # 保存
        np.save(OUTPUT_DIR / 'clinical_bert_embeddings.npy', embeddings_array)
        stay_ids_df.to_csv(OUTPUT_DIR / 'embedding_stay_ids.csv', index=False)
        
        print("\n" + "=" * 60)
        print(f"完成！")
        print(f"  提取了 {len(all_stay_ids):,} 个 stay_id 的嵌入")
        print(f"  嵌入维度: {embeddings_array.shape}")
        print(f"  保存到: {OUTPUT_DIR}")
    else:
        print("警告：未提取到任何嵌入")


if __name__ == "__main__":
    main()
