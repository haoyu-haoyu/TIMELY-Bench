"""
Cross-Modal Attention 多模态数据加载器

Part 2 Day 1: 加载时序 + 文本的多模态数据
含数据泄漏防范措施
"""

import json
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional


# 标签泄漏词列表
LABEL_LEAK_TERMS = [
    'expired', 'death', 'deceased', 'died', 'mortality',
    'comfort care', 'CMO', 'DNR', 'hospice', 'passed away',
    'pronounced', 'time of death', 'TOD', 'pronounced dead',
    'patient expired', 'cardiac arrest', 'code blue'
]


def mask_label_terms(text: str) -> str:
    """遮蔽可能泄漏标签的词汇"""
    if not text:
        return text
    for term in LABEL_LEAK_TERMS:
        text = re.sub(rf'\b{term}\b', '[MASKED]', text, flags=re.IGNORECASE)
    return text


def filter_notes_by_time(notes: List[Dict], max_hours: int = 24) -> List[Dict]:
    """只保留观察窗口内的笔记"""
    return [n for n in notes if n.get('hour', 0) <= max_hours]


class MultimodalEpisodeDataset(Dataset):
    """
    多模态 Episode 数据集
    
    加载时序特征和文本特征，应用数据泄漏防范
    """
    
    def __init__(
        self,
        episode_files: List[Path],
        task: str = 'mortality',
        max_seq_len: int = 48,
        text_dim: int = 50,
        observation_hours: int = 24,
        apply_leak_prevention: bool = True
    ):
        """
        Args:
            episode_files: Episode JSON 文件路径列表
            task: 预测任务 ('mortality' or 'prolonged_los')
            max_seq_len: 最大序列长度
            text_dim: 文本特征维度
            observation_hours: 观察窗口（小时）
            apply_leak_prevention: 是否应用泄漏防范
        """
        self.episode_files = episode_files
        self.task = task
        self.max_seq_len = max_seq_len
        self.text_dim = text_dim
        self.observation_hours = observation_hours
        self.apply_leak_prevention = apply_leak_prevention
        
        # 时序特征名
        self.ts_features = [
            'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate',
            'temperature', 'spo2', 'gcs', 'urineoutput'
        ]
        self.ts_dim = len(self.ts_features)
    
    def __len__(self):
        return len(self.episode_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_file = self.episode_files[idx]
        
        try:
            with open(ep_file) as f:
                ep = json.load(f)
        except Exception:
            return self._empty_sample()
        
        # 提取标签
        labels = ep.get('labels', {})
        outcome = labels.get('outcome', {})
        
        if self.task == 'mortality':
            label = outcome.get('mortality', 0) or 0
        else:
            label = outcome.get('prolonged_los', 0) or 0
        
        # 提取时序特征
        ts_tensor = self._extract_timeseries(ep)
        
        # 提取文本特征（含泄漏防范）
        text_tensor = self._extract_text_features(ep)
        
        # 提取 subject_id 用于 GroupKFold
        subject_id = ep.get('patient', {}).get('subject_id', 0)
        
        return {
            'timeseries': ts_tensor,
            'text': text_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'subject_id': subject_id
        }
    
    def _extract_timeseries(self, ep: Dict) -> torch.Tensor:
        """提取时序特征矩阵 (T, D_ts)"""
        ts = ep.get('timeseries', {})
        vitals_list = ts.get('vitals', [])
        
        # 初始化矩阵
        seq = np.zeros((self.max_seq_len, self.ts_dim), dtype=np.float32)
        
        if isinstance(vitals_list, list):
            for t, record in enumerate(vitals_list[:self.max_seq_len]):
                if isinstance(record, dict):
                    for i, feat in enumerate(self.ts_features):
                        val = record.get(feat)
                        if val is not None:
                            seq[t, i] = float(val)
        
        return torch.from_numpy(seq)
    
    def _extract_text_features(self, ep: Dict) -> torch.Tensor:
        """提取文本特征向量 (D_txt,)"""
        clinical = ep.get('clinical_text', {})
        notes = clinical.get('notes', [])
        
        # 过滤时间窗口外的笔记
        if self.apply_leak_prevention:
            notes = filter_notes_by_time(notes, self.observation_hours)
        
        # 合并文本
        all_text = " ".join([n.get('text', '') for n in notes])
        
        # 遮蔽泄漏词
        if self.apply_leak_prevention:
            all_text = mask_label_terms(all_text)
        
        # 简单特征：词频统计
        text_features = np.zeros(self.text_dim, dtype=np.float32)
        
        # 医学关键词列表
        medical_terms = [
            'fever', 'infection', 'sepsis', 'pneumonia', 'respiratory',
            'cardiac', 'renal', 'liver', 'stroke', 'bleeding',
            'hypotension', 'tachycardia', 'bradycardia', 'hypoxia', 'edema',
            'intubation', 'ventilator', 'dialysis', 'surgery', 'antibiotic',
            'sedation', 'vasopressor', 'transfusion', 'stable', 'critical',
            'improving', 'worsening', 'alert', 'responsive', 'conscious',
            'pain', 'nausea', 'vomiting', 'diarrhea', 'constipation',
            'cough', 'dyspnea', 'chest', 'abdomen', 'extremity',
            'lab', 'imaging', 'xray', 'ct', 'ultrasound',
            'normal', 'abnormal', 'elevated', 'decreased', 'positive'
        ]
        
        text_lower = all_text.lower()
        for i, term in enumerate(medical_terms[:self.text_dim]):
            text_features[i] = text_lower.count(term)
        
        return torch.from_numpy(text_features)
    
    def _empty_sample(self) -> Dict[str, torch.Tensor]:
        """返回空样本"""
        return {
            'timeseries': torch.zeros(self.max_seq_len, self.ts_dim),
            'text': torch.zeros(self.text_dim),
            'label': torch.tensor(0.0),
            'subject_id': 0
        }


def get_group_split(
    dataset: MultimodalEpisodeDataset,
    n_splits: int = 5,
    fold: int = 0
) -> Tuple[List[int], List[int]]:
    """
    获取患者级别的数据划分
    
    确保同一患者不会同时出现在 train 和 val
    """
    # 收集 subject_ids
    subject_ids = []
    for i in range(len(dataset)):
        sample = dataset[i]
        subject_ids.append(sample['subject_id'])
    
    subject_ids = np.array(subject_ids)
    labels = np.zeros(len(subject_ids))  # 占位
    
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(np.arange(len(subject_ids)), labels, groups=subject_ids))
    
    train_idx, val_idx = splits[fold]
    
    # 验证无泄漏
    train_subjects = set(subject_ids[train_idx])
    val_subjects = set(subject_ids[val_idx])
    overlap = train_subjects & val_subjects
    assert len(overlap) == 0, f"Patient leakage: {len(overlap)} subjects"
    
    return train_idx.tolist(), val_idx.tolist()


def create_dataloaders(
    episodes_dir: Path,
    task: str = 'mortality',
    batch_size: int = 64,
    n_splits: int = 5,
    fold: int = 0,
    num_workers: int = 4,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader
    """
    # 获取所有 episode 文件
    episode_files = sorted(episodes_dir.glob('TIMELY_v2_*.json'))
    
    if max_samples:
        episode_files = episode_files[:max_samples]
    
    print(f"加载 {len(episode_files)} 个 episodes...")
    
    # 创建数据集
    dataset = MultimodalEpisodeDataset(
        episode_files=episode_files,
        task=task,
        apply_leak_prevention=True
    )
    
    # 获取划分
    train_idx, val_idx = get_group_split(dataset, n_splits, fold)
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据加载
    from pathlib import Path
    
    episodes_dir = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
    
    if episodes_dir.exists():
        train_loader, val_loader = create_dataloaders(
            episodes_dir,
            task='mortality',
            batch_size=32,
            max_samples=1000
        )
        
        # 测试一个 batch
        for batch in train_loader:
            print(f"Timeseries: {batch['timeseries'].shape}")
            print(f"Text: {batch['text'].shape}")
            print(f"Labels: {batch['label'].shape}")
            break
    else:
        print(f"Episodes dir not found: {episodes_dir}")
