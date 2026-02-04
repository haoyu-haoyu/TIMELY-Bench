"""
Cross-Modal Attention 训练脚本

Part 2 Day 4-5: 完整训练流程
包含 5-fold CV, Early Stopping, 评估指标
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from dataset_multimodal import MultimodalEpisodeDataset, get_group_split
from cross_modal_attention import CrossModalModel


class Trainer:
    """Cross-Modal Attention 训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 0.001,
        patience: int = 10
    ):
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
    
    def train_epoch(self, dataloader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            ts = batch['timeseries'].to(self.device)
            text = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(ts, text)
            loss = self.criterion(logits.squeeze(), labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader) -> dict:
        """评估模型"""
        self.model.eval()
        all_labels = []
        all_probs = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                ts = batch['timeseries'].to(self.device)
                text = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _ = self.model(ts, text)
                loss = self.criterion(logits.squeeze(), labels)
                
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs if isinstance(probs, np.ndarray) else [probs])
                total_loss += loss.item()
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        try:
            auroc = roc_auc_score(all_labels, all_probs)
            auprc = average_precision_score(all_labels, all_probs)
        except Exception:
            auroc = 0.5
            auprc = 0.0
        
        return {
            'loss': total_loss / len(dataloader),
            'auroc': auroc,
            'auprc': auprc
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 50
    ) -> dict:
        """完整训练流程"""
        best_auroc = 0
        best_epoch = 0
        patience_counter = 0
        history = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.scheduler.step(val_metrics['auroc'])
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_auroc': val_metrics['auroc'],
                'val_auprc': val_metrics['auprc']
            })
            
            print(f"Epoch {epoch+1:02d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val AUROC: {val_metrics['auroc']:.4f}")
            
            # Early stopping
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                best_epoch = epoch + 1
                patience_counter = 0
                # 保存最佳模型
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # 恢复最佳模型
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        return {
            'best_auroc': best_auroc,
            'best_epoch': best_epoch,
            'history': history
        }


def train_cross_modal(
    task: str = 'mortality',
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    hidden_dim: int = 128,
    max_samples: int = None
):
    """
    运行 Cross-Modal Attention 训练
    """
    print("=" * 60)
    print("Cross-Modal Attention Training")
    print("=" * 60)
    print(f"Task: {task}")
    print(f"Folds: {n_folds}")
    print(f"Epochs: {epochs}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 数据路径
    episodes_dir = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'cross_modal'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 episode 文件
    episode_files = sorted(episodes_dir.glob('TIMELY_v2_*.json'))
    if max_samples:
        episode_files = episode_files[:max_samples]
    
    print(f"Episodes: {len(episode_files)}")
    
    # 创建数据集
    dataset = MultimodalEpisodeDataset(
        episode_files=episode_files,
        task=task,
        apply_leak_prevention=True  # 防泄漏
    )
    
    # 5-fold 交叉验证
    fold_results = []
    
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # 获取划分（患者级别）
        train_idx, val_idx = get_group_split(dataset, n_folds, fold)
        
        # 创建子数据集
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # 创建模型
        model = CrossModalModel(
            ts_dim=9,
            text_dim=50,
            hidden_dim=hidden_dim,
            n_heads=4
        )
        
        # 训练
        trainer = Trainer(model, patience=10)
        result = trainer.train(train_loader, val_loader, epochs)
        
        fold_results.append({
            'fold': fold + 1,
            'best_auroc': result['best_auroc'],
            'best_epoch': result['best_epoch']
        })
        
        print(f"Fold {fold + 1} best AUROC: {result['best_auroc']:.4f}")
    
    # 汇总结果
    aurocs = [r['best_auroc'] for r in fold_results]
    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs)
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Mean AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(results_dir / 'cross_modal_cv_results.csv', index=False)
    
    summary = {
        'task': task,
        'model': 'CrossModalAttention',
        'n_folds': n_folds,
        'mean_auroc': mean_auroc,
        'std_auroc': std_auroc,
        'timestamp': datetime.now().isoformat()
    }
    
    pd.DataFrame([summary]).to_csv(results_dir / 'cross_modal_summary.csv', index=False)
    
    print(f"\nResults saved to: {results_dir}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mortality')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    train_cross_modal(
        task=args.task,
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
