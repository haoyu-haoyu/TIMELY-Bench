"""
Cross-Modal Attention 模型

Part 2 Day 2-3: 跨模态注意力机制
时序作为 Query, 文本作为 Key/Value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    
    时序特征作为 Query，文本特征作为 Key/Value
    """
    
    def __init__(
        self,
        ts_dim: int = 9,
        text_dim: int = 50,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        gru_layers: int = 2
    ):
        """
        Args:
            ts_dim: 时序特征维度
            text_dim: 文本特征维度
            hidden_dim: 隐藏层维度
            n_heads: 注意力头数
            dropout: Dropout 率
            gru_layers: GRU 层数
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # 时序编码器
        self.ts_encoder = nn.GRU(
            input_size=ts_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=False
        )
        
        # 文本投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 文本扩展为序列（用于 attention）
        self.text_expand = nn.Linear(hidden_dim, hidden_dim * 4)
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 融合层
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类头
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        ts: torch.Tensor,
        text: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            ts: 时序特征 (B, T, D_ts)
            text: 文本特征 (B, D_txt)
            return_attention: 是否返回注意力权重
        
        Returns:
            logits: 预测 logits (B, 1)
            attn_weights: 可选的注意力权重 (B, n_heads, T, N_text)
        """
        B, T, _ = ts.shape
        
        # 编码时序
        ts_enc, ts_hidden = self.ts_encoder(ts)  # (B, T, H)
        
        # 投影文本
        text_proj = self.text_proj(text)  # (B, H)
        
        # 扩展文本为伪序列用于 attention
        text_expanded = self.text_expand(text_proj)  # (B, H*4)
        text_seq = text_expanded.view(B, 4, self.hidden_dim)  # (B, 4, H)
        
        # Cross-Modal Attention
        # Q: 时序, K/V: 文本
        attn_out, attn_weights = self.cross_attn(
            query=ts_enc,       # (B, T, H)
            key=text_seq,       # (B, 4, H)
            value=text_seq      # (B, 4, H)
        )  # attn_out: (B, T, H), attn_weights: (B, T, 4)
        
        # 残差连接
        ts_fused = self.fusion_norm(ts_enc + attn_out)
        
        # 池化
        ts_pooled = ts_fused[:, -1, :]  # 最后一个时间步
        attn_pooled = attn_out.mean(dim=1)  # 平均池化
        
        # 融合
        fused = torch.cat([ts_pooled, attn_pooled], dim=-1)  # (B, H*2)
        fused = self.fusion_mlp(fused)  # (B, H//2)
        
        # 分类
        logits = self.classifier(fused)  # (B, 1)
        
        if return_attention:
            return logits, attn_weights
        return logits, None


class CrossModalModel(nn.Module):
    """
    完整的跨模态预测模型
    
    包含时序编码、文本编码、跨模态注意力和分类
    """
    
    def __init__(
        self,
        ts_dim: int = 9,
        text_dim: int = 50,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        gru_layers: int = 2
    ):
        super().__init__()
        
        self.cross_modal_attn = CrossModalAttention(
            ts_dim=ts_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            gru_layers=gru_layers
        )
    
    def forward(
        self,
        ts: torch.Tensor,
        text: torch.Tensor,
        return_attention: bool = False
    ):
        return self.cross_modal_attn(ts, text, return_attention)
    
    def predict_proba(self, ts: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """返回概率预测"""
        logits, _ = self.forward(ts, text)
        return torch.sigmoid(logits)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    import torch
    
    # 创建模型
    model = CrossModalModel(
        ts_dim=9,
        text_dim=50,
        hidden_dim=128,
        n_heads=4
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # 测试输入
    batch_size = 32
    seq_len = 48
    
    ts = torch.randn(batch_size, seq_len, 9)
    text = torch.randn(batch_size, 50)
    
    # 前向传播
    logits, attn = model(ts, text, return_attention=True)
    
    print(f"Input ts: {ts.shape}")
    print(f"Input text: {text.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Attention weights: {attn.shape if attn is not None else 'None'}")
    
    # 测试概率预测
    probs = model.predict_proba(ts, text)
    print(f"Probabilities: {probs[:5].squeeze()}")
