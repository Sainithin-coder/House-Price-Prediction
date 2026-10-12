"""
Deep Learning Model: Joint Self-Attention Network for House Price Prediction
Architecture:
  - Numerical Branch: Dense → LayerNorm → Self-Attention (intra-feature)
  - Categorical Branch: Embedding → Self-Attention
  - Joint Cross-Attention: Fuses both modalities
  - Regression Head: MLP → output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. Self-Attention Block
# ─────────────────────────────────────────────────────────────────────────────
class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention over a sequence of feature tokens."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, S, D = x.shape
        residual = x
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B, H, S, S)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)                  # (B, H, S, d_k)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(context)
        return self.norm(out + residual), attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-Attention Block (Joint Fusion)
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionBlock(nn.Module):
    """Cross-attention: query from one modality, key/value from the other."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, context):
        B, Sq, D = query.shape
        _, Sc, _ = context.shape
        residual = query
        Q = self.W_q(query).view(B, Sq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(context).view(B, Sc, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(context).view(B, Sc, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Sq, D)
        return self.norm(self.out_proj(out) + residual), attn


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
class NumericalTokenizer(nn.Module):
    """Projects each scalar numerical feature into a d_model embedding."""
    def __init__(self, n_num_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)  # applied per feature
        self.n_features = n_num_features
        self.d_model = d_model

    def forward(self, x):
        # x: (B, n_num_features)
        B, F = x.shape
        x = x.unsqueeze(-1)               # (B, F, 1)
        return self.proj(x)               # (B, F, d_model)


class CategoricalTokenizer(nn.Module):
    """Embeds each categorical feature index and projects to d_model."""
    def __init__(self, cat_dims: dict, d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 1, embedding_dim=d_model)
            for dim in cat_dims.values()
        ])

    def forward(self, x):
        # x: (B, n_cat_features)  each column is an int index
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1)   # (B, n_cat, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Joint Self-Attention Network
# ─────────────────────────────────────────────────────────────────────────────
class JointSelfAttentionNet(nn.Module):
    """
    Heterogeneous House Price Prediction Network with Joint Self-Attention.

    Processing pipeline:
      Numerical features  → NumericalTokenizer → SelfAttention × 2
      Categorical features→ CategoricalTokenizer → SelfAttention × 2
      Joint Fusion        → CrossAttention (num queries cat) + CrossAttention (cat queries num)
      Pooling             → Mean-pool both streams → concat → MLP Head
    """
    def __init__(self,
                 n_num_features: int,
                 cat_dims: dict,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_self_attn_layers: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.d_model = d_model

        # Tokenizers
        self.num_tokenizer = NumericalTokenizer(n_num_features, d_model)
        self.cat_tokenizer = CategoricalTokenizer(cat_dims, d_model)

        # Self-Attention stacks
        self.num_self_attn = nn.ModuleList(
            [SelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_self_attn_layers)]
        )
        self.cat_self_attn = nn.ModuleList(
            [SelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_self_attn_layers)]
        )

        # Joint Cross-Attention
        self.cross_attn_num2cat = CrossAttentionBlock(d_model, n_heads, dropout)
        self.cross_attn_cat2num = CrossAttentionBlock(d_model, n_heads, dropout)

        # Regression head
        fused_dim = d_model * 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_num, x_cat):
        # Tokenize
        num_tokens = self.num_tokenizer(x_num)    # (B, n_num, d_model)
        cat_tokens = self.cat_tokenizer(x_cat)    # (B, n_cat, d_model)

        # Intra-modality Self-Attention
        attn_weights_num, attn_weights_cat = [], []
        for layer in self.num_self_attn:
            num_tokens, aw = layer(num_tokens)
            attn_weights_num.append(aw)
        for layer in self.cat_self_attn:
            cat_tokens, aw = layer(cat_tokens)
            attn_weights_cat.append(aw)

        # Joint Cross-Attention (inter-modality fusion)
        num_fused, _ = self.cross_attn_num2cat(num_tokens, cat_tokens)
        cat_fused, _ = self.cross_attn_cat2num(cat_tokens, num_tokens)

        # Mean pooling over sequence dim
        num_pooled = num_fused.mean(dim=1)   # (B, d_model)
        cat_pooled = cat_fused.mean(dim=1)   # (B, d_model)

        # Concatenate and regress
        fused = torch.cat([num_pooled, cat_pooled], dim=-1)  # (B, 2*d_model)
        out = self.head(fused).squeeze(-1)
        return out, attn_weights_num, attn_weights_cat


# ─────────────────────────────────────────────────────────────────────────────
# 5. Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────
def build_model(n_num_features: int, cat_dims: dict, device='cpu') -> JointSelfAttentionNet:
    model = JointSelfAttentionNet(
        n_num_features=n_num_features,
        cat_dims=cat_dims,
        d_model=64,
        n_heads=4,
        n_self_attn_layers=2,
        dropout=0.15
    ).to(device)
    return model
