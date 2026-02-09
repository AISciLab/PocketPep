import os
import esm
import numpy as np
import torch
from torch import nn
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,q_hidden_size,k_hidden_size,v_hidden_size,hidden_size, num_heads, qkv_bias=True, dropout=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(q_hidden_size, hidden_size, bias=qkv_bias)
        self.key = nn.Linear(k_hidden_size, hidden_size, bias=qkv_bias)
        self.value = nn.Linear(v_hidden_size, hidden_size, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask=None):
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
                    self.head_dim ** 0.5)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [16,1,1,152]
            mask = mask.float()
            mask = (1.0 - mask) * -1e9
            attn_scores = attn_scores + mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(context.size(0), context.size(2), -1)
        output = self.out_proj(context)
        return output


class Block(nn.Module):

    def __init__(self, hidden_size,pocket_hidden_size ,num_heads, mlp_ratio=4.0, ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size,hidden_size,hidden_size,hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(hidden_size,pocket_hidden_size,pocket_hidden_size,hidden_size, num_heads)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)



    def forward(self,pocket,pocket_mask, pp, pp_mask):
        x = pp
        pp = self.norm1(pp)
        pp = self.attn(pp,pp,pp,pp_mask)
        pp = self.norm2(pp)
        pp = self.cross_attn(pp,pocket,pocket,pocket_mask)
        pp = x +  self.mlp(pp)
        return pp

class Scale100(nn.Module):
    def forward(self, x):
        return x * 100

class Affinity_Reward(nn.Module):
    def __init__(self,hidden_size,pocket_hidden_size,depth=4,mlp_ratio=4.0):
        super().__init__()
        self.pp_embedder = nn.Sequential(
            nn.Linear(20, 128, bias=True),  # 初始线性层 (384 -> 34)
            nn.LayerNorm(128),  # LayerNorm (对384维归一化)
            nn.GELU(),  # 激活函数
            nn.Linear(128, hidden_size, bias=True),  # 第二个全连接层 (384 -> 256)
            nn.Dropout(p=0.1)  # Dropout层
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(hidden_size, pocket_hidden_size,8, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.final_layer = nn.Sequential(
            Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,out_features=1, act_layer=approx_gelu, drop=0),
        )
        self.initialize_weights(hidden_size)

    def initialize_weights(self,hidden_size):
        pos_embed = get_1d_sincos_pos_embed_from_grid(hidden_size, np.arange(50))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, pocket_emb,pocket_mask,peptide,pp_mask):
        pp = self.pp_embedder(peptide)
        seq_len = pp.shape[1]
        pp = pp + self.pos_embed[:, :seq_len, :]
        for block in self.blocks:
            pp = block(pocket_emb, pocket_mask,pp,  pp_mask)
        pp = pp.mean(dim=1)
        score = self.final_layer(pp)
        return score

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


