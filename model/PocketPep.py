import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias=True, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
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
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -1e9
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(context.size(0), context.size(2), -1)

        output = self.out_proj(context)

        return output

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.input_size = hidden_size
        self.output_size = num_classes

        self.y_embedder_my = nn.Sequential(
            nn.Linear(self.input_size, 256, bias=True),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.output_size, bias=True),
            nn.Dropout(p=dropout_prob)
        )
    def embedding_mlp(self,labels):
        y1 = self.y_embedder_my(labels)
        return y1

    def forward(self, labels):
        embedding = self.embedding_mlp(labels)
        return embedding


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)



    def forward(self, x, pocket,pocket_mask):
        x = x +  self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm3(x),pocket,pocket,pocket_mask)
        x = x +  self.mlp(self.norm2(x))
        return x

class FinalLayer(nn.Module):

    def __init__(self, hidden_size,out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size,1152*out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        if(self.out_channels != 1):
            B, L = x.shape[:2]
            x = x.reshape(B, L, 2, 1152)
            x = x.permute(0, 2, 1, 3)
        return x

class Pocket_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim*4),
            nn.ReLU(),
            nn.Linear(in_dim*4, out_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DiT(nn.Module):

    def __init__(
            self,
            in_channels=1,
            hidden_size=1024,
            depth=12,
            num_heads=2,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.x_embedder_my = nn.Sequential(
            nn.Linear(1152, 256, bias=True),
            nn.LayerNorm(256),  # LayerNorm (对384维归一化)
            nn.GELU(),  # 激活函数
            nn.Linear(256, hidden_size, bias=True),
            nn.Dropout(p=0.1)  # Dropout层
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(hidden_size,384, class_dropout_prob)

        self.pos_embed = nn.Parameter(torch.zeros(1, 50, hidden_size), requires_grad=False)

        self.pockets = nn.ModuleList([
            Pocket_MLP(hidden_size,  hidden_size) for _ in range(depth)
        ])

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, 2, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size,self.out_channels)
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        pos_embed = get_1d_sincos_pos_embed_from_grid(1024, np.arange(50))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y,mask=None,pocket_mask=None):
        if(mask is None):
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=torch.float32)
        mask = mask.unsqueeze(1).unsqueeze(-1)
        x = self.x_embedder_my(x)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:,:seq_len,:]
        t = self.t_embedder(t)
        y = self.y_embedder(y)
        pocket = t.unsqueeze(1) + y
        c = t + y.mean(dim = 1)
        for block,pemb in zip(self.blocks,self.pockets):
            my_pocket = pemb(pocket)
            x = block(x,my_pocket,pocket_mask)
        x = self.final_layer(x, c)*mask
        return x



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
