import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int = 3, embedding_dim: int = 768):
        super().__init__()

        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embedding_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # B,embedding_dim,sqrt(n_patches),sqrt(n_patches)
        x = x.flatten(2)  # B,embedding_dim,n_patches
        x = x.transpose(1, 2)  # B,n_patches,embedding_dim
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, qkv_bias: bool = True, attn_dropout: float = 0.,
                 proj_dropout: float = 0.):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_dropout)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = nn.Dropout(p=proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, patches, dim = x.shape
        if dim != self.dim:
            raise ValueError()

        qkv: torch.Tensor = self.qkv(x)  # B,patches,3*dim
        qkv = qkv.reshape(B, patches, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q: torch.Tensor = qkv[0]
        k: torch.Tensor = qkv[1]
        v: torch.Tensor = qkv[2]

        k_t = k.transpose(-2, -1)

        attn = torch.softmax(q @ k_t * self.scale, dim=-1)
        attn = self.attn_drop(attn)

        wa = attn @ v
        wa = wa.transpose(1, 2)
        wa = wa.flatten(2)

        x = self.proj(wa)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, qkv_bias: bool = True, mlp_ratio: float = 4., attn_drop: float = 0.,
                 drop: float = 0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size: int = 384,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 n_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 n_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop: float = 0.,
                 attn_drop: float = 0, ):
        super().__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))  # +1 for class token
        self.pos_drop = nn.Dropout(p=drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, qkv_bias, mlp_ratio, attn_drop, drop)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, 1e-6)
        self.classification_head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)

        class_token = self.class_token.expand(B, -1, -1)
        x = torch.concat((class_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        class_token_fin = x[:, 0]
        x = self.classification_head(class_token_fin)
        return x
