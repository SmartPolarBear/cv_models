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
    def __init__(self, dim, n_heads, qkv_bias, attn_dropout, proj_dropout):
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
