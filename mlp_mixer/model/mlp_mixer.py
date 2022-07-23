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


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class MixerBlock(nn.Module):
    def __init__(self, n_embeds, channels, token_mlp_dim, channel_mlp_dim, token_drop=0., channel_drop=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(n_embeds)
        self.norm2 = nn.LayerNorm(n_embeds)

        self.mlp1 = MLP(dim=n_embeds, hidden_dim=token_mlp_dim, drop=token_drop)
        self.mlp2 = MLP(dim=channels, hidden_dim=channel_mlp_dim, drop=channel_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_out = self.norm1(x)  # B,N,C
        p_out = p_out.permute(0, 2, 1)  # B,C,N
        p_out = self.mlp1(p_out)
        p_out = p_out.permute(0, 2, 1)  # B,N,C
        x = x + p_out

        c_out = self.norm2(x)
        c_out = self.mlp2(c_out)
        x = x + c_out

        return x


class MLPMixer(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 embedding_dim=768,
                 in_channels=3,
                 blocks=16,
                 classes=4,
                 token_mlp_ratio=4.,
                 channels_mlp_ratio=4.,
                 token_drop=0.,
                 channels_drop=0.,
                 ):
        super().__init__()

        self.n_embeds = image_size // patch_size
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embedding_dim)

        token_mlp_dim = int(self.n_embeds * token_mlp_ratio)
        channels_mlp_dim = int(embedding_dim * channels_mlp_ratio)

        self.blocks = nn.ModuleList([
            MixerBlock(self.n_embeds, embedding_dim, token_mlp_dim, channels_mlp_dim, token_drop, channels_drop)
            for _ in range(blocks)
        ])

        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(in_features=embedding_dim, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = torch.mean(x, dim=1)  # GAP
        x = self.head(x)

        return x
