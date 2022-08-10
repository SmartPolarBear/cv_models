import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.randn(shape, dtype=x.dtype, device=x.device)

    random_tensor.floor_()

    out = x.div(keep_prob) * random_tensor
    return out


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.prob, self.training)


class ConvNeXTBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 7, drop_path: float = 0., layer_scale: float = 1e-6):
        super(ConvNeXTBlock, self).__init__()

        self.dw = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=kernel_size, stride=1, padding=3, groups=in_channels)
        self.ln = nn.LayerNorm(normalized_shape=in_channels)

        self.pw1 = nn.Linear(in_features=in_channels, out_features=in_channels * 4)
        self.gelu = nn.GELU()
        self.pw2 = nn.Linear(in_features=in_channels * 4, out_features=in_channels)

        self.layer_scale = nn.Parameter(layer_scale * torch.ones(in_channels)) if layer_scale > 0 else None
        self.drop_path = DropPath(drop_prob=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dw(x)
        x = self.ln(x)  # B,C,H,W

        x.permute(0, 2, 3, 1)  # B,H,W,C

        x = self.pw1(x)
        x = self.gelu(x)
        x = self.pw2(x)

        if self.layer_scale is not None:
            x = self.layer_scale * x

        x.permute(0, 3, 1, 2)  # B,C,H,W

        x = residual + self.drop_path(x)
        return x
