import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum, unique, auto
from typing import Tuple


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


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXTBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 7, drop_path: float = 0., layer_scale: float = 1e-6):
        super(ConvNeXTBlock, self).__init__()

        self.dw = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=kernel_size, stride=1, padding=3, groups=in_channels)
        self.ln = LayerNorm(normalized_shape=in_channels)

        self.pw1 = nn.Linear(in_features=in_channels, out_features=in_channels * 4)
        self.gelu = nn.GELU()
        self.pw2 = nn.Linear(in_features=in_channels * 4, out_features=in_channels)

        self.layer_scale = nn.Parameter(layer_scale * torch.ones(in_channels)) if layer_scale > 0 else None
        self.drop_path = DropPath(drop_prob=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # B,C,H,W

        x = self.dw(x)

        x = x.permute(0, 2, 3, 1)  # B,H,W,C

        x = self.ln(x)
        x = self.pw1(x)
        x = self.gelu(x)
        x = self.pw2(x)

        if self.layer_scale is not None:
            x = self.layer_scale * x

        x = x.permute(0, 3, 1, 2)  # B,C,H,W

        x = residual + self.drop_path(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownsampleBlock, self).__init__()
        self.ln = LayerNorm(normalized_shape=in_channels, data_format='channels_first')
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.conv(x)

        return x


@unique
class ConvNeXTConfig(Enum):
    T = ((96, 192, 384, 768), (3, 3, 9, 3))
    S = ((96, 192, 384, 768), (3, 3, 27, 3))
    B = ((128, 256, 512, 1024), (3, 3, 27, 3))
    L = ((192, 384, 768, 1536), (3, 3, 27, 3))
    XL = ((256, 512, 1024, 2048), (3, 3, 27, 3))


class ConvNeXT(nn.Module):
    def __init__(self, in_channels: int = 3, conf: ConvNeXTConfig = ConvNeXTConfig.B,
                 num_classes: int = 10, drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6, head_init_scale: float = 1.):
        super(ConvNeXT, self).__init__()

        self.channels, self.blocks = conf.value

        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=self.channels[0],
                          kernel_size=4, stride=4),
                LayerNorm(normalized_shape=self.channels[0], data_format='channels_first')
            )
        ])

        for i in range(1, 4):
            self.downsamples.append(
                DownsampleBlock(in_channels=self.channels[i - 1], out_channels=self.channels[i])
            )

        self.stages = nn.ModuleList()
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.blocks))]
        cur = 0
        for c, b in zip(self.channels, self.blocks):
            self.stages.append(nn.Sequential(
                *[ConvNeXTBlock(in_channels=c, drop_path=drop_rates[cur + i], layer_scale=layer_scale_init_value)
                  for i in range(b)]
            ))
            cur += b

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            LayerNorm(self.channels[-1], data_format='channels_first'),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=self.channels[-1], out_features=num_classes)
        )

        self.classifier[3].weight.data.mul_(head_init_scale)
        self.classifier[3].bias.data.mul_(head_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.stages[i](x)

        x = self.classifier(x)

        return x
