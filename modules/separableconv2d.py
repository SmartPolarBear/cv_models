import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 1, padding: int = 0,
                 stride: int = 1, dilation: int = 1,
                 bias: bool = False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, padding=padding,
                                   stride=stride, dilation=dilation,
                                   bias=bias, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, padding=0, stride=1, dilation=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
