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


class ConvBNReLULayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 1, padding: int = 0,
                 stride: int = 1, separable=False, act=False, act_before=False):
        super(ConvBNReLULayer, self).__init__()

        ConvType = nn.Conv2d if not separable else SeparableConv2d

        self.conv = ConvType(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, padding=padding,
                             stride=stride)

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.act = nn.Sequential() if not act else nn.ReLU()

        self.before = act_before

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.before:
            x = self.act(x)

        x = self.conv(x)
        x = self.bn(x)

        if not self.before:
            x = self.act(x)

        return x


class EntryBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(EntryBlock, self).__init__()

        self.shortcut = ConvBNReLULayer(
            in_channels=in_channels, out_channels=out_channels,
            stride=2, separable=False, act=False, act_before=False
        )

        self.conv1 = ConvBNReLULayer(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, stride=1, padding=1, separable=True,
                                     act=True, act_before=False)
        self.conv2 = ConvBNReLULayer(in_channels=out_channels, out_channels=out_channels,
                                     kernel_size=3, stride=1, padding=1, separable=True,
                                     act=False, act_before=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = residual + x
        return x


class MiddleBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(MiddleBlock, self).__init__()

        self.convs = nn.Sequential(
            *[ConvBNReLULayer(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=3, stride=1, padding=1, separable=True,
                              act=True, act_before=True) for _ in range(3)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.convs(x)
        x = residual + x
        return x


class ExitBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ExitBlock, self).__init__()

        self.shortcut = ConvBNReLULayer(
            in_channels=in_channels, out_channels=out_channels,
            stride=2, separable=False, act=False, act_before=False
        )

        self.conv1 = ConvBNReLULayer(in_channels=in_channels, out_channels=in_channels,
                                     kernel_size=3, stride=1, padding=1, separable=True,
                                     act=True, act_before=True)

        self.conv2 = ConvBNReLULayer(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, stride=1, padding=1, separable=True,
                                     act=True, act_before=True)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = residual + x
        return x
