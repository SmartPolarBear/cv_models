import torch
import torch.nn as nn

from modules.separableconv2d import SeparableConv2d


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
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = residual + x
        return x


class EntryFlow(nn.Module):
    block_in_channels = (64, 128, 256)
    block_out_channels = (128, 256, 728)

    def __init__(self, in_channels: int):
        super(EntryFlow, self).__init__()

        self.conv1 = ConvBNReLULayer(
            in_channels=in_channels, out_channels=32,
            kernel_size=3, padding=1,
            stride=2, separable=False,
            act=False, act_before=False
        )

        self.conv2 = ConvBNReLULayer(
            in_channels=32, out_channels=64,
            kernel_size=3, padding=1,
            stride=1, separable=False,
            act=False, act_before=False
        )

        self.blocks = nn.Sequential(
            *[EntryBlock(in_channels=self.block_in_channels[i],
                         out_channels=self.block_out_channels[i])
              for i in range(3)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.blocks(x)

        return x


class MiddleFlow(nn.Module):
    def __init__(self, repeat: int = 8):
        super(MiddleFlow, self).__init__()

        self.blocks = nn.Sequential(
            *[MiddleBlock(in_channels=728)
              for i in range(repeat)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)

        return x


class ExitFlow(nn.Module):
    def __init__(self, out_channels: int = 2048):
        super(ExitFlow, self).__init__()

        self.block = ExitBlock(in_channels=728, out_channels=1024)

        self.conv1 = ConvBNReLULayer(in_channels=1024, out_channels=1536,
                                     kernel_size=3, stride=1, padding=1, separable=True,
                                     act=True, act_before=False)

        self.conv2 = ConvBNReLULayer(in_channels=1536, out_channels=out_channels,
                                     kernel_size=3, stride=1, padding=1, separable=True,
                                     act=True, act_before=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Xception(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 10, middle_rep=8):
        super(Xception, self).__init__()

        self.entry = EntryFlow(in_channels=in_channels)
        self.middle = MiddleFlow(repeat=middle_rep)
        self.exit = ExitFlow(out_channels=2048)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)

        x = self.classifier(x)
        return x
