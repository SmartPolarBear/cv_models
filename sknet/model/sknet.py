import torch
import torch.nn as nn

import torchvision
import torchvision.ops as ops

from resnet.model.resnet import ResNetClassifierHead

from modules.separableconv2d import SeparableConv2d


class ConvBNReLULayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1,
                 stride: int = 1, separable=False, act=False):
        super(ConvBNReLULayer, self).__init__()

        if separable:
            self.conv = SeparableConv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, padding=padding, dilation=dilation,
                                        stride=stride)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, padding=padding, dilation=dilation,
                                  groups=groups, stride=stride)

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.act = nn.Sequential() if not act else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)

        x = self.act(x)

        return x


class SKConv2d(nn.Module):
    def __init__(self, in_channels: int, groups: int, branches: int, ratio: int, stride: int = 1, l: int = 32):
        super(SKConv2d, self).__init__()

        d = max(in_channels // ratio, l)

        self.in_channels = in_channels
        self.branches = branches

        self.split = nn.ModuleList([
            ConvBNReLULayer(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=3, dilation=1 + i, padding=1 + i, stride=stride,
                            groups=groups, separable=True, act=True)
            for i in range(branches)])

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=d, kernel_size=1, stride=1)
        self.fc2s = nn.ModuleList(
            [nn.Conv2d(in_channels=d, out_channels=in_channels, kernel_size=1, stride=1)
             for _ in range(branches)]
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [conv(x) for conv in self.split]
        feats = torch.concat(feats, dim=1)

        B, C, H, W = feats.shape

        feats = feats.view(B, self.branches, self.in_channels, H, W)

        attn = torch.sum(feats, dim=1)
        attn = self.gap(attn)
        attn = self.fc1(attn)
        attn = [conv(attn) for conv in self.fc2s]

        attn = torch.concat(attn, dim=1)
        attn = attn.view(B, self.branches, self.in_channels, 1, 1)

        out = feats * attn
        out = torch.sum(out, dim=1)

        return out


class SKBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cardinality: int = 32, group_depth: int = 4,
                 branches: int = 2, ratio: int = 16, l: int = 32,
                 downsample: bool = False):
        super(SKBlock, self).__init__()

        group_channels = cardinality * group_depth

        self.conv1 = ConvBNReLULayer(in_channels=in_channels, out_channels=group_channels,
                                     kernel_size=1, stride=1, padding=0, act=True)

        self.conv2 = SKConv2d(in_channels=group_channels, stride=1 if not downsample else 2,
                              groups=cardinality, branches=branches, ratio=ratio, l=l)

        self.conv3 = ConvBNReLULayer(in_channels=group_channels, out_channels=out_channels,
                                     kernel_size=1, stride=1,
                                     padding=0, act=False)

        self.shortcut_conv = nn.Sequential() if not downsample and in_channels == out_channels else \
            nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, stride=1 if not downsample else 2),
                          nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.shortcut_conv(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + short
        x = self.relu(x)

        return x


class SKStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int,
                 cardinality: int = 32, group_depth: int = 4,
                 branches: int = 2, ratio: int = 16, l: int = 32,
                 downsample: bool = True):
        super(SKStage, self).__init__()

        self.blocks = nn.ModuleList([
            SKBlock(in_channels=in_channels, out_channels=out_channels,
                    cardinality=cardinality, group_depth=group_depth,
                    branches=branches, ratio=ratio, l=l,
                    downsample=downsample)
        ])
        for i in range(layers - 2):
            self.blocks.append(SKBlock(in_channels=out_channels, out_channels=out_channels,
                                       downsample=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SKNet(nn.Module):
    stage_layers_map = {50: [3, 4, 6, 3],
                        101: [3, 4, 23, 3],
                        152: [3, 4, 36, 3]}

    in_channels_map = [64, 256, 512, 1024]
    out_channels_map = [256, 512, 1024, 2048]

    def __init__(self, in_channels: int, layers: int, num_classes: int,
                 cardinality: int = 32, group_depth: int = 4,
                 branches: int = 2, ratio: int = 16, l: int = 32,
                 ):
        super(SKNet, self).__init__()

        assert layers in [50, 101, 152]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()

        self.stages = nn.ModuleList([
            SKStage(in_channels=self.in_channels_map[i], out_channels=self.out_channels_map[i],
                    cardinality=cardinality, group_depth=group_depth,
                    layers=self.stage_layers_map[layers][i], downsample=i != 0,
                    branches=branches, ratio=ratio, l=l,
                    )
            for i in range(4)
        ])

        self.head = ResNetClassifierHead(in_channels=self.out_channels_map[3], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.bn1(x)
        x = self.relu1(x)

        for stage in self.stages:
            x = stage(x)

        x = self.head(x)
        return x
