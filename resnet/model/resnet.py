import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4,
                               kernel_size=1, stride=1,
                               padding='same')
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=in_channels // 4, out_channels=out_channels // 4,
                               kernel_size=3, stride=1 if not downsample else 2,
                               padding='same')
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=in_channels // 4, out_channels=out_channels,
                               kernel_size=1, stride=1,
                               padding='same')
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut_conv = nn.Sequential() if not downsample and in_channels == out_channels else \
            nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, stride=1 if not downsample else 2,
                                    padding='same'),
                          nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.shortcut_conv(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + short
        x = self.relu(x)

        return x


class ResNetStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int, downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            BottleneckBlock(in_channels=in_channels, out_channels=out_channels, downsample=downsample)
        ])
        for i in range(layers - 2):
            self.blocks.append(BottleneckBlock(in_channels=out_channels, out_channels=out_channels,
                                               downsample=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ResNetClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(ResNetClassifierHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=in_channels, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)
        x = self.linear(x)
        return x


class ResNet(nn.Module):
    stage_layers_map = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 4, 36, 3]}

    in_channels_map = [64, 256, 512, 1024]
    out_channels_map = [256, 512, 1024, 2048]

    def __init__(self, in_channels: int, layers: int, num_classes: int):
        super(ResNet, self).__init__()

        assert layers in [50, 101, 152]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()

        self.stages = nn.ModuleList([
            ResNetStage(in_channels=self.in_channels_map[i], out_channels=self.out_channels_map[i],
                        layers=self.stage_layers_map[layers][i], downsample=i != 0)
            for i in range(4)
        ])

        self.head = ResNetClassifierHead(in_channels=self.out_channels_map[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.bn1(x)
        x = self.relu1(x)

        for stage in self.stages:
            x = stage(x)

        x = self.head(x)
        return x
