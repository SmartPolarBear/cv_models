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
