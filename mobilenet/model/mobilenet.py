import torch
import torch.nn as nn

from typing import Final


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReluLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 group: int = 1,
                 act: bool = True):
        super(ConvBNReluLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=group)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.act = nn.Sequential() if not act else nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class InvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 expansion_ratio: float = 6.0,
                 stride=1):
        super(InvertedBottleneckBlock, self).__init__()

        expanded: int = int(expansion_ratio * in_channels)

        self.pw1 = ConvBNReluLayer(in_channels=in_channels, out_channels=expanded)

        self.dw = ConvBNReluLayer(in_channels=expanded, out_channels=expanded,
                                  kernel_size=3, stride=stride,
                                  padding=1, group=expanded)

        self.pw2 = ConvBNReluLayer(in_channels=expanded, out_channels=out_channels, act=False)

        self.use_shortcut = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)

        if self.use_shortcut:
            x = x + residual

        return x


class MobileNetV2(nn.Module):
    configs: Final = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 1, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, in_channels=3, classes: int = 4,
                 width_mul: float = 1.0, round_nearest: int = 8,
                 classifier_drop=0.):
        super(MobileNetV2, self).__init__()

        first_channels = _make_divisible(32 * width_mul, round_nearest)
        last_channels = _make_divisible(1280 * width_mul, round_nearest)

        self.conv1 = ConvBNReluLayer(in_channels=in_channels, out_channels=first_channels, stride=2)

        self.layers = nn.ModuleList()
        input_channels = first_channels
        for t, c, n, s in self.configs:
            out_channels = _make_divisible(c * width_mul, round_nearest)
            for i in range(n):
                self.layers.append(
                    InvertedBottleneckBlock(in_channels=input_channels, out_channels=out_channels,
                                            stride=s if i == 0 else 1, expansion_ratio=t)
                )
                input_channels = t

        self.conv2 = ConvBNReluLayer(in_channels=input_channels, out_channels=last_channels, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_drop),
            nn.Linear(last_channels, classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        for layer in self.layers:
            x = layer(x)

        x = self.conv2(x)

        x = self.gap(x)
        x = self.classifier(x)

        return x
