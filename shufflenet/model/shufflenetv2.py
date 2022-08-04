import torch
import torch.nn as nn

from typing import Tuple, Final


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W).permute(0, 2, 1, 3, 4).reshape(B, C, H, W)

        return x


class ChannelSplit(nn.Module):
    def __init__(self, ratio: float):
        super(ChannelSplit, self).__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = int(x.shape[1] * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


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

        self.act = nn.Sequential() if not act else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class ShuffleNetBlock(nn.Module):
    def __init__(self, in_channels: int, split_ratio: float = 0.5):
        super(ShuffleNetBlock, self).__init__()
        self.split = ChannelSplit(ratio=split_ratio)

        conv_channels: int = int(split_ratio * in_channels)

        self.conv1 = ConvBNReluLayer(in_channels=conv_channels, out_channels=conv_channels)
        self.conv2 = ConvBNReluLayer(in_channels=conv_channels, out_channels=conv_channels,
                                     kernel_size=3, stride=1, padding=1,
                                     group=conv_channels, act=False)
        self.conv3 = ConvBNReluLayer(in_channels=conv_channels, out_channels=conv_channels)

        self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.split(x)

        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)

        y = torch.concat((x1, x2), dim=1)
        y = self.shuffle(y)

        return y


class ShuffleNetDownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ShuffleNetDownsampleBlock, self).__init__()

        mid_channels: int = out_channels // 2

        self.conv1 = ConvBNReluLayer(in_channels=in_channels, out_channels=mid_channels)
        self.conv2 = ConvBNReluLayer(in_channels=mid_channels, out_channels=mid_channels,
                                     kernel_size=3, stride=2, padding=1,
                                     group=mid_channels, act=False)
        self.conv3 = ConvBNReluLayer(in_channels=mid_channels, out_channels=mid_channels)

        self.shortcut = nn.Sequential(
            ConvBNReluLayer(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=3, stride=2, padding=1, act=False),
            ConvBNReluLayer(in_channels=in_channels, out_channels=mid_channels)
        )

        self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.shortcut(x)

        x2 = self.conv1(x)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)

        y = torch.concat((x1, x2), dim=1)
        y = self.shuffle(y)

        return y


class ShuffleNetV2(nn.Module):
    out_channels = {
        0.5: (48, 96, 192, 1024),
        1.0: (116, 232, 464, 1024),
        1.5: (176, 352, 704, 1024),
        2.0: {224, 488, 976, 2048}
    }

    blocks = (3, 7, 3)

    def __init__(self, in_channels: int, num_classes: int = 10,
                 net: float = 1.0, split_ratio: float = 0.5,
                 classifier_drop: float = 0.):
        super(ShuffleNetV2, self).__init__()

        channels: Final = self.out_channels[net]

        cur_channels = 24
        self.conv1 = ConvBNReluLayer(in_channels=in_channels, out_channels=cur_channels,
                                     kernel_size=3, padding=1, stride=1, act=False)

        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(
                ShuffleNetDownsampleBlock(in_channels=cur_channels, out_channels=channels[i])
            )
            self.layers.extend(
                ShuffleNetBlock(in_channels=channels[i], split_ratio=split_ratio)
                for _ in range(self.blocks[i])
            )
            cur_channels = channels[i]

        self.conv2 = ConvBNReluLayer(in_channels=channels[2], out_channels=channels[3],
                                     kernel_size=3, padding=1, stride=1, act=False)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=classifier_drop),
            nn.Linear(in_features=channels[3], out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        for layer in self.layers:
            x = layer(x)

        x = self.conv2(x)

        x = self.classifier(x)
        return x
