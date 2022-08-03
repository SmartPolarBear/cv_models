import torch
import torch.nn as nn


class ConvBNReluLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 group: int,
                 act: bool = True):
        super(ConvBNReluLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=group)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.act = nn.Sequential()
        if act:
            self.act.append(nn.ReLU6())

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
