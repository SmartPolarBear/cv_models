import torch
import torch.nn as nn

from resnet.model.resnet import ResNetClassifierHead


class MHSABlock(nn.Module):
    def __init__(self, in_channels: int, heads: int, w: int, d: int):
        super(MHSABlock, self).__init__()

        self.heads = heads

        self.wq = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=1, stride=1)
        self.wk = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=1, stride=1)
        self.wv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=1, stride=1)

        self.rh = nn.Parameter(torch.randn(1, heads, in_channels // heads, 1, d), requires_grad=True)
        self.rw = nn.Parameter(torch.randn(1, heads, in_channels // heads, w, 1), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        q = self.wq(x).view(B, self.heads, C // self.heads, -1)
        k = self.wk(x).view(B, self.heads, C // self.heads, -1)
        v = self.wv(x).view(B, self.heads, C // self.heads, -1)

        qk = torch.matmul(q.permute(0, 1, 3, 2), k)

        pos_emb = (self.rh + self.rw).view(1, self.heads, C // self.heads, -1)
        qr = torch.matmul(pos_emb.permute(0, 1, 3, 2), q)

        attn = self.softmax(qk + qr)

        y = torch.matmul(v, attn.permute(0, 1, 3, 2))
        y = y.view(B, C, H, W)

        return y


class ConvBNReLULayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 1, padding: int = 0,
                 stride: int = 1, act=False):
        super(ConvBNReLULayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding,
                              stride=stride)

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.act = nn.Sequential() if not act else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)

        x = self.act(x)

        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 mhsa: bool = False,
                 downsample: bool = False, **kwargs):
        super().__init__()

        self.conv1 = ConvBNReLULayer(in_channels=in_channels, out_channels=in_channels // 4,
                                     kernel_size=1, stride=1, padding=0, act=True)

        if not mhsa:
            self.conv2 = ConvBNReLULayer(in_channels=in_channels // 4, out_channels=in_channels // 4,
                                         kernel_size=3, stride=1 if not downsample else 2,
                                         padding=1, act=True)
        else:
            assert 'heads' in kwargs.keys()
            assert 'w' in kwargs.keys()
            assert 'h' in kwargs.keys()

            self.conv2 = nn.Sequential(
                MHSABlock(in_channels=in_channels // 4, heads=kwargs['heads'], w=kwargs['w'], d=kwargs['h']),
                nn.AvgPool2d(kernel_size=2, stride=2) if downsample else nn.Sequential(),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU()
            )

        self.conv3 = ConvBNReLULayer(in_channels=in_channels // 4, out_channels=out_channels,
                                     kernel_size=1, stride=1, padding=0, act=True)

        self.shortcut_conv = nn.Sequential() if not downsample and in_channels == out_channels else \
            ConvBNReLULayer(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, stride=1 if not downsample else 2, act=False)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        short = self.shortcut_conv(x)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = x + short
        x = self.relu(x)

        return x


class BoTNetStage(nn.Module):
    def __init__(self, image_size: int, in_channels: int,
                 out_channels: int, layers: int, heads: int = 4,
                 downsample: bool = True, mhsa: bool = True):
        super().__init__()

        feat_size = image_size // 32

        self.blocks = nn.ModuleList([
            BottleneckBlock(in_channels=in_channels, out_channels=out_channels,
                            downsample=downsample, mhsa=mhsa, w=feat_size, h=feat_size, heads=heads)
        ])
        for i in range(layers - 2):
            self.blocks.append(BottleneckBlock(in_channels=out_channels, out_channels=out_channels,
                                               downsample=False, mhsa=mhsa, w=feat_size, h=feat_size, heads=heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class BoTNet(nn.Module):
    stage_layers_map = {50: [3, 4, 6, 3],
                        101: [3, 4, 23, 3],
                        152: [3, 4, 36, 3]}

    in_channels_map = [64, 256, 512, 1024]
    out_channels_map = [256, 512, 1024, 2048]

    def __init__(self, image_size: int, in_channels: int, layers: int, num_classes: int, heads: int = 4):
        super(BoTNet, self).__init__()

        assert layers in [50, 101, 152]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()

        self.stages = nn.ModuleList([
            BoTNetStage(image_size=image_size, in_channels=self.in_channels_map[i],
                        out_channels=self.out_channels_map[i], layers=self.stage_layers_map[layers][i],
                        downsample=i != 0, mhsa=i == 3, heads=heads)
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
