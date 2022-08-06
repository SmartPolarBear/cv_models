import torch
import torch.nn as nn


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

        self.rh = nn.Parameter(torch.randn(1, heads, in_channels // heads, 1, d))
        self.rw = nn.Parameter(torch.randn(1, heads, in_channels // heads, w, 1))

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
