"""IS-Net architecture used by the supplied BRIA RMBG 1.4 checkpoint."""
import torch
from torch import nn
from torch.nn import functional as F


def up(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=False)


class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1, stride=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate, stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU7(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        for i in range(1, 7):
            setattr(self, f"rebnconv{i}", REBNCONV(out_ch if i == 1 else mid_ch, mid_ch))
            if i <= 5:
                setattr(self, f"pool{i}", nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        for i in range(1, 7):
            setattr(self, f"rebnconv{i}d", REBNCONV(mid_ch * 2, out_ch if i == 1 else mid_ch))

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin); hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2)); hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(self.pool4(hx4)); hx6 = self.rebnconv6(self.pool5(hx5))
        hx7 = self.rebnconv7(hx6)
        d6 = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        d5 = self.rebnconv5d(torch.cat((up(d6, hx5), hx5), 1))
        d4 = self.rebnconv4d(torch.cat((up(d5, hx4), hx4), 1))
        d3 = self.rebnconv3d(torch.cat((up(d4, hx3), hx3), 1))
        d2 = self.rebnconv2d(torch.cat((up(d3, hx2), hx2), 1))
        d1 = self.rebnconv1d(torch.cat((up(d2, hx1), hx1), 1))
        return d1 + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        for i in range(1, 6):
            setattr(self, f"rebnconv{i}", REBNCONV(out_ch if i == 1 else mid_ch, mid_ch))
            if i <= 4: setattr(self, f"pool{i}", nn.MaxPool2d(2, 2, ceil_mode=True))
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        for i in range(1, 6): setattr(self, f"rebnconv{i}d", REBNCONV(mid_ch * 2, out_ch if i == 1 else mid_ch))

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); h2 = self.rebnconv2(self.pool1(h1))
        h3 = self.rebnconv3(self.pool2(h2)); h4 = self.rebnconv4(self.pool3(h3))
        h5 = self.rebnconv5(self.pool4(h4)); h6 = self.rebnconv6(h5)
        d5 = self.rebnconv5d(torch.cat((h6, h5), 1)); d4 = self.rebnconv4d(torch.cat((up(d5, h4), h4), 1))
        d3 = self.rebnconv3d(torch.cat((up(d4, h3), h3), 1)); d2 = self.rebnconv2d(torch.cat((up(d3, h2), h2), 1))
        d1 = self.rebnconv1d(torch.cat((up(d2, h1), h1), 1))
        return d1 + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        for i in range(1, 5):
            setattr(self, f"rebnconv{i}", REBNCONV(out_ch if i == 1 else mid_ch, mid_ch))
            if i <= 3: setattr(self, f"pool{i}", nn.MaxPool2d(2, 2, ceil_mode=True))
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        for i in range(1, 5): setattr(self, f"rebnconv{i}d", REBNCONV(mid_ch * 2, out_ch if i == 1 else mid_ch))

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); h2 = self.rebnconv2(self.pool1(h1))
        h3 = self.rebnconv3(self.pool2(h2)); h4 = self.rebnconv4(self.pool3(h3)); h5 = self.rebnconv5(h4)
        d4 = self.rebnconv4d(torch.cat((h5, h4), 1)); d3 = self.rebnconv3d(torch.cat((up(d4, h3), h3), 1))
        d2 = self.rebnconv2d(torch.cat((up(d3, h2), h2), 1)); d1 = self.rebnconv1d(torch.cat((up(d2, h1), h1), 1))
        return d1 + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch); self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch); self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x); h1 = self.rebnconv1(hxin); h2 = self.rebnconv2(self.pool1(h1))
        h3 = self.rebnconv3(self.pool2(h2)); h4 = self.rebnconv4(h3)
        d3 = self.rebnconv3d(torch.cat((h4, h3), 1)); d2 = self.rebnconv2d(torch.cat((up(d3, h2), h2), 1))
        d1 = self.rebnconv1d(torch.cat((up(d2, h1), h1), 1))
        return d1 + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch); self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4); self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2); self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x); h1 = self.rebnconv1(hxin); h2 = self.rebnconv2(h1)
        h3 = self.rebnconv3(h2); h4 = self.rebnconv4(h3)
        d3 = self.rebnconv3d(torch.cat((h4, h3), 1)); d2 = self.rebnconv2d(torch.cat((d3, h2), 1))
        d1 = self.rebnconv1d(torch.cat((d2, h1), 1))
        return d1 + hxin


class RMBG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage1 = RSU7(64, 32, 64); self.pool12 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128); self.pool23 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256); self.pool34 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512); self.pool45 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512); self.pool56 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        self.stage5d = RSU4F(1024, 256, 512); self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128); self.stage2d = RSU6(256, 32, 64); self.stage1d = RSU7(128, 16, 64)
        for name, ch in [("side1",64),("side2",64),("side3",128),("side4",256),("side5",512),("side6",512)]:
            setattr(self, name, nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x):
        hxin = self.conv_in(x); h1 = self.stage1(hxin); h2 = self.stage2(self.pool12(h1))
        h3 = self.stage3(self.pool23(h2)); h4 = self.stage4(self.pool34(h3))
        h5 = self.stage5(self.pool45(h4)); h6 = self.stage6(self.pool56(h5))
        d5 = self.stage5d(torch.cat((up(h6, h5), h5), 1)); d4 = self.stage4d(torch.cat((up(d5, h4), h4), 1))
        d3 = self.stage3d(torch.cat((up(d4, h3), h3), 1)); d2 = self.stage2d(torch.cat((up(d3, h2), h2), 1))
        d1 = self.stage1d(torch.cat((up(d2, h1), h1), 1))
        outputs = [self.side1(d1), self.side2(d2), self.side3(d3), self.side4(d4), self.side5(d5), self.side6(h6)]
        return [torch.sigmoid(up(o, x)) for o in outputs]
