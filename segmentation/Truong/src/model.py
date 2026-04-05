import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                               mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg  = torch.mean(x, dim=1, keepdim=True)
        mx,_ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class ResNet50UNetCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        bb        = models.resnet50(weights=None)
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.pool = bb.maxpool
        self.enc1 = bb.layer1
        self.enc2 = bb.layer2
        self.enc3 = bb.layer3
        self.enc4 = bb.layer4
        self.bottleneck = ConvBlock(2048, 1024)
        self.dec4 = UpBlock(1024, 1024, 512)
        self.cbam = CBAM(512)
        self.dec3 = UpBlock(512,  512,  256)
        self.dec2 = UpBlock(256,  256,  128)
        self.dec1 = UpBlock(128,  64,   64)
        self.final_up   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final_conv = nn.Sequential(
            ConvBlock(32, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        e0 = self.enc0(x);  p  = self.pool(e0)
        e1 = self.enc1(p);  e2 = self.enc2(e1)
        e3 = self.enc3(e2); e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.cbam(self.dec4(b, e3))
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        return self.final_conv(self.final_up(d1))

class ParamMLP(nn.Module):
    def __init__(self, input_dim=14, hidden=[64, 64, 32], output_dim=4):
        super().__init__()
        layers = []
        prev   = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h),
                       nn.BatchNorm1d(h),
                       nn.ReLU(),
                       nn.Dropout(0.2)]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

    def to_params(self, output, param_ranges):
        from config import Config
        out    = output.squeeze().cpu().detach().numpy()
        keys   = list(param_ranges.keys())
        params = {}
        for i, key in enumerate(keys):
            lo, hi      = param_ranges[key]
            params[key] = float(lo + out[i] * (hi - lo))
        params["sizepre"] = max(8, int(round(params["sizepre"] / 2) * 2))
        return params