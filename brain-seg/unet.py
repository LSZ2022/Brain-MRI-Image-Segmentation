from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()

        def _block(in_channels, features):
            return nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, 3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            )

        f = init_features
        self.enc1 = _block(in_channels, f)
        self.dnsamp1 = nn.MaxPool2d(2, 2)
        self.enc2 = _block(f, f*2)
        self.dnsamp2 = nn.MaxPool2d(2, 2)
        self.enc3 = _block(f*2, f*4)
        self.dnsamp3 = nn.MaxPool2d(2, 2)
        self.enc4 = _block(f*4, f*8)
        self.dnsamp4 = nn.MaxPool2d(2, 2)

        self.bottleneck = _block(f*8, f*16)

        self.upsamp4 = nn.ConvTranspose2d(f*16, f*8, 2, 2)
        self.dec4 = _block(f*8*2, f*8)
        self.upsamp3 = nn.ConvTranspose2d(f*8, f*4, 2, 2)
        self.dec3 = _block(f*4*2, f*4)
        self.upsamp2 = nn.ConvTranspose2d(f*4, f*2, 2, 2)
        self.dec2 = _block(f*2*2, f*2)
        self.upsamp1 = nn.ConvTranspose2d(f*2, f, 2, 2)
        self.dec1 = _block(f*2, f)

        self.conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.dnsamp1(e1))
        e3 = self.enc3(self.dnsamp2(e2))
        e4 = self.enc4(self.dnsamp3(e3))
        b = self.bottleneck(self.dnsamp4(e4))

        d4 = self.upsamp4(b)
        d4 = torch.cat([d4, e4], 1)
        d3 = self.upsamp3(self.dec4(d4))
        d3 = torch.cat([d3, e3], 1)
        d2 = self.upsamp2(self.dec3(d3))
        d2 = torch.cat([d2, e2], 1)
        d1 = self.upsamp1(self.dec2(d2))
        d1 = torch.cat([d1, e1], 1)
        return torch.sigmoid(self.conv(self.dec1(d1)))

# ==============================
# 👇 已修复：PixelUnshuffle 通道匹配版本
# ==============================

# 下采样对比
class UNet_MaxPool(UNet):
    pass

class UNet_StridedConv(UNet):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__(in_channels, out_channels, init_features)
        f = init_features
        self.dnsamp1 = nn.Conv2d(f, f, 2, 2)
        self.dnsamp2 = nn.Conv2d(f*2, f*2, 2, 2)
        self.dnsamp3 = nn.Conv2d(f*4, f*4, 2, 2)
        self.dnsamp4 = nn.Conv2d(f*8, f*8, 2, 2)

class UNet_PixelUnshuffle(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        f = init_features

        def _block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.enc1 = _block(in_channels, f)
        self.pool1 = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(f*4, f*2, 1))

        self.enc2 = _block(f*2, f*2)
        self.pool2 = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(f*2*4, f*4, 1))

        self.enc3 = _block(f*4, f*4)
        self.pool3 = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(f*4*4, f*8, 1))

        self.enc4 = _block(f*8, f*8)
        self.pool4 = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(f*8*4, f*16, 1))

        self.bottleneck = _block(f*16, f*16)

        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, 2)
        self.dec4 = _block(f*8*2, f*8)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, 2)
        self.dec3 = _block(f*4*2, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, 2)
        self.dec2 = _block(f*2*2, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, 2)
        self.dec1 = _block(f*2, f)

        self.out = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], 1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], 1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out(d1))

# 上采样对比
class UNet_ConvTranspose(UNet):
    pass

class UNet_Bilinear(UNet):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__(in_channels, out_channels, init_features)
        self.upsamp4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamp3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamp2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamp1 = nn.UpsamplingBilinear2d(scale_factor=2)

class UNet_PixelShuffle(UNet):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__(in_channels, out_channels, init_features)
        f = init_features
        self.upsamp4 = nn.Sequential(nn.Conv2d(f*16, f*8*4, 1), nn.PixelShuffle(2))
        self.upsamp3 = nn.Sequential(nn.Conv2d(f*8, f*4*4, 1), nn.PixelShuffle(2))
        self.upsamp2 = nn.Sequential(nn.Conv2d(f*4, f*2*4, 1), nn.PixelShuffle(2))
        self.upsamp1 = nn.Sequential(nn.Conv2d(f*2, f*4, 1), nn.PixelShuffle(2))

# 1. 原始：ConvTranspose2d
class UNet_Up_ConvTrans(UNet):
    pass

# 2. Bilinear 上采样
# 2. Bilinear 上采样 【✅ 完美修复通道报错版本】
class UNet_Up_Bilinear(UNet):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__(in_channels, out_channels, init_features)
        f = init_features

        #  Bilinear 只会放大尺寸，不会改通道 → 必须加 1x1 卷积降通道
        self.upsamp4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(f * 16, f * 8, kernel_size=1)  # 通道从 512 → 256
        )
        self.upsamp3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(f * 8, f * 4, kernel_size=1)   # 通道从 256 → 128
        )
        self.upsamp2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(f * 4, f * 2, kernel_size=1)   # 通道从 128 → 64
        )
        self.upsamp1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(f * 2, f, kernel_size=1)       # 通道从 64 → 32
        )
# 3. PixelShuffle 上采样（完美修复通道）
class UNet_Up_PixelShuffle(UNet):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__(in_channels, out_channels, init_features)
        f = init_features
        self.upsamp4 = nn.Sequential(nn.Conv2d(f*16, f*8 * 4, 1), nn.PixelShuffle(2))
        self.upsamp3 = nn.Sequential(nn.Conv2d(f*8, f*4 * 4, 1), nn.PixelShuffle(2))
        self.upsamp2 = nn.Sequential(nn.Conv2d(f*4, f*2 * 4, 1), nn.PixelShuffle(2))
        self.upsamp1 = nn.Sequential(nn.Conv2d(f*2, f * 4, 1), nn.PixelShuffle(2))