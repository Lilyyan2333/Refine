import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)

        x = F.pad(
            x,
            [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2,
            ]
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_ch=16):
        super().__init__()
        bottleneck_ch = base_ch * 8

        self.inc = DoubleConv3D(in_channels, base_ch)
        self.down1 = Down3D(base_ch, base_ch * 2)
        self.down2 = Down3D(base_ch * 2, base_ch * 4)
        self.down3 = Down3D(base_ch * 4, bottleneck_ch)

        self.up1 = Up3D(bottleneck_ch, base_ch * 4, base_ch * 4)
        self.up2 = Up3D(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3 = Up3D(base_ch * 2, base_ch, base_ch)

        self.outc = nn.Conv3d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits