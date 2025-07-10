import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    def forward(self, x):
        out = self.rb(x)
        out = torch.add(x, out)
        return out


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )
    def forward(self, x):
        out = self.upsample_block(x)
        return out


class SRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=16, upscale_factor=4):
        super(SRResNet, self).__init__()
        assert upscale_factor in [2, 4, 8], "Upscale factor must be one of [2, 4, 8]."
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        self.res_out_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample_blocks = nn.Sequential(
            *[UpSampleBlock(64, 64) for _ in range(upscale_factor // 2)]
        )
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        residual = x
        x = self.residual_blocks(x)
        x = self.res_out_conv(x)
        x = torch.add(x, residual)
        x = self.upsample_blocks(x)
        x = self.out_conv(x)
        return x