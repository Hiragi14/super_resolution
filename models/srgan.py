import torch
import torch.nn as nn
import torch.nn.functional as F
from .srresnet import SRResNet


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=5, upscale_factor=4):
        super(Generator, self).__init__()
        self.srresnet = SRResNet(in_channels, out_channels, num_residual_blocks, upscale_factor)

    def forward(self, x):
        return self.srresnet(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super(Discriminator, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.model = nn.Sequential(
            self._conv_block(num_filters, 64, kernel_size=3, stride=2, padding=1),
            self._conv_block(64, 128, kernel_size=3, stride=2, padding=1),
            self._conv_block(128, 128, kernel_size=3, stride=2, padding=1),
            self._conv_block(128, 256, kernel_size=3, stride=2, padding=1),
            self._conv_block(256, 256, kernel_size=3, stride=2, padding=1),
            self._conv_block(256, 512, kernel_size=3, stride=2, padding=1),
            self._conv_block(512, 512, kernel_size=3, stride=2, padding=1),
            self._dence_block()
        )
    
    def forward(self, x):
        x = self.in_conv(x)
        x = self.model(x)
        return x

    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels) if out_channels != 1 else nn.Identity()
        )
    
    def _dence_block(self):
        return nn.Sequential(
            nn.Linear(16*16*512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )