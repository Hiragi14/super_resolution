import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.block(x)

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats):
        m = []
        if scale in [2, 3]:
            m += [nn.Conv2d(n_feats, n_feats * scale * scale, 3, 1, 1), nn.PixelShuffle(scale)]
        elif scale == 4:
            for _ in range(2):
                m += [nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1), nn.PixelShuffle(2)]
        else:
            raise NotImplementedError(f'Unsupported scale: {scale}')
        super().__init__(*m)

class EDSR(nn.Module):
    def __init__(self, scale=2, n_resblocks=16, n_feats=64):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)
        body = [ResidualBlock(n_feats) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(Upsampler(scale, n_feats), nn.Conv2d(n_feats, 3, 3, 1, 1))

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x
