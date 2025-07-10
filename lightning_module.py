import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from models.srcnn import SRCNN
from models.edsr import EDSR
from models.srresnet import SRResNet


def calculate_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    psnr = -10 * torch.log10(mse)
    return psnr

class SRCNNLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-4, scale=2, in_channels=3):
        super().__init__()
        self.model = SRCNN(in_channels=in_channels)
        self.loss_fn = torch.nn.L1Loss()
        self.save_hyperparameters()

        self.lr = lr
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.forward(lr)
        loss = self.loss_fn(sr, hr)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.forward(lr)
        loss = self.loss_fn(sr, hr)
        psnr = calculate_psnr(sr, hr)
        self.log('val_loss', loss)
        self.log('val_psnr', psnr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class EDSRLightningModule(pl.LightningModule):
    def __init__(self, scale=2, lr=1e-4):
        super().__init__()
        self.model = EDSR(scale=scale)
        self.criterion = torch.nn.L1Loss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = self.criterion(sr, hr)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = self.criterion(sr, hr)
        psnr = calculate_psnr(sr, hr)
        self.log('val_loss', loss)
        self.log('val_psnr', psnr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class SRResNetLightningModule(pl.LightningModule):
    def __init__(self, scale=2, lr=1e-4):
        super().__init__()
        self.model = SRResNet(upscale_factor=scale)  # Assuming SRResNet is similar to EDSR
        self.criterion = torch.nn.L1Loss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = self.criterion(sr, hr)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = self.criterion(sr, hr)
        psnr = calculate_psnr(sr, hr)
        self.log('val_loss', loss)
        self.log('val_psnr', psnr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)