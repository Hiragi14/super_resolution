import torch
import os
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()

        # VGG19 pretrained on ImageNet
        vgg = vgg19(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Extract relu5_4 (features.35)
        self.feature_extractor = create_feature_extractor(
            vgg,
            return_nodes={'35': 'feat'}  # features[35] is relu5_4
        )

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # L1 loss is more robust to outliers than MSE
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        """
        Args:
            sr: Super-resolved image (B, 3, H, W), values in [0,1]
            hr: High-resolution GT image (B, 3, H, W), values in [0,1]
        Returns:
            Perceptual loss between VGG features
        """
        # Normalize to ImageNet stats
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std

        # Extract VGG features
        sr_feat = self.feature_extractor(sr_norm)['feat']
        hr_feat = self.feature_extractor(hr_norm)['feat']

        # Compute perceptual loss
        loss = self.criterion(sr_feat, hr_feat)
        return loss


class ESRGANLightningModule(pl.LightningModule):
    def __init__(self, generator, discriminator, lr=1e-4, beta1=0.9, beta2=0.999):
        super(ESRGANLightningModule, self).__init__()
        self.automatic_optimization = False
        self.generator = generator
        self.discriminator = discriminator
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.content_loss = VGGLoss()
        # self.adversarial_loss = torch.nn.MSELoss() # torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3, betas=(self.beta1, self.beta2))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-5, betas=(self.beta1, self.beta2))
        
        g_scheduler = {
            "scheduler": CosineAnnealingLR(g_optimizer, T_max=300, eta_min=1e-6),
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_g"
        }
        d_scheduler = {
            "scheduler": StepLR(d_optimizer, step_size=50, gamma=0.5),
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_d"
        }
        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]

    def training_step(self, batch, batch_idx):
        lr_images, hr_images = batch
        opt_g, opt_d = self.optimizers()

        # ======== Generator Step ========
        self.set_requires_grad(self.discriminator, False)
        self.set_requires_grad(self.generator, True)
        opt_g.zero_grad()

        sr_images = self.generator(lr_images)
        d_real = self.discriminator(hr_images)
        d_fake = self.discriminator(sr_images)

        # VGG-based perceptual loss
        content_loss = self.content_loss(sr_images, hr_images)

        # === RaGAN Generator Loss ===
        d_real_mean = torch.mean(d_real)
        d_fake_mean = torch.mean(d_fake)

        g_loss = 0.5 * F.mse_loss(d_real - d_fake_mean, torch.ones_like(d_real)) + \
                0.5 * F.mse_loss(d_fake - d_real_mean, torch.zeros_like(d_fake))

        total_g_loss = content_loss + 0.001 * g_loss
        self.manual_backward(total_g_loss)
        opt_g.step()

        # ======== Discriminator Step ========
        self.set_requires_grad(self.generator, False)
        self.set_requires_grad(self.discriminator, True)
        opt_d.zero_grad()

        sr_images = self.generator(lr_images).detach()
        d_real = self.discriminator(hr_images)
        d_fake = self.discriminator(sr_images)

        d_real_mean = torch.mean(d_real)
        d_fake_mean = torch.mean(d_fake)

        d_loss = 0.5 * F.mse_loss(d_real - d_fake_mean, torch.ones_like(d_real)) + \
                0.5 * F.mse_loss(d_fake - d_real_mean, torch.zeros_like(d_fake))

        self.manual_backward(d_loss)
        opt_d.step()

        # Logging
        self.log_dict({
            'g_total_loss': total_g_loss.detach(),
            'g_content_loss': content_loss.detach(),
            'g_adv_loss': g_loss.detach(),
            'd_loss': d_loss.detach(),
            'lr_g': opt_g.param_groups[0]['lr'],
            'lr_d': opt_d.param_groups[0]['lr']
        }, prog_bar=True)

    
    def set_requires_grad(self, net, requires_grad):
        for p in net.parameters():
            p.requires_grad = requires_grad
    
    def on_train_end(self):
        super().on_train_end()
        log_dir = self.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        # Generatorの保存
        gen_path = os.path.join(log_dir, "g_final_weight.pt")
        torch.save(self.generator.state_dict(), gen_path)
        self.print(f"[INFO] Generator weights saved to {gen_path}")

        # Discriminatorの保存
        disc_path = os.path.join(log_dir, "d_final_weight.pt")
        torch.save(self.discriminator.state_dict(), disc_path)
        self.print(f"[INFO] Discriminator weights saved to {disc_path}")
        
    


class GeneratorLightningModule(pl.LightningModule):
    def __init__(self, generator, lr=1e-4):
        super(GeneratorLightningModule, self).__init__()
        self.generator = generator
        self.lr = lr
        self.criterion = torch.nn.L1Loss()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        lr_images, hr_images = batch
        sr_images = self(lr_images)
        loss = self.criterion(sr_images, hr_images)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "lr"
            }
        }
    
    def on_train_end(self):
        super().on_train_end()
        log_dir = self.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        # Generatorの保存
        gen_path = os.path.join(log_dir, "g_pre_weight.pt")
        torch.save(self.generator.state_dict(), gen_path)
        self.print(f"[INFO] Generator weights saved to {gen_path}")