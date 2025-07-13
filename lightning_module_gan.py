import torch
import os
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


# class VGGPerceptualLoss(torch.nn.Module):
#     def __init__(self):
#         super(VGGPerceptualLoss, self).__init__()
#         blocks = []
#         blocks.append(vgg16(pretrained=True).features[:4].eval())
#         blocks.append(vgg16(pretrained=True).features[4:9].eval())
#         blocks.append(vgg16(pretrained=True).features[9:16].eval())
#         blocks.append(vgg16(pretrained=True).features[16:23].eval())
#         blocks.append(vgg16(pretrained=True).features[23:30].eval())
#         for bl in blocks:
#             for p in bl:
#                 p.requires_grad = False
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
#         self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)

#     def forward(self, fakeFrame, frameY):
#         fakeFrame = (fakeFrame - self.mean) / self.std
#         frameY = (frameY - self.mean) / self.std
#         loss = 0.0
#         x = fakeFrame
#         y = frameY
#         for block in self.blocks:
#             x = block(x)
#             y = block(y)
#             loss += torch.nn.functional.l1_loss(x, y)
#         return loss


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


class SRGANLightningModule(pl.LightningModule):
    def __init__(self, generator, discriminator, lr=1e-4, beta1=0.9, beta2=0.999):
        super(SRGANLightningModule, self).__init__()
        self.automatic_optimization = False
        self.generator = generator
        self.discriminator = discriminator
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.content_loss = VGGLoss()
        self.adversarial_loss = torch.nn.MSELoss() # torch.nn.BCEWithLogitsLoss()

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
        opt_generator, opt_discriminator = self.optimizers()
        
        # ---- Generator training step ----
        # fix the discriminator parameters to avoid updating them during generator training
        self.set_requires_grad(self.discriminator, False)
        self.set_requires_grad(self.generator, True)
        
        opt_generator.zero_grad()
        
        sr_images = self.generator(lr_images)
        d_pred = self.discriminator(sr_images)
        
        # define real and fake labels for the discriminator
        label_shape = d_pred.shape
        real_labels = torch.ones(label_shape, device=hr_images.device)
        
        # Calculate the content loss using VGG perceptual loss, adversarial loss using binary cross-entropy
        content_loss = self.content_loss(sr_images, hr_images)
        adversarial_loss = self.adversarial_loss(d_pred, real_labels)
        # Combine the content loss and adversarial loss
        # The weight for adversarial loss can be adjusted based on the desired balance(original paper uses 0.001)
        g_loss_perceptual = content_loss + adversarial_loss*0.001
        # Backward pass for generator
        self.manual_backward(g_loss_perceptual)
        opt_generator.step()
        
        # ---- Discriminator training step ----
        opt_discriminator.zero_grad()
        # fix the generator parameters to avoid updating them during discriminator training
        self.set_requires_grad(self.discriminator, True)
        self.set_requires_grad(self.generator, False)
        
        sr_images = self.generator(lr_images)
        d_real = self.discriminator(hr_images)
        d_fake = self.discriminator(sr_images.detach())
        
        # define real and fake labels for the discriminator
        real_labels = torch.ones(d_real.size(), device=hr_images.device)
        fake_labels = torch.zeros(d_fake.size(), device=hr_images.device)
        real_loss = self.adversarial_loss(d_real, real_labels)
        fake_loss = self.adversarial_loss(d_fake, fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_discriminator.step()
        
        self.log_dict({
            'g_loss_perceptual': g_loss_perceptual.detach(),
            'g_loss_content': content_loss.detach(),
            'g_loss_adv': adversarial_loss.detach(),
            'd_loss': d_loss.detach(),
            'lr_g': opt_generator.param_groups[0]["lr"],
            'lr_d': opt_discriminator.param_groups[0]["lr"]
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

        # Discriminatorの保存（必要なら）
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