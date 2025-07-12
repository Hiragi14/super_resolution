import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import vgg16


class SRGANLightningModule(pl.LightningModule):
    def __init__(self, generator, discriminator, lr=1e-4, beta1=0.9, beta2=0.999):
        super(SRGANLightningModule, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return [g_optimizer, d_optimizer], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        lr_images, hr_images = batch
        real_labels = torch.ones(hr_images.size(0), 1, device=hr_images.device)
        fake_labels = torch.zeros(hr_images.size(0), 1, device=hr_images.device)
        
        if optimizer_idx == 0:
            # Generator training step
            # fix the discriminator parameters to avoid updating them during generator training
            self.set_requires_grad(self.discriminator, False)
            self.set_requires_grad(self.generator, True)
            
            sr_images = self.generator(lr_images)
            d_pred = self.discriminator(sr_images.detach().clone())
            
            # TODO: Define loss functions
            content_loss = self.content_loss(sr_images, hr_images)
            adversarial_loss = self.adversarial_loss(d_pred, real_labels)
            g_loss_perceptual = content_loss + adversarial_loss*0.001
            self.log_dict({
                'g_loss_perceptual': g_loss_perceptual,
                'g_loss_content': content_loss,
                'g_loss_adv': adversarial_loss
                })
            return g_loss_perceptual
        
        if optimizer_idx == 1:
            # Discriminator training step
            # fix the generator parameters to avoid updating them during discriminator training
            self.set_requires_grad(self.discriminator, True)
            self.set_requires_grad(self.generator, False)
            
            sr_images = self.generator(lr_images)
            d_real = self.discriminator(hr_images)
            d_fake = self.discriminator(sr_images.detach().clone())
            
            real_loss = self.adversarial_loss(d_real, real_labels)
            fake_loss = self.adversarial_loss(d_fake, fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            
            self.log('d_loss', d_loss)
            return d_loss