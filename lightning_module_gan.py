import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import vgg16


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(vgg16(pretrained=True).features[:4].eval())
        blocks.append(vgg16(pretrained=True).features[4:9].eval())
        blocks.append(vgg16(pretrained=True).features[9:16].eval())
        blocks.append(vgg16(pretrained=True).features[16:23].eval())
        blocks.append(vgg16(pretrained=True).features[23:30].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)

    def forward(self, fakeFrame, frameY):
        fakeFrame = (fakeFrame - self.mean) / self.std
        frameY = (frameY - self.mean) / self.std
        loss = 0.0
        x = fakeFrame
        y = frameY
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
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
        self.content_loss = VGGPerceptualLoss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return [g_optimizer, d_optimizer], []

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
        self.log_dict({
            'g_loss_perceptual': g_loss_perceptual.detach(),
            'g_loss_content': content_loss.detach(),
            'g_loss_adv': adversarial_loss.detach()
            }, prog_bar=True)
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
        
        self.log('d_loss', d_loss.detach(), prog_bar=True)
        self.manual_backward(d_loss)
        opt_discriminator.step()
    
    def set_requires_grad(self, net, requires_grad):
        for p in net.parameters():
            p.requires_grad = requires_grad


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
        return torch.optim.Adam(self.parameters(), lr=self.lr)