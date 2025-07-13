import pytorch_lightning as pl
from models.srgan import Generator, Discriminator
from lightning_module_gan import SRGANLightningModule, GeneratorLightningModule
from datamodule import DIV2KDataModule
import torch
from lightning.pytorch.loggers import CSVLogger

DIR_LOGS= './logs'
MODEL_NAME = 'srgan_lightning'
MAXEPOCH = 300
SCALE = 2

def main():
    generator = Generator(upscale_factor=SCALE)
    discriminator = Discriminator()
    data = DIV2KDataModule(data_dir='/ldisk/DeepLearning/Dataset/DIV2K/', batch_size=16, scale=SCALE, num_workers=4)
    
    # ---- Generator Training ----
    g_train_model = GeneratorLightningModule(generator=generator)
    trainer = pl.Trainer(
        logger=CSVLogger(DIR_LOGS, name=MODEL_NAME),
        accelerator="gpu",
        devices=1,
        max_epochs=200,
        check_val_every_n_epoch=1,
        precision=16,  # Enable mixed precision training
    )
    trainer.fit(g_train_model, datamodule=data)
    save_path = f"{DIR_LOGS}/{MODEL_NAME}/g_pre_weights.pth"
    torch.save(g_train_model.generator.state_dict(), save_path)
    # g_model = torch.load("./logs/srgan_lightning/version_5/g_pre_weights.pth")
    # generator.load_state_dict(g_model)
    
    # ---- SRGAN Training ----
    model = SRGANLightningModule(generator=g_train_model.generator, discriminator=discriminator)
    # model = SRGANLightningModule(generator=generator, discriminator=discriminator)
    trainer = pl.Trainer(
        logger=CSVLogger(DIR_LOGS, name=MODEL_NAME),
        accelerator="gpu",
        devices=1,
        max_epochs=MAXEPOCH,
        check_val_every_n_epoch=1,
        precision=16,  # Enable mixed precision training
    )
    trainer.fit(model, datamodule=data)
    save_path_g = f"{DIR_LOGS}/{MODEL_NAME}/g_final_weights.pth"
    save_path_d = f"{DIR_LOGS}/{MODEL_NAME}/d_final_weights.pth"
    torch.save(model.discriminator.state_dict(), save_path_d)
    torch.save(model.generator.state_dict(), save_path_g)

if __name__ == '__main__':
    main()