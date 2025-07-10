import pytorch_lightning as pl
from lightning_module import SRCNNLightningModule
from datamodule import DIV2KDataModule
import torch
from lightning.pytorch.loggers import CSVLogger

DIR_LOGS= './logs'
MODEL_NAME = 'srcnn_lightning_Ychannel'
MAXEPOCH = 50
SCALE = 3

def main():
    model = SRCNNLightningModule(lr=1e-4, scale=SCALE, in_channels=1)
    data = DIV2KDataModule(data_dir='/ldisk/DeepLearning/Dataset/DIV2K/', batch_size=16, scale=SCALE, num_workers=4)

    trainer = pl.Trainer(
        logger=CSVLogger(DIR_LOGS, name=MODEL_NAME),
        accelerator="gpu",
        devices=1,
        max_epochs=MAXEPOCH,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, datamodule=data)

    # ★ 最終エポックの重みを .pth で保存
    save_path = f"{DIR_LOGS}/{MODEL_NAME}/final_weights.pth"
    torch.save(model.model.state_dict(), save_path)

if __name__ == '__main__':
    main()
