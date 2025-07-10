import pytorch_lightning as pl
from lightning_module import SRResNetLightningModule
from datamodule import DIV2KDataModule
import torch
from datetime import datetime
from lightning.pytorch.loggers import CSVLogger

DATE = datetime.now().strftime('%Y%m%d-%H%M%S')
DIR_LOGS= './logs'
MODEL_NAME = 'srresnet_lightning'
MAXEPOCH = 300
SCALE = 4

def main():
    model = SRResNetLightningModule(scale=SCALE, lr=1e-4)
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
    save_path = f"{DIR_LOGS}/{MODEL_NAME}/{DATE}-final_weights.pth"
    torch.save(model.model.state_dict(), save_path)

if __name__ == '__main__':
    main()