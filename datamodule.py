import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import random
import numpy as np

def paired_random_crop(img_gt, img_lq, gt_patch_size, scale):
    """Crop aligned patches from GT and LQ images.
    Args:
        img_gt (ndarray): HR image (H, W, C)
        img_lq (ndarray): LR image (h, w, C)
        gt_patch_size (int): cropped size for GT image
        scale (int): scale factor between HR and LR
    """
    h_lq, w_lq = img_lq.shape[:2]
    lq_patch_size = gt_patch_size // scale

    # ランダムにLQの左上座標を決定
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # LQを切り出し
    img_lq_patch = img_lq[top:top + lq_patch_size, left:left + lq_patch_size, ...]

    # 対応するGTを切り出し
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gt_patch = img_gt[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]

    return img_gt_patch, img_lq_patch


class DIV2KDataset_Y(Dataset):
    def __init__(self, root_dir, split='train', scale=2, gt_size=96):
        self.split = split
        self.scale = scale
        self.gt_size = gt_size
        if split == 'train':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_train_HR')
            self.lr_dir = os.path.join(root_dir, f'DIV2K_train_LR_bicubic/X{scale}')
            self.indices = range(1, 801)
        elif split == 'val':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_valid_HR')
            self.lr_dir = os.path.join(root_dir, f'DIV2K_valid_LR_bicubic/X{scale}')
            self.indices = range(801, 901)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_id = f"{self.indices[idx]:04d}"
        lr_path = os.path.join(self.lr_dir, f"{img_id}x{self.scale}.png")
        hr_path = os.path.join(self.hr_dir, f"{img_id}.png")

        # PIL → numpy
        img_lq = np.array(Image.open(lr_path).convert('YCbCr'))[..., 0]  # Y channel
        img_gt = np.array(Image.open(hr_path).convert('YCbCr'))[..., 0]  # Y channel

        if self.split == 'train':
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, self.scale)

        # numpy → tensor
        img_gt = self.to_tensor(img_gt)
        img_lq = self.to_tensor(img_lq)

        return img_lq, img_gt


class DIV2KDataset(Dataset):
    def __init__(self, root_dir, split='train', scale=2, gt_size=96):
        self.split = split
        self.scale = scale
        self.gt_size = gt_size
        if split == 'train':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_train_HR')
            self.lr_dir = os.path.join(root_dir, f'DIV2K_train_LR_bicubic/X{scale}')
            self.indices = range(1, 801)
        elif split == 'val':
            self.hr_dir = os.path.join(root_dir, 'DIV2K_valid_HR')
            self.lr_dir = os.path.join(root_dir, f'DIV2K_valid_LR_bicubic/X{scale}')
            self.indices = range(801, 901)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_id = f"{self.indices[idx]:04d}"
        lr_path = os.path.join(self.lr_dir, f"{img_id}x{self.scale}.png")
        hr_path = os.path.join(self.hr_dir, f"{img_id}.png")

        # PIL → numpy
        img_lq = np.array(Image.open(lr_path).convert('RGB'))
        img_gt = np.array(Image.open(hr_path).convert('RGB'))

        if self.split == 'train':
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, self.scale)

        # numpy → tensor
        img_gt = self.to_tensor(img_gt)
        img_lq = self.to_tensor(img_lq)

        return img_lq, img_gt


class DIV2KDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data/DIV2K', scale=2, batch_size=16, num_workers=4, gt_size=96):
        super().__init__()
        self.data_dir = data_dir
        self.scale = scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gt_size = gt_size

    def setup(self, stage=None):
        self.train_ds = DIV2KDataset(self.data_dir, split='train', scale=self.scale, gt_size=self.gt_size)
        self.val_ds = DIV2KDataset(self.data_dir, split='val', scale=self.scale, gt_size=self.gt_size)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=self.num_workers)
