# swav/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class NumpySwAVDataset(Dataset):
    """
    Returns 2 global crops (128) + 4 local crops (64) → 6 views.
    """
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*.npy"))

        # Global crops (2x 128x128)
        self.global_trans = A.Compose([
            A.RandomResizedCrop(width=128, height=128, scale=(0.4, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

        # Local crops (4x 64x64)
        self.local_trans = A.Compose([
            A.RandomResizedCrop(width=64, height=64, scale=(0.05, 0.4), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx]).astype(np.float32)  # (6, H, W)
        img = img.transpose(1, 2, 0)  # (H, W, 6)

        views = []
        for _ in range(2):
            views.append(self.global_trans(image=img)['image'])
        for _ in range(4):
            views.append(self.local_trans(image=img)['image'])

        return views  # List[6, C, H, W]