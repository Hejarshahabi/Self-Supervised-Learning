# dinov2/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class NumpyDINODataset(Dataset):
    """
    2 global crops (128) + 6 local crops (64)
    """
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*.npy"))

        self.global_trans = A.Compose([
            A.RandomResizedCrop(128, 128, scale=(0.4, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

        self.local_trans = A.Compose([
            A.RandomResizedCrop(64, 64, scale=(0.05, 0.4)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx]).astype(np.float32)
        img = img.transpose(1, 2, 0)  # (H, W, 6)

        views = []
        for _ in range(2):
            views.append(self.global_trans(image=img)['image'])
        for _ in range(6):
            views.append(self.local_trans(image=img)['image'])

        return views