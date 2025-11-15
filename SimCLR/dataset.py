# simclr/dataset.py
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class NumpySimCLRDataset(Dataset):
    """
    Loads .npy files with shape (6, H, W).
    Returns two augmented views for SimCLR.
    """
    def __init__(self, root_dir: str, tile_size: int = 128):
        self.root_dir = Path(root_dir)
        self.tile_size = tile_size
        self.files = sorted(list(self.root_dir.glob("*.npy")))

        self.aug = A.Compose([
            A.RandomCrop(width=tile_size, height=tile_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        img = np.load(path).astype(np.float32)  # (6, H, W)

        # Two independent views
        aug1 = self.aug(image=img.transpose(1, 2, 0))  # A expects (H, W, C)
        aug2 = self.aug(image=img.transpose(1, 2, 0))

        view1 = aug1['image']  # (C, H, W)
        view2 = aug2['image']

        return view1, view2