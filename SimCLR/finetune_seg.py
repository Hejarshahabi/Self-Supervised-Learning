# simclr/finetune_seg.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
import segmentation_models_pytorch as smp


class NumpySegDataset(Dataset):
    """
    Loads:
      img_*.npy  → (6, H, W)
      mask_*.npy → (H, W), {0,1}
    """
    def __init__(self, root_dir: str, tile_size: int = 128):
        self.root_dir = Path(root_dir)
        self.tile_size = tile_size

        # Match img and mask by index in filename
        self.img_files = sorted(self.root_dir.glob("img_*.npy"))
        self.mask_files = sorted(self.root_dir.glob("mask_*.npy"))

        assert len(self.img_files) == len(self.mask_files), "Mismatched img/mask count"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        img = np.load(img_path).astype(np.float32)   # (6, H, W)
        mask = np.load(mask_path).astype(np.uint8)   # (H, W)

        H, W = img.shape[1:]

        # Random crop
        i = np.random.randint(0, H - self.tile_size)
        j = np.random.randint(0, W - self.tile_size)

        img_crop = img[:, i:i+self.tile_size, j:j+self.tile_size]
        mask_crop = mask[i:i+self.tile_size, j:j+self.tile_size]

        # Normalize (Sentinel-2 reflectance roughly 0–10000 → scale to 0–1)
        img_crop = img_crop / 10000.0
        img_crop = torch.from_numpy(img_crop).float()
        mask_crop = torch.from_numpy(mask_crop).long()

        return img_crop, mask_crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='finetune data folder')
    parser.add_argument('--pretrained', type=str, required=True, help='simclr_*.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', type=str, default='seg_checkpoints')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & Loader
    dataset = NumpySegDataset(args.data, tile_size=128)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True)

    # U-Net with 6-channel input
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=6,
        classes=2,
    ).to(device)

    # Load SimCLR encoder
    simclr_state = torch.load(args.pretrained, map_location=device)
    encoder_state = {k.replace("encoder.", ""): v
                     for k, v in simclr_state.items() if k.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded SimCLR encoder. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Loss & Optimizer
    criterion = smp.losses.DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:03d} | Dice Loss: {avg_loss:.4f}")

        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), f"{args.out}/seg_epoch{epoch}.pth")


if __name__ == "__main__":
    main()