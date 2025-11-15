# swav/finetune_seg.py
# IDENTICAL to SimCLR version, just load SwAV encoder

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
import segmentation_models_pytorch as smp


class NumpySegDataset(Dataset):
    def __init__(self, root_dir: str, tile_size: int = 128):
        self.root_dir = Path(root_dir)
        self.tile_size = tile_size
        self.img_files = sorted(self.root_dir.glob("img_*.npy"))
        self.mask_files = sorted(self.root_dir.glob("mask_*.npy"))

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img = np.load(self.img_files[idx]).astype(np.float32) / 10000.0
        mask = np.load(self.mask_files[idx]).astype(np.uint8)

        H, W = img.shape[1:]
        i = np.random.randint(0, H - self.tile_size)
        j = np.random.randint(0, W - self.tile_size)

        img = torch.from_numpy(img[:, i:i+self.tile_size, j:j+self.tile_size]).float()
        mask = torch.from_numpy(mask[i:i+self.tile_size, j:j+self.tile_size]).long()
        return img, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', type=str, default='checkpoints_seg')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = NumpySegDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True)

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=6, classes=2).to(device)

    # Load SwAV backbone
    swav_state = torch.load(args.pretrained, map_location=device)
    backbone_state = {k.replace("backbone.", ""): v for k, v in swav_state.items() if k.startswith("backbone.")}
    model.encoder.load_state_dict(backbone_state, strict=False)
    print("SwAV encoder loaded.")

    criterion = smp.losses.DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
        print(f"Epoch {epoch} | Dice Loss: {total_loss/len(loader):.4f}")

        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), f"{args.out}/seg_epoch{epoch}.pth")

    print("Fine-tuning done.")


if __name__ == "__main__":
    main()