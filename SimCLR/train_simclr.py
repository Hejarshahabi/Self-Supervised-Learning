# simclr/train_simclr.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import SimCLR
from dataset import NumpySimCLRDataset
from loss import nt_xent_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='root folder with sample subfolders')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--out', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- Data ----------
    dataset = NumpySimCLRDataset(root_dir=args.data, tile_size=128)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=8, pin_memory=True,
                        drop_last=True)

    # ---------- Model ----------
    model = SimCLR(dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------- Training ----------
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for (view1, view2) in tqdm(loader, desc=f'Epoch {epoch}'):
            view1, view2 = view1.to(device), view2.to(device)

            _, z1 = model(view1)
            _, z2 = model(view2)

            loss = nt_xent_loss(z1, z2, temperature=args.temp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch:03d} | Loss: {avg_loss:.4f}')

        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(),
                       f'{args.out}/simclr_epoch{epoch}.pth')

if __name__ == '__main__':
    main()