# swav/train_swav.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from swav.model import SwAV
from swav.dataset import NumpySwAVDataset
from swav.loss import swav_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--n_proto', type=int, default=512)
    parser.add_argument('--queue_size', type=int, default=3840)
    parser.add_argument('--out', type=str, default='checkpoints_swav')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = NumpySwAVDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)

    # Model
    model = SwAV(feat_dim=128, n_prototypes=args.n_proto, queue_size=args.queue_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Queue
    use_queue = False
    queue = None

    print(f"Training SwAV on {len(dataset)} samples")

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for views in tqdm(loader, desc=f"Epoch {epoch}"):
            views = [v.to(device) for v in views]  # List of 6 crops

            # Forward
            z_list = [model(v) for v in views[:2]]  # Only global crops for loss

            # Queue update
            with torch.no_grad():
                if use_queue and queue is not None:
                    model.dequeue_and_enqueue(torch.cat(z_list, dim=0))

            loss = swav_loss(z_list, model, temperature=args.temp, queue=queue)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Start using queue after 10 epochs
            if epoch == 10 and not use_queue:
                use_queue = True
                queue = torch.zeros(args.queue_size, 128).to(device)

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt = os.path.join(args.out, f"swav_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"Saved: {ckpt}")

    print("SwAV training completed.")


if __name__ == "__main__":
    main()