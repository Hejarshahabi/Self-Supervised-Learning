# dinov2/train_dinov2.py
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dinov2.model import DINOv2
from dinov2.dataset import NumpyDINODataset
from dinov2.loss import dino_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.996)
    parser.add_argument('--out', type=str, default='checkpoints_dinov2')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = NumpyDINODataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)

    # Model
    model = DINOv2().to(device)
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training DINOv2 on {len(dataset)} samples")

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for views in tqdm(loader, desc=f"Epoch {epoch}"):
            views = [v.to(device) for v in views]

            # Global crops for student
            student_views = views[:2]
            # All crops for teacher
            teacher_views = views

            loss = 0.0
            for s_view in student_views:
                for t_view in teacher_views:
                    s_out, t_out = model(s_view, t_view)
                    loss += dino_loss(s_out, t_out)
            loss /= (len(student_views) * len(teacher_views))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher
            model.update_teacher(args.momentum)
            model.update_center(t_out)

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt = os.path.join(args.out, f"dinov2_epoch{epoch}.pth")
            torch.save(model.student.state_dict(), ckpt)
            print(f"Saved: {ckpt}")

    print("DINOv2 training completed.")


if __name__ == "__main__":
    main()