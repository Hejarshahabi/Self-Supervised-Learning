#!/usr/bin/env python3
"""
Unified script to train and fine-tune SSL models (DINOv2, SimCLR, SwAV).
"""
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))


def train_dinov2(args):
    """Train DINOv2 model."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import sys
    import importlib.util
    # Import DINOv2 modules directly by file path
    dinov2_dir = Path(__file__).parent / 'DINOv2'
    spec_model = importlib.util.spec_from_file_location("model", dinov2_dir / "model.py")
    model_module = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(model_module)
    
    spec_dataset = importlib.util.spec_from_file_location("dataset", dinov2_dir / "dataset.py")
    dataset_module = importlib.util.module_from_spec(spec_dataset)
    spec_dataset.loader.exec_module(dataset_module)
    
    spec_loss = importlib.util.spec_from_file_location("loss", dinov2_dir / "loss.py")
    loss_module = importlib.util.module_from_spec(spec_loss)
    spec_loss.loader.exec_module(loss_module)
    
    DINOv2 = model_module.DINOv2
    NumpyDINODataset = dataset_module.NumpyDINODataset
    dino_loss = loss_module.dino_loss
    
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = NumpyDINODataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)
    
    model = DINOv2().to(device)
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"Training DINOv2 on {len(dataset)} samples")
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for views in tqdm(loader, desc=f"Epoch {epoch}"):
            views = [v.to(device) for v in views]
            
            student_views = views[:2]
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


def train_simclr(args):
    """Train SimCLR model."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import sys
    simclr_path = str(Path(__file__).parent / 'SimCLR')
    sys.path.insert(0, simclr_path)
    from model import SimCLR
    from dataset import NumpySimCLRDataset
    from loss import nt_xent_loss
    
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = NumpySimCLRDataset(root_dir=args.data, tile_size=128)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=8, pin_memory=True,
                        drop_last=True)
    
    model = SimCLR(dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
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
    
    print("SimCLR training completed.")


def train_swav(args):
    """Train SwAV model."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import sys
    import importlib.util
    # Import SwAV modules directly by file path
    swav_dir = Path(__file__).parent / 'SwAV'
    spec_model = importlib.util.spec_from_file_location("model", swav_dir / "model.py")
    model_module = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(model_module)
    
    spec_dataset = importlib.util.spec_from_file_location("dataset", swav_dir / "dataset.py")
    dataset_module = importlib.util.module_from_spec(spec_dataset)
    spec_dataset.loader.exec_module(dataset_module)
    
    spec_loss = importlib.util.spec_from_file_location("loss", swav_dir / "loss.py")
    loss_module = importlib.util.module_from_spec(spec_loss)
    spec_loss.loader.exec_module(loss_module)
    
    SwAV = model_module.SwAV
    NumpySwAVDataset = dataset_module.NumpySwAVDataset
    swav_loss = loss_module.swav_loss
    
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = NumpySwAVDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)
    
    model = SwAV(feat_dim=128, n_prototypes=args.n_proto, queue_size=args.queue_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    use_queue = False
    queue = None
    
    print(f"Training SwAV on {len(dataset)} samples")
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for views in tqdm(loader, desc=f"Epoch {epoch}"):
            views = [v.to(device) for v in views]
            
            z_list = [model(v) for v in views[:2]]
            
            with torch.no_grad():
                if use_queue and queue is not None:
                    model.dequeue_and_enqueue(torch.cat(z_list, dim=0))
            
            loss = swav_loss(z_list, model, temperature=args.temp, queue=queue)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
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


def finetune_dinov2(args):
    """Fine-tune DINOv2 for segmentation."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np
    import segmentation_models_pytorch as smp
    import sys
    import importlib.util
    import types
    
    # Import DINOv2 modules directly by file path
    dinov2_dir = Path(__file__).parent / 'DINOv2'
    parent_dir = Path(__file__).parent
    
    # Load model module
    spec_model = importlib.util.spec_from_file_location("dinov2.model", dinov2_dir / "model.py")
    model_module = importlib.util.module_from_spec(spec_model)
    sys.modules['dinov2.model'] = model_module
    spec_model.loader.exec_module(model_module)
    
    # Create dinov2 package module
    dinov2_pkg = types.ModuleType('dinov2')
    dinov2_pkg.model = model_module
    sys.modules['dinov2'] = dinov2_pkg
    
    # Now load finetune_seg which can import from dinov2.model
    spec_finetune = importlib.util.spec_from_file_location("dinov2.finetune_seg", dinov2_dir / "finetune_seg.py")
    finetune_module = importlib.util.module_from_spec(spec_finetune)
    spec_finetune.loader.exec_module(finetune_module)
    
    ViT = model_module.ViT
    NumpySegDataset = finetune_module.NumpySegDataset
    
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = NumpySegDataset(args.data, tile_size=128)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True)
    
    encoder = ViT(img_size=128, patch_size=8, in_chans=6).to(device)
    dinov2_state = torch.load(args.pretrained, map_location=device)
    encoder.load_state_dict(dinov2_state)
    print("DINOv2 encoder loaded.")
    
    model = smp.Unet(
        encoder_name="mit_b0",
        encoder_weights=None,
        in_channels=6,
        classes=2,
    ).to(device)
    
    model.encoder = encoder
    
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
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
        
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), f"{args.out}/seg_epoch{epoch}.pth")
    
    print("Fine-tuning done.")


def finetune_simclr(args):
    """Fine-tune SimCLR for segmentation."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import segmentation_models_pytorch as smp
    import sys
    simclr_path = str(Path(__file__).parent / 'SimCLR')
    sys.path.insert(0, simclr_path)
    from finetune_seg import NumpySegDataset
    
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = NumpySegDataset(args.data, tile_size=128)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True)
    
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=6,
        classes=2,
    ).to(device)
    
    simclr_state = torch.load(args.pretrained, map_location=device)
    encoder_state = {k.replace("encoder.", ""): v
                     for k, v in simclr_state.items() if k.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded SimCLR encoder. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
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
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:03d} | Dice Loss: {avg_loss:.4f}")
        
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), f"{args.out}/seg_epoch{epoch}.pth")
    
    print("Fine-tuning done.")


def finetune_swav(args):
    """Fine-tune SwAV for segmentation."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import segmentation_models_pytorch as smp
    import sys
    import importlib.util
    import types
    
    # Import SwAV modules - finetune_seg doesn't import swav.model, so simpler
    swav_dir = Path(__file__).parent / 'SwAV'
    sys.path.insert(0, str(swav_dir))
    
    # Load finetune_seg module
    spec_finetune = importlib.util.spec_from_file_location("finetune_seg", swav_dir / "finetune_seg.py")
    finetune_module = importlib.util.module_from_spec(spec_finetune)
    spec_finetune.loader.exec_module(finetune_module)
    
    NumpySegDataset = finetune_module.NumpySegDataset
    
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = NumpySegDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True)
    
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=6, classes=2).to(device)
    
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


def main():
    parser = argparse.ArgumentParser(
        description='Train or fine-tune SSL models (DINOv2, SimCLR, SwAV)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train DINOv2
  python run.py --model dinov2 --mode train --data /path/to/data --epochs 100
  
  # Fine-tune SimCLR for segmentation
  python run.py --model simclr --mode finetune --data /path/to/seg_data --pretrained checkpoints/simclr_epoch100.pth
  
  # Train SwAV with custom parameters
  python run.py --model swav --mode train --data /path/to/data --batch_size 32 --lr 3e-4 --temp 0.1
        """
    )
    
    # Main arguments
    parser.add_argument('--model', type=str, required=True,
                       choices=['dinov2', 'simclr', 'swav'],
                       help='Model to use: dinov2, simclr, or swav')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'finetune'],
                       help='Mode: train (SSL) or finetune (segmentation)')
    
    # Common training arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training/fine-tuning data directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (model-specific defaults if not set)')
    parser.add_argument('--out', type=str, default=None,
                       help='Output directory for checkpoints (model-specific defaults if not set)')
    
    # Fine-tuning specific
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained checkpoint (required for finetune mode)')
    
    # Model-specific arguments
    parser.add_argument('--temp', type=float, default=None,
                       help='Temperature parameter (for SimCLR/SwAV)')
    parser.add_argument('--momentum', type=float, default=None,
                       help='Momentum for teacher update (for DINOv2, default: 0.996)')
    parser.add_argument('--n_proto', type=int, default=None,
                       help='Number of prototypes (for SwAV, default: 512)')
    parser.add_argument('--queue_size', type=int, default=None,
                       help='Queue size (for SwAV, default: 3840)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'finetune' and args.pretrained is None:
        parser.error('--pretrained is required for finetune mode')
    
    # Set default outputs
    if args.out is None:
        if args.mode == 'train':
            args.out = f'checkpoints_{args.model}'
        else:
            args.out = f'checkpoints_seg_{args.model}'
    
    # Set model-specific defaults
    if args.lr is None:
        if args.model == 'dinov2':
            args.lr = 5e-4
        elif args.model in ['simclr', 'swav']:
            args.lr = 3e-4
    
    if args.temp is None:
        if args.model == 'simclr':
            args.temp = 0.5
        elif args.model == 'swav':
            args.temp = 0.1
    
    if args.momentum is None and args.model == 'dinov2':
        args.momentum = 0.996
    
    if args.n_proto is None and args.model == 'swav':
        args.n_proto = 512
    
    if args.queue_size is None and args.model == 'swav':
        args.queue_size = 3840
    
    # Route to appropriate function
    print("=" * 60)
    print(f"Model: {args.model.upper()}")
    print(f"Mode: {args.mode}")
    print(f"Data path: {args.data}")
    print(f"Output directory: {args.out}")
    print("=" * 60)
    
    try:
        if args.mode == 'train':
            if args.model == 'dinov2':
                train_dinov2(args)
            elif args.model == 'simclr':
                train_simclr(args)
            elif args.model == 'swav':
                train_swav(args)
        else:  # finetune
            if args.model == 'dinov2':
                finetune_dinov2(args)
            elif args.model == 'simclr':
                finetune_simclr(args)
            elif args.model == 'swav':
                finetune_swav(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
