# simclr/inference.py
"""
Inference script for SimCLR and segmentation models.
Supports feature extraction and segmentation inference.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os

from model import SimCLR
from dataset import NumpySimCLRDataset
from finetune_seg import NumpySegDataset
import segmentation_models_pytorch as smp


def extract_features(model, dataloader, device, output_dir):
    """Extract features using SimCLR encoder."""
    model.eval()
    features = []
    paths = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (view1, view2) in enumerate(tqdm(dataloader, desc="Extracting features")):
            view1 = view1.to(device)
            h, z = model(view1)  # Use encoder features (h) or projected features (z)
            
            # Store features (use h for encoder features, z for projected)
            features.append(h.cpu().numpy())
            paths.extend([f"sample_{idx}_{i}" for i in range(h.shape[0])])
    
    features = np.concatenate(features, axis=0)
    
    # Save features
    output_path = Path(output_dir) / "features.npy"
    np.save(output_path, features)
    print(f"Saved features shape {features.shape} to {output_path}")
    
    return features


def segment_images(model, dataloader, device, output_dir):
    """Run segmentation inference."""
    model.eval()
    predictions = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (x, _) in enumerate(tqdm(dataloader, desc="Running segmentation")):
            x = x.to(device)
            pred = model(x)
            
            # Apply softmax and get class predictions
            pred_probs = F.softmax(pred, dim=1)
            pred_classes = torch.argmax(pred_probs, dim=1)
            
            predictions.append({
                'probabilities': pred_probs.cpu().numpy(),
                'classes': pred_classes.cpu().numpy()
            })
            
            # Save individual predictions
            for i in range(pred_classes.shape[0]):
                np.save(Path(output_dir) / f"pred_{idx}_{i}.npy", pred_classes[i].numpy())
    
    print(f"Saved {len(predictions)} batch predictions to {output_dir}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained models")
    parser.add_argument('--mode', type=str, choices=['features', 'segment'], required=True,
                        help='Inference mode: extract features or run segmentation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default='inference_output',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='Tile size for input images')
    parser.add_argument('--dim', type=int, default=128,
                        help='Projection dimension for SimCLR (features mode only)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'features':
        # Load SimCLR model
        print(f"Loading SimCLR model from {args.model_path}")
        model = SimCLR(dim=args.dim).to(device)
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Load dataset
        dataset = NumpySimCLRDataset(root_dir=args.data, tile_size=args.tile_size)
        # Only need one view for inference
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)
        
        # Extract features
        extract_features(model, dataloader, device, args.output)
        
    elif args.mode == 'segment':
        # Load segmentation model
        print(f"Loading segmentation model from {args.model_path}")
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=6,
            classes=2,
        ).to(device)
        
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Load dataset
        dataset = NumpySegDataset(root_dir=args.data, tile_size=args.tile_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)
        
        # Run segmentation
        segment_images(model, dataloader, device, args.output)


if __name__ == '__main__':
    main()

