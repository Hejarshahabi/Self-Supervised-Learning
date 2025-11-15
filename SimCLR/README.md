# SimCLR for Sentinel-2 Satellite Imagery

A PyTorch implementation of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) for self-supervised learning on Sentinel-2 satellite imagery, with fine-tuning capabilities for segmentation tasks.

## Overview

This repository implements:
- **SimCLR**: Self-supervised contrastive learning for 6-channel Sentinel-2 imagery
- **Segmentation Fine-tuning**: Transfer learning from SimCLR features to segmentation tasks
- **Inference**: Feature extraction and segmentation inference scripts

## Project Structure

```
SimCLR/
├── dataset.py          # Dataset loader for SimCLR training
├── model.py            # SimCLR model architecture (ResNet-50 + projection head)
├── loss.py             # NT-Xent contrastive loss function
├── train_simclr.py     # Training script for SimCLR
├── finetune_seg.py     # Fine-tuning script for segmentation
├── inference.py        # Inference script for features and segmentation
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install torch torchvision
pip install albumentations
pip install segmentation-models-pytorch
pip install numpy tqdm
```

## Data Format

### SimCLR Training Data
- **Format**: `.npy` files containing arrays of shape `(6, H, W)`
- **Directory**: Place all `.npy` files in a single directory
- **Channels**: 6 channels (e.g., Sentinel-2 bands: B2, B3, B4, B8, B11, B12)

### Segmentation Fine-tuning Data
- **Images**: `img_*.npy` files with shape `(6, H, W)`
- **Masks**: `mask_*.npy` files with shape `(H, W)`, values in `{0, 1}`
- **Directory**: Both image and mask files in the same directory
- **Naming**: Files must be sorted such that `img_0.npy` pairs with `mask_0.npy`, etc.

## Usage

### 1. Train SimCLR Model

Train a self-supervised SimCLR model on your dataset:

```bash
python train_simclr.py \
    --data /path/to/training/data \
    --epochs 100 \
    --batch_size 32 \
    --lr 3e-4 \
    --temp 0.5 \
    --out checkpoints
```

**Arguments:**
- `--data`: Path to directory containing `.npy` files
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 3e-4)
- `--temp`: Temperature parameter for contrastive loss (default: 0.5)
- `--out`: Output directory for checkpoints (default: checkpoints)

**Output:** Model checkpoints saved as `simclr_epoch{epoch}.pth` in the output directory.

### 2. Fine-tune for Segmentation

Fine-tune a U-Net segmentation model using SimCLR-pretrained encoder:

```bash
python finetune_seg.py \
    --data /path/to/segmentation/data \
    --pretrained checkpoints/simclr_epoch100.pth \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --out seg_checkpoints
```

**Arguments:**
- `--data`: Path to directory with `img_*.npy` and `mask_*.npy` files
- `--pretrained`: Path to SimCLR checkpoint
- `--epochs`: Number of fine-tuning epochs (default: 50)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--out`: Output directory for checkpoints (default: seg_checkpoints)

**Output:** Segmentation model checkpoints saved as `seg_epoch{epoch}.pth`.

### 3. Run Inference

#### Extract Features

Extract encoder features from images using a trained SimCLR model:

```bash
python inference.py \
    --mode features \
    --model_path checkpoints/simclr_epoch100.pth \
    --data /path/to/test/data \
    --output feature_output \
    --batch_size 16 \
    --dim 128
```

**Output:** Features saved as `features.npy` in the output directory.

#### Run Segmentation

Perform segmentation inference on test images:

```bash
python inference.py \
    --mode segment \
    --model_path seg_checkpoints/seg_epoch50.pth \
    --data /path/to/test/data \
    --output seg_output \
    --batch_size 16
```

**Output:** Segmentation predictions saved as `.npy` files in the output directory.

## Model Architecture

### SimCLR
- **Encoder**: ResNet-50 backbone modified for 6-channel input
- **Projection Head**: 2-layer MLP (2048 → 2048 → 128)
- **Loss**: Normalized Temperature-scaled Cross Entropy (NT-Xent)

### Segmentation Model
- **Architecture**: U-Net with ResNet-50 encoder
- **Input**: 6-channel images
- **Output**: 2-class segmentation maps
- **Loss**: Dice Loss

## Data Augmentation

SimCLR training uses the following augmentations:
- Random crop (256×256)
- Horizontal/Vertical flips
- Random 90° rotation
- Random brightness/contrast adjustment
- Gaussian noise
- Coarse dropout

## Implementation Details

- **Optimizer**: Adam
- **Learning Rate Schedule**: Cosine annealing
- **Device**: Automatically uses CUDA if available, otherwise CPU
- **Mixed Precision**: Not implemented (can be added if needed)

## Tips

1. **Large Datasets**: Adjust `num_workers` in DataLoader for faster data loading
2. **Memory**: Reduce `batch_size` if encountering OOM errors
3. **Temperature**: Lower temperature (0.1-0.5) typically works well for contrastive learning
4. **Checkpoints**: Models are saved every 10 epochs and at the final epoch


