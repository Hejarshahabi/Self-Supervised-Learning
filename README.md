# SSL Models for Sentinel-2 Satellite Imagery

A unified PyTorch implementation of three self-supervised learning (SSL) methods for Sentinel-2 satellite imagery:
- **DINOv2**: Vision Transformer (ViT) based self-distillation with no labels
- **SimCLR**: Simple contrastive learning with ResNet-50 backbone
- **SwAV**: Swapping Assignments between Views with ResNet-50 backbone

All models support fine-tuning for segmentation tasks using U-Net architecture.

## Installation

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install torch torchvision
pip install numpy albumentations
pip install segmentation-models-pytorch
pip install tqdm einops
```

### System Requirements

- Python 3.7+
- CUDA-capable GPU (recommended, but CPU works for smaller datasets)
- 16GB+ RAM recommended

## Quick Start

### Training a Model

Use the unified `run.py` script to train any of the three models:

```bash
# Train DINOv2
python run.py --model dinov2 --mode train --data /path/to/data --epochs 100

# Train SimCLR
python run.py --model simclr --mode train --data /path/to/data --epochs 100 --batch_size 32

# Train SwAV
python run.py --model swav --mode train --data /path/to/data --epochs 100 --batch_size 32
```

### Fine-tuning for Segmentation

After training, fine-tune the model for segmentation:

```bash
# Fine-tune DINOv2
python run.py --model dinov2 --mode finetune --data /path/to/seg_data --pretrained checkpoints_dinov2/dinov2_epoch100.pth

# Fine-tune SimCLR
python run.py --model simclr --mode finetune --data /path/to/seg_data --pretrained checkpoints_simclr/simclr_epoch100.pth

# Fine-tune SwAV
python run.py --model swav --mode finetune --data /path/to/seg_data --pretrained checkpoints_swav/swav_epoch100.pth
```

## Data Format

### Training Data (SSL)

- **Format**: `.npy` files containing arrays of shape `(6, H, W)`
- **Directory**: Place all `.npy` files in a single directory
- **Channels**: 6 channels (Sentinel-2 bands: B2, B3, B4, B8, B11, B12)
- **Example structure**:
  ```
  /path/to/data/
    ├── sample_0.npy
    ├── sample_1.npy
    └── ...
  ```

### Fine-tuning Data (Segmentation)

- **Images**: `img_*.npy` files with shape `(6, H, W)`
- **Masks**: `mask_*.npy` files with shape `(H, W)`, values in `{0, 1}`
- **Directory**: Both image and mask files in the same directory
- **Naming**: Files must be sorted such that `img_0.npy` pairs with `mask_0.npy`
- **Example structure**:
  ```
  /path/to/seg_data/
    ├── img_0.npy
    ├── mask_0.npy
    ├── img_1.npy
    ├── mask_1.npy
    └── ...
  ```

## Usage Guide

### Command-Line Arguments

#### Common Arguments

- `--model`: Model to use (`dinov2`, `simclr`, or `swav`) **[required]**
- `--mode`: Mode of operation (`train` or `finetune`) **[required]**
- `--data`: Path to training/fine-tuning data directory **[required]**
- `--epochs`: Number of training epochs (default: `100`)
- `--batch_size`: Batch size (default: `16`)
- `--lr`: Learning rate (model-specific defaults if not set)
- `--out`: Output directory for checkpoints (model-specific defaults if not set)

#### Fine-tuning Specific

- `--pretrained`: Path to pretrained checkpoint (required for `finetune` mode) **[required for finetune]**

#### Model-Specific Arguments

**DINOv2:**
- `--momentum`: Momentum for teacher update (default: `0.996`)

**SimCLR:**
- `--temp`: Temperature parameter for contrastive loss (default: `0.5`)

**SwAV:**
- `--temp`: Temperature parameter (default: `0.1`)
- `--n_proto`: Number of prototypes (default: `512`)
- `--queue_size`: Queue size for SwAV (default: `3840`)

### Examples

#### Training Examples

```bash
# Train DINOv2 with default parameters
python run.py --model dinov2 --mode train --data ./data/train

# Train SimCLR with custom parameters
python run.py --model simclr --mode train --data ./data/train \
    --epochs 200 --batch_size 32 --lr 1e-4 --temp 0.7 --out ./checkpoints_simclr

# Train SwAV with custom queue size
python run.py --model swav --mode train --data ./data/train \
    --batch_size 32 --n_proto 1024 --queue_size 7680
```

#### Fine-tuning Examples

```bash
# Fine-tune DINOv2 for segmentation
python run.py --model dinov2 --mode finetune \
    --data ./data/seg --pretrained ./checkpoints_dinov2/dinov2_epoch100.pth \
    --epochs 50 --batch_size 16 --lr 1e-4

# Fine-tune SimCLR with custom output directory
python run.py --model simclr --mode finetune \
    --data ./data/seg --pretrained ./checkpoints_simclr/simclr_epoch100.pth \
    --out ./segmentation_checkpoints
```

## Model-Specific Details

### DINOv2

- **Architecture**: Vision Transformer (ViT) with student-teacher framework
- **Input Size**: 128×128 (global crops) + 64×64 (local crops)
- **Patch Size**: 8×8 (16 patches per side)
- **Default Learning Rate**: 5e-4
- **Optimizer**: AdamW with weight decay 1e-4
- **Features**:
  - Self-distillation with momentum teacher
  - Multi-crop strategy (2 global + 6 local crops)
  - Center update mechanism

### SimCLR

- **Architecture**: ResNet-50 encoder + projection head
- **Input Size**: 128×128
- **Default Learning Rate**: 3e-4
- **Default Temperature**: 0.5
- **Optimizer**: Adam
- **Features**:
  - Simple contrastive learning
  - NT-Xent loss
  - Data augmentation pipeline

### SwAV

- **Architecture**: ResNet-50 encoder + projection head + prototypes
- **Input Size**: 128×128 (global crops) + 64×64 (local crops)
- **Default Learning Rate**: 3e-4
- **Default Temperature**: 0.1
- **Default Prototypes**: 512
- **Default Queue Size**: 3840
- **Optimizer**: Adam
- **Features**:
  - Online clustering with prototypes
  - Multi-crop strategy (2 global + 4 local crops)
  - Queue mechanism (starts after 10 epochs)

## Output Files

### Training Outputs

Models are saved in the output directory (default: `checkpoints_{model}`):

- **DINOv2**: `dinov2_epoch{epoch}.pth` (saved every 10 epochs and at final epoch)
- **SimCLR**: `simclr_epoch{epoch}.pth` (saved every 10 epochs and at final epoch)
- **SwAV**: `swav_epoch{epoch}.pth` (saved every 10 epochs and at final epoch)

### Fine-tuning Outputs

Segmentation models are saved in the output directory (default: `checkpoints_seg_{model}`):

- `seg_epoch{epoch}.pth` (saved every 10 epochs and at final epoch)

## Project Structure

```
SSL/
├── DINOv2/
│   ├── dataset.py          # DINOv2 dataset loader
│   ├── model.py            # DINOv2 model architecture
│   ├── loss.py             # DINO loss function
│   ├── train_dinov2.py     # Original training script
│   └── finetune_seg.py     # Original fine-tuning script
├── SimCLR/
│   ├── dataset.py          # SimCLR dataset loader
│   ├── model.py            # SimCLR model architecture
│   ├── loss.py             # NT-Xent loss function
│   ├── train_simclr.py     # Original training script
│   └── finetune_seg.py     # Original fine-tuning script
├── SwAV/
│   ├── dataset.py          # SwAV dataset loader
│   ├── model.py            # SwAV model architecture
│   ├── loss.py             # SwAV loss function
│   ├── train_swav.py       # Original training script
│   └── finetune_seg.py     # Original fine-tuning script
├── run.py                  # Unified training/fine-tuning script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Tips and Best Practices

1. **Batch Size**: Adjust based on GPU memory. DINOv2 typically uses smaller batches (16) due to multi-crop strategy.

2. **Learning Rate**: Start with defaults and adjust based on convergence:
   - If loss decreases too slowly: increase learning rate
   - If loss is unstable: decrease learning rate

3. **Temperature (SimCLR/SwAV)**:
   - Lower temperature (0.1-0.3): Sharper distributions, better for fine-grained features
   - Higher temperature (0.5-1.0): Softer distributions, more general features

4. **Data Loading**: The scripts use 8 workers by default. Adjust `num_workers` in the code if needed.

5. **Checkpoints**: Models are saved every 10 epochs. You can resume training by modifying the scripts to load checkpoints.

6. **GPU Memory**: If encountering OOM errors:
   - Reduce batch size
   - Reduce number of crops (DINOv2/SwAV)
   - Use gradient accumulation (requires code modification)

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### CUDA Out of Memory

Reduce batch size or input image size:

```bash
python run.py --model simclr --mode train --data ./data --batch_size 16
```

### Slow Training

- Ensure you're using GPU (check with `nvidia-smi`)
- Increase `num_workers` in DataLoader if CPU has many cores
- Use mixed precision training (requires code modification)
