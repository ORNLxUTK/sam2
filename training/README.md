# SAM 2 Training Pipeline

Training infrastructure for fine-tuning [SAM 2](https://github.com/facebookresearch/sam2) (Segment Anything Model 2) on domain-specific video segmentation datasets. Supports both **LoRA** (Low-Rank Adaptation) and **full fine-tuning** strategies, with distributed multi-GPU training via DDP.

Developed by **ORNLxUTK** (Oak Ridge National Laboratory x MARCI Lab @ University of Tennessee, Knoxville) for additive manufacturing video segmentation of TIG, LWAM, PAW, and polymer (visible + infrared) datasets.

## Quick Start

```bash
# From the repository root (sam2/)
uv sync

# Launch LoRA training with a config
uv run --active training/train.py \
    -c sam2/configs/sam2.1_training/MAZAK_LoRA4_tiny.yaml

# Launch full fine-tuning
uv run --active training/train.py \
    -c sam2/configs/sam2.1_training/MAZAK_finetune_tiny.yaml

# Multi-GPU training
python training/train.py \
    -c sam2/configs/sam2.1_training/MAZAK_LoRA4_tiny.yaml \
    --use-cluster 0 \
    --num-gpus 4
```

For SLURM clusters:
```bash
python training/train.py \
    -c sam2/configs/sam2.1_training/MAZAK_LoRA4_tiny.yaml \
    --use-cluster 1 \
    --num-gpus 4 \
    --num-nodes 1 \
    --partition $PARTITION \
    --account $ACCOUNT
```

## Directory Structure

```
training/
├── train.py                # Entry point (Hydra launcher, local or SLURM/submitit)
├── trainer.py              # Core Trainer class (training loop, checkpointing, LoRA)
├── optimizer.py            # Optimizer construction with per-param-group scheduling
├── loss_fns.py             # Multi-step mask + IoU + dice loss functions
├── ablations.py            # Config generator & SLURM job submitter for ablation experiments
├── model/
│   └── sam2.py             # SAM2Train (extends SAM2Base with training-specific logic)
├── dataset/
│   ├── sam2_datasets.py    # Dataset registration and construction
│   ├── vos_dataset.py      # VOS (Video Object Segmentation) dataset class
│   ├── vos_raw_dataset.py  # Raw dataset loaders (PNG, SA1B, JSON formats)
│   ├── vos_sampler.py      # Frame sampling (RandomUniformSampler, EvalSampler)
│   ├── vos_segment_loader.py  # Segment/mask loading
│   ├── transforms.py       # Video-consistent data augmentation
│   └── utils.py            # Dataset utilities
├── utils/
│   ├── train_utils.py      # Meters, checkpoint discovery, distributed setup
│   ├── checkpoint_utils.py # Checkpoint save/load, parameter filtering
│   ├── data_utils.py       # BatchedVideoDatapoint and collation
│   ├── distributed.py      # Distributed training utilities
│   └── logger.py           # Logging and TensorBoard integration
├── scripts/
│   └── sav_frame_extraction_submitit.py  # SA-V video frame extraction
└── assets/
    └── MOSE_sample_val_list.txt
```

## Configuration

Training uses [Hydra](https://hydra.cc/) for configuration. Config files live in `sam2/configs/sam2.1_training/`.

### Config Naming Convention

| Pattern | Example | Meaning |
|---------|---------|---------|
| `{DATASET}_LoRA{rank}_{size}.yaml` | `MAZAK_LoRA4_tiny.yaml` | LoRA rank 4, tiny model |
| `{DATASET}_finetune_{size}.yaml` | `MAZAK_finetune_large.yaml` | Full fine-tune, large model |

### Key Config Sections

```yaml
scratch:
  resolution: 1024          # Input resolution (square crop)
  train_batch_size: 1       # Per-GPU batch size
  num_frames: 8             # Frames sampled per video clip
  max_num_objects: 3         # Max tracked objects per clip
  base_lr: 0.001            # Learning rate for encoder params
  vision_lr: 0.001          # Learning rate for LoRA / vision params
  num_epochs: 100

dataset:
  img_folder: /path/to/JPEGImages/train   # VOC-style image directory
  gt_folder: /path/to/Annotations/train   # VOC-style mask directory

trainer:
  LoRA:
    use_lora: true           # false for full fine-tuning
    r: 4                     # LoRA rank (2, 4, 8, 16, 32)
    use_rslora: true         # Rank-Stabilized LoRA scaling
    adapter_name: "SAM2_LoRA_tiny"
```

### Per-Dataset Resolution

| Dataset | Resolution |
|---------|-----------|
| LWAM | 1024 |
| TIG | 768 |
| PLASMA | 768 |
| visPOLYMER | 512 |
| irPOLYMER | 256 |

## Fine-Tuning Strategies

### LoRA (Recommended for domain adaptation)

Applies low-rank adapters to **all linear layers** across the entire SAM 2 model using the [PEFT](https://github.com/huggingface/peft) library.

**How it works:**
- PEFT's `get_peft_model()` wraps the model after pretrained weights are loaded
- `target_modules="all-linear"` adds LoRA to every `nn.Linear` in the image encoder, memory attention, memory encoder, and mask decoder
- `lora_alpha = r` with `use_rslora=True` for rank-stabilized scaling
- Only LoRA parameters are optimized; base weights are frozen
- Weight decay: 0.0 for LoRA params, 0.1 for others
- Layer decay (0.9) on the image encoder backbone, overridden to 1.0 for LoRA params

**Supported ranks:** 2, 4, 8, 16, 32

### Full Fine-Tuning

All model weights are updated. Set `trainer.LoRA.use_lora: false`. The optimizer will automatically include all trainable parameters.

## Training Pipeline

```
1. Initialization
   ├── Instantiate SAM 2 model (SAM2Train)
   ├── Load SAM 2.1 pretrained checkpoint
   └── (If LoRA) Wrap with PEFT get_peft_model()

2. Training Loop (per epoch)
   ├── Forward: multi-frame video clips with iterative point sampling
   ├── Loss: focal + dice + IoU + classification (multi-step, multi-mask)
   ├── Backward: AMP (bfloat16), gradient clipping (max_norm=0.1)
   └── Optimizer step with cosine LR decay

3. Validation (configurable frequency)
   ├── Compute validation loss (no gradients)
   ├── Track best checkpoint by validation loss
   └── Early stopping after 20 epochs without improvement

4. Checkpointing
   ├── Regular: checkpoint.pt + checkpoint_lora/ (if LoRA)
   └── Best: best_checkpoint_epoch_XXXX.pt + _lora/
```

## Checkpointing

### LoRA Mode

Checkpoints are split into two parts:

| Component | Contents |
|-----------|----------|
| `.pt` file | Optimizer state, loss state, epoch, steps, best_val_loss, training metadata (**no model weights**) |
| `_lora/` directory | LoRA adapter weights via PEFT `save_pretrained()` (safetensors format) |

### Full Fine-Tune Mode

The `.pt` file contains the full model state dict plus all training metadata.

### Automatic Resume

Training automatically discovers and resumes from the latest checkpoint in `save_dir`. To resume from a specific checkpoint, set:
```yaml
trainer:
  checkpoint:
    resume_from: /path/to/checkpoint.pt
```

## Loss Function

`MultiStepMultiMasksAndIous` (in `loss_fns.py`) combines four loss terms:

| Loss | Weight | Purpose |
|------|--------|---------|
| Focal loss | 20 | Pixel-level mask prediction (handles class imbalance) |
| Dice loss | 1 | Region-level overlap |
| IoU loss | 1 | Predicted vs. actual IoU regression |
| Classification loss | 1 | Object presence score |

The loss is computed across multiple iterative correction steps and selects the best mask channel from multi-mask outputs based on combined focal + dice loss.

## Data Format

### Expected Directory Layout

```
dataset_root/
├── JPEGImages/
│   └── train/
│       └── video_name/
│           ├── 00000.jpg
│           ├── 00001.jpg
│           └── ...
└── Annotations/
    └── train/
        └── video_name/
            ├── 00000.png   # Segmentation masks
            ├── 00001.png
            └── ...
```

Annotation masks are PNG images where each unique nonzero pixel value represents a different object. Background is 0.

### Augmentation

Video-consistent transforms (applied identically across all frames in a clip):
- Random horizontal flip
- Random affine (25 deg rotation, 20 deg shear)
- Resize to target resolution (square)
- Color jitter, random grayscale (5%)
- ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

## Ablation Experiments

`ablations.py` generates configs and submits SLURM jobs for systematic experiments.

**Experiment grid:**
- LoRA ranks: [2, 4, 8, 16, 32]
- Model sizes: tiny, small, base_plus, large
- Datasets: LWAM, irPOLYMER, visPOLYMER, TIG, PLASMA (5-fold cross-validation each)

```bash
# Generate all YAML configs (ranks x sizes x datasets x folds)
python training/ablations.py --create-configs

# Submit SLURM jobs with automatic queue management (max 28 concurrent which was UTK ISAAC cluster limit at the time of development)
python training/ablations.py --submit-jobs
```

## Model Architecture

SAM 2 has four main components. When LoRA is enabled, adapters are added to all linear layers across all components.

| Component | Role | Key Dimensions |
|-----------|------|----------------|
| **Image Encoder** | Hiera vision transformer backbone + FPN neck | embed_dim varies, neck d_model=256 |
| **Memory Attention** | 4-layer cross-attention (RoPE) conditioning current frame on past frames | d_model=256 |
| **Memory Encoder** | Mask downsampling + ConvNeXt fusion for memory storage | out_dim=64, 7 memory slots |
| **Mask Decoder** | TwoWayTransformer + output MLPs for mask/IoU/score prediction | 8 heads, MLP dim=2048 |

### Model Sizes

| Size | Hiera embed_dim | Stages |
|------|----------------|--------|
| Tiny | 96 | [1, 2, 7, 2] |
| Small | 96 | [1, 2, 11, 2] |
| Base+ | 112 | [2, 5, 21, 3] |
| Large | 144 | [2, 6, 36, 4] |

## Distributed Training

Multi-GPU training via PyTorch DDP with NCCL backend. Configured in the YAML:

```yaml
launcher:
  num_nodes: 1
  gpus_per_node: 4

submitit:
  partition: gpu
  timeout_min: 4320   # 3 days
```

For SLURM clusters, uses [submitit](https://github.com/facebookincubator/submitit) for job submission. Single-node local training is also supported.

## Dependencies

Key dependencies (managed via `uv`, see `pyproject.toml`):

| Package | Purpose |
|---------|---------|
| `peft >= 0.17.1` | LoRA adapters via Hugging Face PEFT |
| `torch` | PyTorch framework |
| `hydra-core` | Configuration management |
| `iopath` | File I/O abstraction |
| `tensorboard` | Training visualization |
| `submitit` | SLURM job submission |

## Monitoring

Training metrics are logged to TensorBoard:
```bash
tensorboard --logdir sam2_logs/<config_name>/tensorboard/
```

Key metrics:
- `Losses/train_*_loss` — per-step training losses
- `Losses/val_*_loss` — validation losses
- `Optim/*` — learning rate and weight decay schedules
- `Step_Stats/*` — batch time, data loading time, memory usage

## End-to-End Workflow

```
Raw Data
  → Datasets/ (annotation conversion to VOC-style masks)
  → DatasetVariants/ (cross-validation splits + preprocessing)
  → training/ (this directory — SAM 2 fine-tuning)
  → SAM2inference/ (inference + evaluation with DAVIS/VOS metrics)
```

See the parent repository's [README.md](../README.md) for the full, general SAM 2 pipeline overview.
