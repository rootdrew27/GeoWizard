# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GeoWizard is a diffusion-based deep learning system for monocular 3D geometry estimation (depth and surface normal prediction) from single images. Published at ECCV 2024.

**Two model versions exist:**
- **V1**: Uses CLIP image encoder for conditioning
- **V2**: Uses text embeddings for conditioning, produces more robust normals for diverse image styles

## Development Commands

```bash
# Environment setup
conda create -n geowizard python=3.9
conda activate geowizard
pip install -r requirements.txt

# V1 Inference (from geowizard/ directory)
python run_infer.py \
    --input_dir input/example \
    --output_dir output \
    --ensemble_size 3 \
    --denoise_steps 10 \
    --domain "indoor"

# V2 Inference
python run_infer_v2.py \
    --input_dir input/example \
    --output_dir output \
    --ensemble_size 3 \
    --denoise_steps 10 \
    --domain "indoor"

# BiNI 3D reconstruction (from bini/ directory)
python bilateral_normal_integration_numpy.py --path data/test_1 -k 2 --iter 50 --tol 1e-5

# Training (modify root_path and output_dir in script first)
cd geowizard/training/scripts
sh train_depth_normal.sh    # V1
sh train_depth_normal_v2.sh # V2
```

**Inference Parameters:**
- `--domain`: "indoor" (default, works for most), "outdoor", or "object" (background-free objects)
- `--ensemble_size`: Number of predictions to average (3 for speed, 10 for academic comparison)
- `--denoise_steps`: Diffusion steps (10 for speed, 50 for academic comparison)

## Architecture

### Core Pipeline Flow
```
Input Image -> VAE Encoder -> Latent Space -> UNet (conditioned on CLIP/text) -> VAE Decoder -> Depth + Normal Maps
```

### Key Components

**geowizard/models/**: Inference model implementations
- `geowizard_pipeline.py`: V1 pipeline (CLIP image conditioning)
- `geowizard_v2_pipeline.py`: V2 pipeline (text embeddings)
- `unet_2d_condition.py`: Custom UNet for dual output (depth + normal)
- `attention.py`, `transformer_2d.py`: Attention mechanisms with xformers support

**geowizard/utils/**: Inference utilities
- `depth_ensemble.py`, `normal_ensemble.py`: Multi-prediction averaging
- `depth2normal.py`: Depth to normal conversion
- `image_util.py`: Image preprocessing

**geowizard/training/**: Training pipeline (mirrors inference structure)
- `training/train_depth_normal.py`, `train_depth_normal_v2.py`: Training scripts
- `dataloader/mix_loader.py`: Multi-dataset loader (HyperSim, Replica, Objaverse, etc.)
- `dataloader/transforms.py`: Data augmentation

**bini/**: Bilateral Normal Integration for 3D mesh reconstruction
- `bilateral_normal_integration_numpy.py`: CPU version
- `bilateral_normal_integration_cupy.py`: GPU version

### Design Notes

- Training and inference utilities are duplicated (`geowizard/utils/` and `geowizard/training/utils/`)
- Pipelines inherit from HuggingFace `DiffusionPipeline`
- Distributed training uses HuggingFace `accelerate`
- Memory optimization via gradient checkpointing and xformers
- Input size: 576x768 for training, configurable for inference via `--processing_res`

## Key Dependencies

- `diffusers==0.25.0`: Diffusion pipeline framework
- `transformers==4.39.1`: CLIP models and text encoders
- `accelerate==0.28.0`: Distributed training
- `xformers==0.0.21`: Memory-efficient attention
- `torch==2.0.1`, `torchvision==0.15.2`: Core ML framework
