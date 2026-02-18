# Installation

## Prerequisites

- NVIDIA GPU
- NVIDIA driver compatible with CUDA 12.x
- Conda (Miniconda or Anaconda)

Verify CUDA:

```bash
nvidia-smi
```

## Environment Setup

```bash
conda create -n leanrl-env python=3.13.1 -y
conda activate leanrl-env 
```

### CUDA toolkit

```bash
conda install -c nvidia cuda-toolkit=12.2
```

### PyTorch (CUDA 12.6)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Core packages

```bash
pip install pydantic==2.12.5 pyyaml==6.0.3 huggingface_hub==0.36.0 transformers==4.57.6 deepspeed==0.18.6 datasets==4.5.0 vllm==0.15.1 mlflow==3.8.1
```

### FlashAttention

```bash
pip install flash-attn --no-build-isolation --config-settings="--jobs=8" --verbose
```