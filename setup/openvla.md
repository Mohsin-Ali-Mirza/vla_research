# Installation Guide: OpenVLA & LIBERO Setup

This guide walks you through setting up the OpenVLA environment, installing the LIBERO benchmark, and preparing the model weights.

---

### 1. Environment & Dependency Setup

First, create the environment using the following configuration. We recommend saving this as `environment.yml` and running `conda env create -f environment.yml`.

```yaml
name: openvla
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - wheel
  - ninja
  - patchelf
  - pytorch=2.2.0
  - torchvision=0.17.0
  - torchaudio=2.2.0
  - pytorch-cuda=12.1
  - pip:
      - git+https://github.com/kvablack/dlimp.git
      - transformers==4.40.1
      - accelerate==0.29.3
      - timm==0.9.10
      - tokenizers==0.19.1
      - robosuite==1.4.1
      - bddl==3.6.0
      - gym==0.23.1
      - draccus==0.8.0
      - einops==0.8.2
      - wandb==0.24.2
      - h5py==3.16.0

```

### 2. Flash-Attention Installation

After activating your environment (`conda activate openvla`), install `flash-attn`.
*⚠️ **Warning:** Building from source can take ~20 minutes.*

```bash
cd openvla
MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation "flash-attn==2.5.5"

```

### 3. LIBERO Benchmark Setup

Clone and install the LIBERO repository:

```bash
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt

```

### 4. Project Installation

Install the OpenVLA package in editable mode:

```bash
cd openvla
pip install -e . 

```

### 5. Model Download

Download the `openvla-7b` checkpoint for LIBERO:

```bash
hf download openvla/openvla-7b-finetuned-libero-object \
    --local-dir /media/ext4-data/share/groot-vla/openvla/model_checkpoints/openvla-7b-finetuned-libero-object

```

### Troubleshooting & Configuration

* **Evaluation Paths:** If you encounter import errors during evaluation, verify the `benchmark` import path in:
`/media/ext4-data/share/groot-vla/openvla/experiments/robot/libero/run_libero_eval.py`
Ensure the path aligns with your local LIBERO installation:
```python
#option 1
from libero.libero import benchmark

#option 2
from libero import benchmark
```

For Official Documentation refer to https://github.com/openvla/openvla
