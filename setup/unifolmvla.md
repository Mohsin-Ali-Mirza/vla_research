# Installation Guide: UnifoLM-VLA

This guide outlines the environment configuration, repository installation, and model acquisition for the UnifoLM-VLA project.

---

### 1. Environment & Dependency Setup

Create and activate the environment using the provided YAML configuration to ensure proper linking of CUDA and system libraries.

**`environment.yml`:**

```yaml
name: unifolm-vla
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10.18
  - pip=26.0.1
  - wheel=0.46.3
  - packaging=26.0
  - ncurses=6.5
  - openssl=3.6.2
  - patchelf=0.17.2
  - sqlite=3.51.2
  - libgcc=15.2.0
  - libstdcxx=15.2.0
  - zlib=1.3.1
  - bzip2=1.0.8
  - expat=2.8.1
  - pytorch=2.5.1
  - torchvision=0.20.1
  - pytorch-cuda=12.1
  - pip:
      - ninja==1.13.0
      - git+https://github.com/kvablack/dlimp.git
      - absl-py==2.4.0
      - accelerate==1.5.2
      - robosuite==1.4.0
      - bddl==1.0.1

```

**Commands:**

```bash
conda env create -f environment.yml
conda activate unifolm-vla

```

---

### 2. Flash-Attention & Repository Installation

Install Flash-Attention and clone the necessary project repositories.

```bash
# Install Flash-Attention
MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation "flash-attn==2.5.6"

# Clone Repositories
git clone https://github.com/unitreerobotics/unifolm-vla.git
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Setup LIBERO
cd LIBERO
pip install -r requirements.txt
pip install libero
pip install future tyro imageio qwen_vl_utils

# Setup UnifoLM-VLA
cd ../unifolm-vla
pip install -e .

```

*If you encounter image processing errors, run:*

```bash
conda install -c conda-forge libtiff -y
pip install --no-cache-dir --force-reinstall pillow

```

---

### 3. Model Downloads

We use `hf_transfer` for high-speed downloads.

```bash
pip install "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p /media/ext4-data/share/groot-vla/model_checkpoints
cd /media/ext4-data/share/groot-vla/model_checkpoints

# Download Checkpoints
hf download username/UnifoLM-VLM-Base --local-dir ./UnifoLM-VLM-Base
hf download username/UnifoLM-VLA-Base --local-dir ./UnifoLM-VLA-Base
hf download username/UnifoLM-VLA-LIBERO --local-dir ./UnifoLM-VLA-LIBERO

```
