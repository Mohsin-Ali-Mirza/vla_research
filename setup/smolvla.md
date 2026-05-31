# Installation Guide: LeRobot & SmolVLA

This guide covers the setup of the LeRobot framework, validation of the SmolVLA model, and installation of necessary robotics benchmarks.

---

### 1. Environment Initialization

We utilize `uv` for fast, reproducible environment management with Python 3.12.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Initialize environment
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

# Core package installation
uv pip install -e ".[smolvla]"
uv pip install -e .
uv pip install lerobot

```

---

### 2. Validation: SmolVLA Load Test

Create a file named `check_smolvla.py` in your `lerobot` folder to verify the installation:

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

model_id = "lerobot/smolvla_base"

try:
    print(f"Loading {model_id}...")
    policy = SmolVLAPolicy.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    policy.to(device).eval()

    print(f"SUCCESS! SmolVLA loaded on {device}") 
    print(f"Model class: {policy.__class__.__name__}")
except Exception as e: 
    print(f"X Error: {e}")

```

Run the validation:

```bash
python check_smolvla.py

```

---

### 3. Debugging Environment

Use the built-in LeRobot utility to verify your hardware configuration, CUDA setup, and library versions.

```bash
lerobot-info

```

---

### 4. Robotics Benchmarks (LIBERO & Robosuite)

Install the required robotics suites for evaluation:

```bash
uv pip install robosuite
uv pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git 
uv pip install -e ".[libero, smolvla]"

```

**Troubleshooting:** If the `egl-probe` build fails due to CMake configuration, use the following export before retrying the installation:

```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
uv pip install -e ".[libero, smolvla]"

```

For Official Documentation visit https://huggingface.co/docs/lerobot/en/smolvla
