# Setup Guides

Documentation for environment configuration, dependencies, and model checkpoints.

| Model |
| :--- |
| **[OpenVLA](./openvla.md)** |
| **[NvidiaGroot](./nvidiagroot.md)** |
| **[UnifolmVLA](./unifolmvla.md)** |
| **[SmolVLA](./smolvla.md)** |

---

# Installation Guide: Flash-Attention

This document outlines the installation methods for `flash-attn`. Due to hardware and version-specific requirements, please follow the steps in order.

### 1. Standard Installation

Attempt the standard installation first:

```bash
pip install flash-attn

```

### 2. Version-Specific Prebuilt Wheel

If the standard installation fails, install a specific version corresponding to your vla model

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```

### 3. Build from Source

If prebuilt wheels fail, you can build from source.
*⚠️ **Warning:** This process may take ~20 minutes.*

```bash
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation

```

### Troubleshooting

For more information, supported versions, and advanced configuration, refer to the official documentation:
**[https://flashattn.dev/](https://flashattn.dev/)**
