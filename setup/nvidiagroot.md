# Installation Guide: NVIDIA Isaac GR00T & Cosmos

This document provides setup instructions for NVIDIA's robotics and world foundation model stacks.

## 1. NVIDIA Isaac GR00T
For step-by-step setup instructions, please refer to the official documentation:

* [Nvidia Isaac GR00T (N1.7 Current)](https://github.com/Nvidia/Isaac-GR00T)
* [Nvidia Isaac GR00T (n1.5-release)](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release)
* [Nvidia Isaac GR00T (n1.6-release)](https://github.com/NVIDIA/Isaac-GR00T/releases/tag/n1.6-release)

---

## 2. NVIDIA Cosmos
If you want to explore the Cosmos World Foundation Model, follow the [official installation documentation](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/setup.md#installation).

### Running Inference (Docker Setup)
I followed the setup steps and used a Docker container. Inside the container, I ran the following command to generate world video content:

```bash
# Set the input prompt
PROMPT_="A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."

# Run video2world generation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
python -m examples.video2world \
    --model_size 2B \
    --input_path assets/video2world/input0.jpg \
    --num_conditional_frames 1 \
    --prompt "${PROMPT_}" \
    --resolution 480 \
    --aspect_ratio 16:9 \
    --num_gpus 1 \
    --disable_guardrail \
    --disable_prompt_refiner \
    --offload_text_encoder \
    --downcast_text_encoder \
    --save_path output/video2world_2b.mp4
