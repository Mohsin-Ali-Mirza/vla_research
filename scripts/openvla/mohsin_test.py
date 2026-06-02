## mohisn_test.py

"""
OpenVLA Inference with Isaac Lab
=================================
Runs a trained OpenVLA policy inside an Isaac Lab environment.

Requirements:
  - Isaac Lab  (isaaclab)
  - OpenVLA    (openvla / transformers)
  - lerobot    (optional – for dataset utilities)
  - torch, torchvision, PIL

Usage:
  python openvla_isaaclab_inference.py \
      --task Isaac-Lift-Cube-Franka-v0 \
      --openvla_model openvla/openvla-7b \
      --num_episodes 5 \
      --vla_device cuda:0
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


# ── Isaac Lab bootstrap (must happen before any omni / isaaclab imports) ──────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OpenVLA + Isaac Lab inference")
parser.add_argument("--task",           type=str,   default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--openvla_model",  type=str,   default="openvla/openvla-7b",
                    help="HuggingFace model ID or local path")
parser.add_argument("--num_episodes",   type=int,   default=5)
parser.add_argument("--max_steps",      type=int,   default=300)
parser.add_argument("--instruction",    type=str,   default="pick up the cube and place it on the target")
parser.add_argument("--camera_key",     type=str,   default="rgb",
                    help="Observation key that holds the RGB image")
parser.add_argument("--unnorm_key",     type=str,   default=None,
                    help="Action un-normalisation key from OpenVLA. "
                         "Leave None to skip un-normalisation.")
parser.add_argument("--action_scale",   type=float, default=100.0)
parser.add_argument("--render",         action="store_true", default=True)
parser.add_argument("--save_video",     action="store_true", default=False)
parser.add_argument("--video_dir",      type=str,   default="./videos")
parser.add_argument("--save_images",    action="store_true", default=False,
                    help="Save every image sent to OpenVLA as a JPEG for inspection")
parser.add_argument("--images_dir",     type=str,   default="./openvla_images",
                    help="Directory to save input images")
parser.add_argument("--vla_device",     type=str,   default="cuda:0",
                    help="Torch device for OpenVLA model (e.g. cuda:0, cpu)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch the Isaac Sim application
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Isaac Lab / Gym imports (after app launch) ─────────────────────────────────
import gymnasium as gym
import isaaclab_tasks  # noqa: F401 – registers all built-in tasks

from isaaclab.envs import ManagerBasedRLEnv  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ── OpenVLA imports ────────────────────────────────────────────────────────────
from transformers import AutoModelForVision2Seq, AutoProcessor


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_openvla(model_id: str, device: str):
    """Load OpenVLA processor and model."""
    print(f"[OpenVLA] Loading model from '{model_id}' …")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",  # remove if flash-attn not installed
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("[OpenVLA] Model ready.")
    return processor, model


def set_viewport_camera(
    pos: tuple = (0.7, 0.0, 0.6),
    target: tuple = (0.0, 0.0, 0.35),
):
    """
    Reposition Isaac Sim's default perspective viewport camera.
    Uses the Isaac Sim set_camera_view utility which is the correct API.
    """
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(
        eye=np.array(pos),
        target=np.array(target),
        camera_prim_path="/OmniverseKit_Persp",
    )
    print(f"[Camera] Viewport repositioned → pos={pos}, target={target}")


def obs_to_pil(env, obs, camera_key: str, env_idx: int = 0,
               workspace_camera=None) -> Image.Image:
    """
    Capture a 224x224 RGB image from the repositioned viewport camera.
    env.render() returns the viewport frame as a numpy (H,W,3) uint8 array.
    """
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. Make sure render_mode='rgb_array' "
            "and --enable_cameras are set."
        )
    img = Image.fromarray(frame.astype(np.uint8))
    # Resize to 224x224 as OpenVLA expects
    img = img.resize((224, 224), Image.LANCZOS)
    return img


@torch.inference_mode()
def predict_action(
    processor,
    model,
    image: Image.Image,
    instruction: str,
    unnorm_key: Optional[str],
    device: str,
) -> np.ndarray:
    """Run a single OpenVLA forward pass and return the raw action array."""
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    action = model.predict_action(
        **inputs,
        unnorm_key=unnorm_key,
        do_sample=False,
    )
    # action is a numpy array of shape (action_dim,)
    return action


def make_isaac_action(
    raw_action: np.ndarray,
    num_envs: int,
    action_scale: float,
    device: str,
) -> torch.Tensor:
    """
    Convert OpenVLA's 7-dim output to Isaac Lab's action format.

    OpenVLA outputs:
        raw_action[0:6] = EEF delta pose  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
        raw_action[6]   = gripper command  (continuous, roughly +1=open, -1=close)

    Isaac Lab task variants and their expected action dims:
        Isaac-Lift-Cube-Franka-IK-Rel-v0  →  7  (6 EEF delta + 1 gripper)  ← USE THIS
        Isaac-Lift-Cube-Franka-v0          →  8  (7 joint pos + 1 gripper)   ← wrong space

    The IK-Rel task feeds EEF deltas to an onboard IK solver, matching
    OpenVLA's output convention perfectly.

    Gripper convention:
        OpenVLA:   +1 = open,  -1 = close  (continuous)
        Isaac Lab: +1 = open,  -1 = close  (thresholded internally)
        → no remapping needed.
    """
    action = torch.tensor(raw_action, dtype=torch.float32, device=device)

    # Scale only the EEF translation/rotation, not the gripper
    action[:6] = action[:6] * action_scale

    # Shape: (1, 7) → expand to (num_envs, 7)
    return action.unsqueeze(0).expand(num_envs, -1)


# ══════════════════════════════════════════════════════════════════════════════
# Video recorder (optional)
# ══════════════════════════════════════════════════════════════════════════════

class VideoRecorder:
    def __init__(self, save_dir: str, fps: int = 30):
        import cv2  # lazy import
        self.cv2 = cv2
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer = None
        self.path = None

    def start(self, episode: int, width: int, height: int):
        self.path = self.save_dir / f"episode_{episode:03d}.mp4"
        fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = self.cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))

    def record(self, image: Image.Image):
        if self.writer is None:
            return
        frame = self.cv2.cvtColor(np.array(image), self.cv2.COLOR_RGB2BGR)
        self.writer.write(frame)

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            print(f"[Video] Saved → {self.path}")
            self.writer = None


# ══════════════════════════════════════════════════════════════════════════════
# Main inference loop
# ══════════════════════════════════════════════════════════════════════════════

def run_inference():
    device = args.vla_device

    # ── 1. Load OpenVLA ────────────────────────────────────────────────────────
    processor, model = load_openvla(args.openvla_model, device)

    # ── 2. Create Isaac Lab environment ───────────────────────────────────────
    print(f"[Isaac Lab] Creating env: {args.task}")
    # Isaac Lab requires the env cfg to be explicitly loaded and passed.
    # parse_env_cfg reads the registered entry point and returns a populated cfg.
    # IMPORTANT: parse_env_cfg must use the same device AppLauncher chose for
    # the sim (args.device). Using a different device causes tensor device
    # mismatches inside Isaac Lab's observation manager.
    sim_device = args.device  # set by AppLauncher (e.g. cuda:1 via --device cuda:1)
    env_cfg = parse_env_cfg(
        args.task,
        device=sim_device,
        num_envs=1,
    )
    env: ManagerBasedRLEnv = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args.render else None,
    )
    num_envs = env.unwrapped.num_envs
    print(f"[Isaac Lab] Running {num_envs} parallel environment(s).")

    # ── Workspace camera ───────────────────────────────────────────────────────
    # Spawn a fixed front-facing camera that gives OpenVLA a proper close-up
    # view of the workspace — similar to the Bridge V2 training data viewpoint.
    # env.reset() must be called once first to initialise the scene.
    # Reposition the viewport camera to a close-up front-facing workspace view
    # so env.render() gives OpenVLA a useful image instead of the bird's-eye default.
    obs, _ = env.reset()
    # ── Camera presets ────────────────────────────────────────────────────────
    # Try each preset by changing the index below (0-3) and checking saved images.
    # The Franka robot base sits at roughly (0, 0, 0.4) in world space.
    #
    #  0 = front-facing,  slightly above table    (recommended first try)
    #  1 = front-facing,  higher up / more tilt
    #  2 = side angle,    45 degrees to the right
    #  3 = slight overhead, close in
    camera_presets = [
        dict(pos=( 1.2,  0.0,  0.7), target=(0.3, 0.0, 0.4)),  # 0: front
        dict(pos=( 1.0,  0.0,  1.0), target=(0.3, 0.0, 0.4)),  # 1: front-high
        dict(pos=( 0.9,  0.9,  0.8), target=(0.3, 0.0, 0.4)),  # 2: side-right (best so far)
        dict(pos=( 0.7,  0.0,  1.2), target=(0.3, 0.0, 0.4)),  # 3: overhead
        # Refined from preset 2 — pulled back, cube more centred in frame
        dict(pos=( 1.2,  1.2,  0.9), target=(0.2, 0.0, 0.3)),  # 4: side-right, wider
        dict(pos=( 1.4,  1.0,  0.8), target=(0.1, 0.0, 0.3)),  # 5: side-right, far back
        dict(pos=( 1.0,  1.4,  0.7), target=(0.1, 0.0, 0.3)),  # 6: more side, lower
    ]
    # Try these three — small target adjustments to bring cube into center
    camera_presets += [
        dict(pos=(1.2, 1.2, 0.9), target=(0.5,  0.0, 0.4)),  # 7: aim at robot base area
        dict(pos=(1.2, 1.2, 0.9), target=(0.4, -0.1, 0.4)),  # 8: aim slightly left
        dict(pos=(1.0, 1.0, 0.8), target=(0.4,  0.0, 0.4)),  # 9: closer + same aim
    ]
    cam = camera_presets[7]   # ← change this index to try different angles
    set_viewport_camera(pos=cam["pos"], target=cam["target"])
    print(f"[Camera] Using preset: pos={cam['pos']}, target={cam['target']}")
    workspace_camera = None  # not used — obs_to_pil uses env.render() directly

    # ── 3. Optional video recorder ────────────────────────────────────────────
    recorder = VideoRecorder(args.video_dir) if args.save_video else None

    # ── 4. Episode loop ───────────────────────────────────────────────────────
    episode_returns = []

    for ep in range(args.num_episodes):
        if ep > 0:
            obs, _ = env.reset()
        # ep==0: obs already set above when initialising the camera
        episode_return = 0.0
        step = 0
        terminated = False
        truncated = False

        print(f"\n{'─'*60}")
        print(f"Episode {ep + 1}/{args.num_episodes}  |  instruction: \"{args.instruction}\"")
        print(f"{'─'*60}")

        # Start video recording
        if recorder is not None:
            first_img = obs_to_pil(env, obs, args.camera_key)
            w, h = first_img.size
            recorder.start(ep, w, h)

        t0 = time.perf_counter()

        while not (terminated or truncated) and step < args.max_steps:
            # ── 4a. Observation → PIL image ───────────────────────────────────
            image = obs_to_pil(env, obs, args.camera_key)

            if recorder is not None:
                recorder.record(image)

            # ── 4b. (optional) Save image for inspection ──────────────────────
            if args.save_images:
                img_dir = Path(args.images_dir) / f"episode_{ep:03d}"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"step_{step:04d}.jpg"
                image.save(img_path)

            # ── 4c. OpenVLA inference ─────────────────────────────────────────
            raw_action = predict_action(
                processor, model, image, args.instruction,
                args.unnorm_key, device,
            )

            # ── Debug: print actions + save exact image OpenVLA received ─────
            if step % 10 == 0:
                np.set_printoptions(precision=4, suppress=True)
                print(f"  [OpenVLA] step {step:>4d} | "
                      f"dx={raw_action[0]:+.4f} dy={raw_action[1]:+.4f} dz={raw_action[2]:+.4f} | "
                      f"droll={raw_action[3]:+.4f} dpitch={raw_action[4]:+.4f} dyaw={raw_action[5]:+.4f} | "
                      f"gripper={raw_action[6]:+.4f}  "
                      f"({'OPEN' if raw_action[6] > 0 else 'CLOSE'})")
                dbg_dir = Path("./debug_images") / f"ep{ep:02d}"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                image.save(dbg_dir / f"step_{step:04d}.jpg")


            # ── 4c. Format action for Isaac Lab ──────────────────────────────
            action = make_isaac_action(raw_action, num_envs, args.action_scale, device)

            # ── 4d. Step the environment ──────────────────────────────────────
            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward.mean().item()
            step += 1

            if step % 50 == 0:
                fps = step / (time.perf_counter() - t0)
                print(f"  step {step:>4d}  |  reward {reward.mean().item():.4f}"
                      f"  |  return {episode_return:.4f}  |  {fps:.1f} fps")

        if recorder is not None:
            recorder.stop()

        episode_returns.append(episode_return)
        status = "SUCCESS ✓" if (terminated and not truncated) else "truncated"
        print(f"  → Episode {ep + 1} finished [{status}]  "
              f"return = {episode_return:.4f}  steps = {step}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"Results over {args.num_episodes} episodes")
    print(f"  Mean return : {np.mean(episode_returns):.4f}")
    print(f"  Std  return : {np.std(episode_returns):.4f}")
    print(f"  Max  return : {np.max(episode_returns):.4f}")
    print(f"{'═'*60}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    run_inference()
