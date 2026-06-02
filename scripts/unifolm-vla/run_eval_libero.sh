export LIBERO_HOME="/media/ext4-data/share/groot-vla/LIBERO"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} 
export PYTHONPATH=$(pwd):${PYTHONPATH}


# your_ckpt=/path/to/your/Unifolm-VLA-Libero/checkpoints/pytorch_model.pt
# vlm_pretrained_path=/path/to/your/Unifolm-VLM-Base
your_ckpt="/media/ext4-data/share/groot-vla/unifolm-vla/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt"
vlm_pretrained_path="/media/ext4-data/share/groot-vla/unifolm-vla/UnifoLM-VLM-Base"
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $5}')
step_name=$(echo "$your_ckpt" | awk -F'/' '{print $6}')

# --- CHANGED FOR LIBERO OBJECT ---
task_suite_name=libero_object    # Changed from libero_spatial
unnorm_key="libero_object_no_noops"   # Changed from libero_spatial_no_noops
# ---------------------------------

num_trials_per_task=10
window_size=2

video_out_path="results/${task_suite_name}/${folder_name}/${step_name}"

DEVICE=1

CUDA_VISIBLE_DEVICES=${DEVICE} python ./experiments/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.vlm-pretrained-path ${vlm_pretrained_path} \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.unnorm-key "$unnorm_key" \
    --args.window-size "$window_size"