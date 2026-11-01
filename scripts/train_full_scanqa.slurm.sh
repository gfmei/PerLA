#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
set -euo pipefail

run_root=$(realpath "$1")
start_epoch=$2
stop_epoch=$3
mode=${4:-train}
cd "$run_root/source"
module load openjdk/11.0.20.1_1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

variant=baseline
extra=()
if [[ ${SLURM_ARRAY_TASK_ID:-0} == 1 ]]; then
    variant=kl
    extra+=(--kl_attention_loss --kl_attention_weight 1.0 --kl_attention_smoothing 0.05)
fi
destination="$run_root/$variant"
if [[ "$mode" == preflight || "$mode" == preflight-speed ]]; then
    destination="$run_root/$mode/$variant"
    extra+=(--preflight_steps 2)
elif [[ "$mode" != train ]]; then
    echo "Unknown training mode: $mode" >&2
    exit 2
fi
repository_root=$(cat "$run_root/repository_root.txt")
python_bin=/leonardo_scratch/fast/AIFPT_agrifood/miniforge3/envs/seer/bin/python
"$python_bin" scripts/run_full_scanqa.py \
    --expected_start_epoch "$start_epoch" --stop_epoch "$stop_epoch" \
    --max_epoch 24 --reseed_each_epoch --seed 0 \
    --use_color --use_normal --detector perla --captioner perla \
    --freeze_detector --freeze_llm --dataset unified_scanqa \
    --root_dir "$repository_root/outputs/data/scannet_scanqa" \
    --vocab "$repository_root/outputs/models/opt-1.3b" \
    --qformer_vocab "$repository_root/outputs/models/bert-base-embedding" \
    --pretrained_weights "$repository_root/pretrained/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth" \
    --checkpoint_dir "$destination" \
    --ngpus 4 --batchsize_per_gpu 4 --dataset_num_workers 4 \
    --dist_url tcp://127.0.0.1:29610 \
    --n_splits 6 --n_neighs 24 --n_clus 128 \
    --fusion_method baseline --neighbor_selection hilbert \
    --warm_lr_epochs 1 --warm_lr 1e-6 --base_lr 1e-4 --final_lr 1e-6 \
    --local_loss_weight 1 --global_loss_weight 1 \
    --max_des_len 224 --max_prompts 1 --use_beam_search --log_every 20 \
    "${extra[@]}"
