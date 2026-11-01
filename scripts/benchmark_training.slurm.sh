#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
set -euo pipefail
repository_root=$(realpath "$1")
source_root=$(realpath "$2")
label=$3
destination="$repository_root/outputs/experiments/20261101-speed/$label"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
python_bin=/leonardo_scratch/fast/AIFPT_agrifood/miniforge3/envs/seer/bin/python
"$python_bin" "${PERLA_BENCHMARK_SCRIPT:-$repository_root/scripts/benchmark_training.py}" \
    --source-root "$source_root" --result "$destination.json" --label "$label" \
    --warmup 3 --steps 12 --seed 0 --max_epoch 24 \
    --use_color --use_normal --detector perla --captioner perla \
    --freeze_detector --freeze_llm --dataset unified_scanqa \
    --root_dir "$repository_root/outputs/data/scannet_scanqa" \
    --vocab "$repository_root/outputs/models/opt-1.3b" \
    --qformer_vocab "$repository_root/outputs/models/bert-base-embedding" \
    --pretrained_weights "$repository_root/pretrained/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth" \
    --checkpoint_dir "$destination" \
    --ngpus 4 --batchsize_per_gpu 4 --dataset_num_workers 4 \
    --dist_url tcp://127.0.0.1:29620 \
    --n_splits 6 --n_neighs 24 --n_clus 128 \
    --fusion_method baseline --neighbor_selection hilbert \
    --warm_lr_epochs 1 --warm_lr 1e-6 --base_lr 1e-4 --final_lr 1e-6 \
    --local_loss_weight 1 --global_loss_weight 1 \
    --max_des_len 224 --max_prompts 1 --use_beam_search \
    --kl_attention_loss --kl_attention_weight 1.0 --kl_attention_smoothing 0.05
