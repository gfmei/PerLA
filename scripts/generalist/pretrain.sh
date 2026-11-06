#!/usr/bin/env bash
# Generalist pretraining on the LL3DA task mixture (paper recipe, baseline fusion).
#
#   bash scripts/generalist/pretrain.sh OUTPUT_DIR [extra main.py arguments]
#
# DATA_ROOT must cover every scene used by ScanQA, ScanRefer, Nr3D and 3D-LLM.
# DATA_ROOT, OPT, QFORMER and NGPUS work as in scripts/train.sh (NGPUS defaults to 4).
set -euo pipefail
cd "$(dirname "$0")/../.."
if (( $# < 1 )); then
    echo "usage: bash scripts/generalist/pretrain.sh OUTPUT_DIR [extra main.py arguments]" >&2
    exit 2
fi
output=$1
shift

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
python main.py \
    --dataset unified_3dllm_scene_description,unified_3dllm_embodied_dialogue,unified_3dllm_embodied_planning,unified_scanqa,unified_densecap_nr3d,unified_densecap_scanrefer \
    --eval_dataset unified_scanqa --criterion CiDEr \
    --detector perla --captioner perla --use_color --use_normal \
    --n_splits 6 --n_neighs 24 --n_clus 128 --fusion_method baseline \
    --root_dir "${DATA_ROOT:-data/scannet/scannet_data}" \
    --vocab "${OPT:-facebook/opt-1.3b}" \
    --qformer_vocab "${QFORMER:-CH3COOK/bert-base-embedding}" \
    --pretrained_weights pretrained/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
    --freeze_detector --freeze_llm \
    --max_epoch 32 --base_lr 1e-4 --final_lr 1e-6 --warm_lr_epochs 1 \
    --eval_every_iteration 10000 --start_eval_after 19999 --save_every 10000 \
    --max_des_len 512 --max_prompts 1 --use_beam_search \
    --ngpus "${NGPUS:-4}" --batchsize_per_gpu 4 \
    --dist_url "tcp://127.0.0.1:${PORT:-29502}" \
    --checkpoint_dir "$output" \
    "$@"
