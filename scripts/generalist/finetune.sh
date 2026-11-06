#!/usr/bin/env bash
# Fine-tune a generalist checkpoint on one benchmark (paper recipe, baseline fusion).
#
#   bash scripts/generalist/finetune.sh scanqa|scanrefer|nr3d GENERALIST_CHECKPOINT OUTPUT_DIR [extra arguments]
#
# DATA_ROOT, OPT, QFORMER and NGPUS work as in scripts/train.sh (NGPUS defaults to 1).
set -euo pipefail
cd "$(dirname "$0")/../.."
if (( $# < 3 )); then
    echo "usage: bash scripts/generalist/finetune.sh scanqa|scanrefer|nr3d GENERALIST_CHECKPOINT OUTPUT_DIR [extra arguments]" >&2
    exit 2
fi
task=$1 generalist=$2 output=$3
shift 3
case "$task" in
    scanqa) task_args=(--dataset unified_scanqa --criterion CiDEr --max_epoch 24 --max_des_len 224
                       --base_lr 1e-6 --final_lr 1e-6 --batchsize_per_gpu 2) ;;
    scanrefer) task_args=(--dataset unified_densecap_scanrefer --criterion CiDEr@0.5 --max_epoch 16
                          --max_des_len 256 --base_lr 1e-6 --final_lr 1e-6 --batchsize_per_gpu 4) ;;
    nr3d) task_args=(--dataset unified_densecap_nr3d --criterion CiDEr@0.5 --max_epoch 32
                     --max_des_len 256 --base_lr 1e-4 --final_lr 1e-6 --batchsize_per_gpu 8) ;;
    *) echo "Unknown task '$task'; use scanqa, scanrefer or nr3d" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
python main.py "${task_args[@]}" \
    --detector perla --captioner perla --use_color --use_normal \
    --n_splits 6 --n_neighs 24 --n_clus 128 --fusion_method baseline \
    --root_dir "${DATA_ROOT:-data/scannet/scannet_data}" \
    --vocab "${OPT:-facebook/opt-1.3b}" \
    --qformer_vocab "${QFORMER:-CH3COOK/bert-base-embedding}" \
    --pretrained_weights "$generalist" \
    --freeze_detector --freeze_llm --warm_lr_epochs 0 \
    --eval_every_iteration 4000 --save_every 10000 \
    --max_prompts 1 --use_beam_search \
    --ngpus "${NGPUS:-1}" \
    --dist_url "tcp://127.0.0.1:${PORT:-29503}" \
    --checkpoint_dir "$output" \
    "$@"
