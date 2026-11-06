#!/usr/bin/env bash
# Train PerLA on one benchmark from the pretrained detector.
#
#   bash scripts/train.sh scanqa|scanrefer|nr3d OUTPUT_DIR [extra main.py arguments]
#
# Environment variables:
#   DATA_ROOT    prepared ScanNet scenes                    (default: data/scannet/scannet_data)
#   OPT          OPT-1.3B weights, local path or Hub id     (default: facebook/opt-1.3b)
#   QFORMER      Q-Former tokenizer/embeddings              (default: CH3COOK/bert-base-embedding)
#   NGPUS        GPUs; the global batch stays 16            (default: 2)
#   STOP_EPOCH   end this run early, keeping the full learning-rate schedule
#   START_EPOCH  epoch the run must resume from (0 for a new run)
#
# Re-running with the same OUTPUT_DIR resumes from its checkpoint.pth.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recipes.sh

if (( $# < 2 )); then
    echo "usage: bash scripts/train.sh scanqa|scanrefer|nr3d OUTPUT_DIR [extra main.py arguments]" >&2
    exit 2
fi
task=$1 output=$2
shift 2
perla_recipe "$task"

ngpus=${NGPUS:-2}
if (( 16 % ngpus )); then
    echo "NGPUS must divide the global batch of 16" >&2
    exit 2
fi
stage=()
if [[ -n ${STOP_EPOCH:-} ]]; then
    stage+=(--stop_epoch "$STOP_EPOCH")
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
python scripts/train.py \
    --expected_start_epoch "${START_EPOCH:-0}" "${stage[@]}" \
    "${task_args[@]}" "${model_args[@]}" \
    --root_dir "${DATA_ROOT:-data/scannet/scannet_data}" \
    --vocab "${OPT:-facebook/opt-1.3b}" \
    --qformer_vocab "${QFORMER:-CH3COOK/bert-base-embedding}" \
    --pretrained_weights pretrained/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
    --freeze_detector --freeze_llm \
    --local_loss_weight 0.1 --global_loss_weight 0.01 \
    --base_lr 1e-4 --final_lr 1e-6 --warm_lr 1e-6 --warm_lr_epochs 1 \
    --seed 0 --reseed_each_epoch \
    --ngpus "$ngpus" --batchsize_per_gpu $(( 16 / ngpus )) --dataset_num_workers 4 \
    --dist_url "tcp://127.0.0.1:${PORT:-29500}" --log_every 20 \
    --checkpoint_dir "$output" \
    "$@"
