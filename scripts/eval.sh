#!/usr/bin/env bash
# Evaluate a PerLA checkpoint on the validation split of one benchmark.
#
#   bash scripts/eval.sh scanqa|scanrefer|nr3d CHECKPOINT [extra main.py arguments]
#
# Predictions and metrics are written next to the checkpoint, or to OUTPUT_DIR if set.
# DATA_ROOT, OPT, QFORMER and NGPUS work as in train.sh (NGPUS defaults to 1 here).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/recipes.sh

if (( $# < 2 )); then
    echo "usage: bash scripts/eval.sh scanqa|scanrefer|nr3d CHECKPOINT [extra main.py arguments]" >&2
    exit 2
fi
task=$1 checkpoint=$2
shift 2
perla_recipe "$task"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
python main.py --test_only --test_ckpt "$checkpoint" \
    --checkpoint_dir "${OUTPUT_DIR:-$(dirname "$checkpoint")}" \
    "${task_args[@]}" "${model_args[@]}" \
    --root_dir "${DATA_ROOT:-data/scannet/scannet_data}" \
    --vocab "${OPT:-facebook/opt-1.3b}" \
    --qformer_vocab "${QFORMER:-CH3COOK/bert-base-embedding}" \
    --freeze_detector --freeze_llm \
    --ngpus "${NGPUS:-1}" --batchsize_per_gpu "${BATCH:-8}" \
    --dist_url "tcp://127.0.0.1:${PORT:-29501}" \
    "$@"
