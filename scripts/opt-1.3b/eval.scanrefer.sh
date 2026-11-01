#!/usr/bin/env bash
set -euo pipefail

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector perla \
    --captioner perla \
    --checkpoint_dir ./ckpts/opt-1.3b/perla-scanrefer-tuned \
    --test_ckpt ./ckpts/opt-1.3b/perla-scanrefer-tuned/checkpoint_best.pth \
    --dataset unified_densecap_scanrefer \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --dist_url tcp://localhost:29500 \
    --criterion 'CiDEr@0.5' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 8 --ngpus 1 \
    --max_des_len 256 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only \
    "$@"
