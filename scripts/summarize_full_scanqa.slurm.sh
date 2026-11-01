#!/usr/bin/env bash
set -euo pipefail
run_root=$(realpath "$1")
cd "$run_root/source"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/leonardo_scratch/fast/AIFPT_agrifood/miniforge3/envs/seer/bin/python \
    scripts/run_full_scanqa.py --summarize "$run_root"
