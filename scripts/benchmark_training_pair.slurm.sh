#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
set -euo pipefail
repository_root=$(realpath "$1")
speed_root="$repository_root/outputs/experiments/20261101-speed"
harness_root="$speed_root/harness"
export PERLA_BENCHMARK_SCRIPT="$harness_root/benchmark_training.py"
cd "$repository_root"
python_bin=/leonardo_scratch/fast/AIFPT_agrifood/miniforge3/envs/seer/bin/python
"$python_bin" -m unittest tests.test_point_sampling -v
bash "$harness_root/benchmark_training.slurm.sh" "$repository_root" "$speed_root/reference-source" paired-reference
bash "$harness_root/benchmark_training.slurm.sh" "$repository_root" "$speed_root/candidate-source" paired-candidate
"$python_bin" "$harness_root/compare_training_benchmarks.py" \
    "$speed_root/paired-reference.json" "$speed_root/paired-candidate.json" \
    --output "$speed_root/comparison.json"
