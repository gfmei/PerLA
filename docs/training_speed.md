# Training speed at unchanged batch size

A matched four-A100 ScanQA training benchmark measured **1.96× throughput**
after optimizing point sampling and radius-neighbor selection. Both runs used
**16 CPUs per task, batch 4 per GPU, and global batch 16**. Model architecture,
precision, optimizer, data, and training objective were unchanged.

| Measurement | Reference | Optimized |
| --- | ---: | ---: |
| Mean complete training step | 4.644 s | 2.366 s |
| Median complete training step | 4.633 s | 2.364 s |
| Training examples/second | 3.445 | 6.762 |
| Peak allocated GPU memory, maximum across ranks | 20.128 GiB | 8.400 GiB |
| Detector forward, mean across ranks/steps | 4.305 s | 2.024 s |
| Projected GPU-hours for 38,352 training steps | 197.89 | 100.83 |

Measured GPU time per training step fell **49.05%**, and peak allocated GPU
memory fell about **58.3%**. The GPU-hour projection excludes validation,
checkpoint writes, initialization, and queueing. An equivalent monetary saving
requires an unchanged GPU-hour price. This is a short throughput measurement,
not a completed full-run cost or accuracy result.

## What changed

Only `libs/pc_utils.py` differs between the measured model source snapshots:

- Farthest-point sampling uses dense `torch.where` updates instead of boolean
  gathers that synchronize CUDA on every iteration. Initial random samples,
  tie handling, and sampled indices are preserved.
- Radius-neighbor selection uses sorted `topk` on input indices instead of
  sorting the entire point cloud for every query. It preserves the original
  radius comparison, input-order neighbors, and first-neighbor padding.

These changes remove overhead in the frozen detector. They do not introduce
new kernels, dependencies, approximation, reduced point counts, or new loss
normalization.

## Measurement and correctness

Slurm job `63173367` ran both variants sequentially on the same four
NVIDIA A100-SXM-64GB GPUs with PyTorch 2.2.0+cu121. Each variant ran three warmup
steps followed by 12 measured optimizer steps on real ScanQA training batches,
with KL attention supervision enabled. The schedule matches the beginning of
the 24-epoch full run. Timings include waiting for data, transfer, forward,
backward, clipping, AdamW, and DDP synchronization. Batch hashing occurs outside
the timed region; initialization, validation, and checkpoint I/O are excluded.

Every rank's initial trainable-parameter hash and complete input-batch hash
matched. All 12 measured, distributed-averaged training losses were identical
(maximum absolute difference **0.0**). CUDA tests also verified identical FPS
indices and radius neighborhoods against the original operations. The CPU
regression suite passed 72 tests, with the CUDA-only test skipped there; all
three sampling tests passed in the GPU allocation.

The checked-in [comparison](results/20261101_training_speed.json),
[reference record](results/20261101_training_speed_reference.json), and
[optimized record](results/20261101_training_speed_optimized.json) retain
timings, hashes, configuration, and source checksums. Full experiment artifacts,
including immutable source and benchmark-script snapshots, are under
`outputs/experiments/20261101-speed/`. The final comparison uses the successful
matched job above; earlier diagnostic and interrupted runs are excluded.

The benchmark scripts are `scripts/benchmark_training.py`,
`scripts/benchmark_training.slurm.sh`, and
`scripts/benchmark_training_pair.slurm.sh`. The pair launcher expects
`reference-source/`, `candidate-source/`, and a frozen `harness/` under the
experiment directory. To validate existing records again:

```bash
python scripts/compare_training_benchmarks.py \
    outputs/experiments/20261101-speed/paired-reference.json \
    outputs/experiments/20261101-speed/paired-candidate.json \
    --output outputs/experiments/20261101-speed/comparison.json
```

Both [full-training variants](full_scanqa_training.md) use the same optimized
code. Final validation accuracy remains to be established by those runs.

The subsequent code cleanup removes unused legacy fusion/clustering variants,
unreferenced module copies, and scratch demos. Retained model functions and
classes were checked for identical syntax trees, and the regression suite was
rerun. The timing records retain the original benchmark source hashes; queued
training runs retain their frozen source snapshots.
