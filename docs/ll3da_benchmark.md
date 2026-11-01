# LL3DA comparison and accuracy experiments

No benchmark scores have been measured for the corrected implementation yet.
Passing regression tests verifies behavior, not ScanQA or ScanRefer accuracy.
The current research objective is to improve PerLA itself with the
[experimental fusion method](adaptive_fusion.md). Compare it with the corrected
PerLA baseline as well as the published references below.

## Targets and comparison rules

| Validation benchmark | Published LL3DA | Published PerLA | Published PerLA target to exceed |
| --- | ---: | ---: | ---: |
| ScanQA CIDEr | 76.79 | 78.13 | >78.13 |
| ScanRefer CIDEr@IoU 0.5 | 65.19 | 69.41 | >69.41 |

These are the reported results in [PerLA, Tables 1–2](https://arxiv.org/html/2411.19774v2).
The evaluators here return raw CIDEr: multiply by 100 for this table (0.7679 becomes 76.79).
Do not compare ScanRefer IoU 0.25 with the IoU 0.5 target.

Use identical ScanNet splits, annotations, preprocessing, frozen detector, OPT-1.3B
backbone, prompts and decoding for a local LL3DA comparison. The
[official LL3DA repository](https://github.com/Open3DA/LL3DA) distinguishes its
updated annotations from the paper recipe in `scripts-v0`; record which version
is used. Report published numbers and locally reproduced numbers separately.

Tune on validation only; keep test annotations out of training and model
selection. After selecting a configuration, repeat with seeds 0, 1 and 2 and
report each score plus mean and standard deviation. Keep prediction files,
checkpoint hashes, Git commit, exact commands, package versions, annotation
hashes, GPU count, batch size and actual optimizer steps with each run.

## What changed and why retraining is needed

The corrected merger uses one coordinate grid for local and global points,
looks up the correct sorted Hilbert keys, and returns distinct source indices.
Coordinates and features now refer to the same neighbors. Attention honors
`max(n_neighs, 4*n_splits)` instead of silently using `2*n_splits`.
These changes alter model features even though checkpoint tensor shapes remain
compatible. Evaluate old checkpoints explicitly; do not assume their quality
is preserved or describe new results as an exact reproduction of the paper.

Missing pretrained files and incomplete frozen detector weights now stop the
run. Training can still resume from its output directory without the original
initialization file. Use a fresh output directory for every ablation because
`checkpoint.pth` automatically resumes model and optimizer state.

## Required assets

- Prepared ScanNet scenes, including `_aligned_vert.npy`, `_aligned_bbox.npy`,
  `_ins_label.npy`, `_sem_label.npy`, and `_spt.npy` files under `--root_dir`.
- Language annotations laid out as in the README, including `data/ScanQA`.
- OPT-1.3B weights/tokenizer and the LL3DA BERT embedding checkpoint for
  `--qformer_vocab` (the default local directory is `bert-base-embedding`).
- The detector checkpoint at `pretrained/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth`.
- An allocated CUDA GPU job and the `seer` environment. The detector imports
  `addict`; version 2.4.0 was used for the CPU validation.

## Initial experiment

Start from a generalist, then fine-tune separately for each task. The PerLA
paper uses six partitions, 24 neighbors, batch 16 and an initial generalist
learning rate of 1e-4. Its generalist-to-expert training performs better than
training experts from scratch. These motivate the configuration below; the
current epoch schedules are starting points, not an exact optimizer-step match
to the paper. [Implementation details and Table K](https://arxiv.org/html/2411.19774v2).

The examples use two allocated GPUs and batch eight per GPU. Adjust those two
values together to keep the effective batch at 16. There is no gradient
accumulation in this trainer. Run from the repository root, with the prepared
assets above; use `--vocab` and `--qformer_vocab` to override model locations.

```bash
conda activate seer
SCANNET_ROOT=/path/to/prepared/scannet

bash scripts/opt-1.3b/train.generalist.sh \
  --root_dir "$SCANNET_ROOT" --ngpus 2 --batchsize_per_gpu 8 \
  --n_splits 6 --n_neighs 24 --seed 0 \
  --local_loss_weight 1 --global_loss_weight 1 \
  --checkpoint_dir ckpts/accuracy/generalist-s0

bash scripts/opt-1.3b/tuning.scanqa.sh \
  --root_dir "$SCANNET_ROOT" --ngpus 2 --batchsize_per_gpu 8 \
  --n_splits 6 --n_neighs 24 --seed 0 \
  --pretrained_weights ckpts/accuracy/generalist-s0/checkpoint_best.pth \
  --base_lr 1e-6 --final_lr 1e-6 \
  --checkpoint_dir ckpts/accuracy/scanqa-lr1e-6-s0

bash scripts/opt-1.3b/tuning.scanrefer.sh \
  --root_dir "$SCANNET_ROOT" --ngpus 2 --batchsize_per_gpu 8 \
  --n_splits 6 --n_neighs 24 --seed 0 \
  --pretrained_weights ckpts/accuracy/generalist-s0/checkpoint_best.pth \
  --base_lr 1e-6 --final_lr 1e-6 \
  --checkpoint_dir ckpts/accuracy/scanrefer-lr1e-6-s0
```

The 1e-6 ScanQA fine-tuning rate follows the
[official LL3DA tuning script](https://github.com/Open3DA/LL3DA/blob/main/scripts/opt-1.3b/tuning.scanqa.sh).
Our previous ScanQA launcher started at 1e-4. Compare constant 1e-6, 3e-6 and
1e-5 from the same generalist checkpoint, changing only the learning rate and
output directory. This is an experiment grid, not a promised improvement.

Evaluate each task's best validation checkpoint with the same partition count
and neighbor count used in training:

```bash
bash scripts/opt-1.3b/eval.scanqa.sh \
  --root_dir "$SCANNET_ROOT" --ngpus 1 --batchsize_per_gpu 8 \
  --n_splits 6 --n_neighs 24 --seed 0 \
  --test_ckpt ckpts/accuracy/scanqa-lr1e-6-s0/checkpoint_best.pth \
  --checkpoint_dir ckpts/accuracy/scanqa-lr1e-6-s0/eval

bash scripts/opt-1.3b/eval.scanrefer.sh \
  --root_dir "$SCANNET_ROOT" --ngpus 1 --batchsize_per_gpu 8 \
  --n_splits 6 --n_neighs 24 --seed 0 --test_min_iou 0.5 \
  --test_ckpt ckpts/accuracy/scanrefer-lr1e-6-s0/checkpoint_best.pth \
  --checkpoint_dir ckpts/accuracy/scanrefer-lr1e-6-s0/eval
```

## Follow-up ablations

After establishing the corrected baseline, hold its learning rate and all
other settings fixed and compare local/global loss weights `(1, 1)`, `(0.1,
0.01)` and `(0, 0)`. Defaults remain `(1, 1)` for compatibility. The smaller
weights are motivated by the paper's nested loss coefficients, but this
implementation's local loss includes fit and min-cut terms, so it is not an
exact implementation of the paper's objective. Select weights using measured
validation results, then repeat the selected setup across seeds.

If the corrected setup remains behind LL3DA, inspect failed examples and
detector recall before expanding the language model or changing training data.
Keep architecture/data changes separate from this controlled comparison.
