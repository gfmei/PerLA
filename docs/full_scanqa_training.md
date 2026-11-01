# Full ScanQA training

**Status, 2026-11-01:** both full-training schedules are held before starting to
conserve GPU resources. All 12 GPU stages and the final CPU comparison are on
user hold; none of the full-training stages has consumed GPU time. The schedule
below records the original plan. The completed short pilots and checks remain
available; there are no active PerLA GPU allocations at this status update.

Two complete ScanQA training schedules were submitted on 2026-11-01: baseline
fusion, and the same model with KL attention grounding. Each uses all 25,563
training questions and evaluates all 4,675 validation questions after every
epoch. These runs start from the frozen detector, OPT-1.3B, and BERT embedding
weights, with a fresh PerLA captioner. They do not resume the short pilots and
do not include generalist pretraining.

## Schedule

| Setting | Both runs |
| --- | --- |
| Epochs | 24 |
| Optimizer steps per epoch | 1,598 |
| Planned optimizer steps | 38,352 |
| Seed | 0 |
| GPUs per run | 4 A100 GPUs, one node |
| CPUs per task | 16 |
| Batch size | 4 per GPU, 16 total except the final batch |
| Learning rate | One warmup epoch from 1e-6 to 1e-4, then cosine toward 1e-6 |
| Partitions / neighbors / clusters | 6 / 24 / 128 |
| Local/global loss weights | 1 / 1 |
| Frozen components | Detector and OPT-1.3B |
| Decoding | Beam search |
| KL candidate | Weight 1.0, smoothing 0.05 |

There is one training seed. These task-specific runs are separate from the
generalist-to-expert recipe in the [benchmark protocol](ll3da_benchmark.md).
Model selection uses validation CIDEr; test annotations are not loaded.

The cluster allows at most 24 hours per GPU job. Each run is divided into six
stages of four epochs, preserving `--max_epoch 24` for the learning-rate schedule.
The next stage resumes model weights, optimizer state, and actual optimizer-step
counts from `checkpoint.pth`. Per-epoch random seeds make restart boundaries
independent of initialization and validation RNG consumption. A requeued stage
can resume any completed epoch within its assigned range.

Checkpoints are replaced atomically, so an interrupted write preserves the last
complete checkpoint. The runner rejects skipped optimizer steps or missing
epoch validations before marking a stage successful. Dependent stages run only
after the corresponding previous stage succeeds.

## Submitted jobs

Array task 0 is baseline; task 1 enables KL. Both complete schedules depend on
short checks of the real four-GPU forward/backward, validation, and checkpoint
paths. Submission is not a claim that training has completed.

Both original four-GPU checks completed with exit code 0: two optimizer steps, no skipped
steps, validation over 16 questions, and successful latest/best checkpoint
writes. Peak allocated memory was 20.555 GiB for baseline and 20.559 GiB for KL.
The initial trainable-parameter and pretrained-checkpoint hashes matched.
These are execution checks, not accuracy results; their records are in
`outputs/experiments/20261101-full-scanqa/preflight_results.json`.

The subsequently optimized point-cloud operations measured 1.96× training
throughput at the same global batch 16, four GPUs, and 16 CPUs per task. See the
[matched speed results](training_speed.md) for measurements and cost limits.
Both full schedules use this optimized code; the original source manifest is
retained as `source_manifest.before_speed.json`.

The optimized baseline and KL checks also completed with exit code 0 using
16 CPUs per task: two optimizer steps, no skipped steps, 16 validation questions,
and successful latest/best checkpoint writes. Their peak allocated memory was
8.653 and 8.657 GiB, respectively. Initialization hashes matched. Records are in
`preflight_speed_results.json`. The held full schedules were then released;
epochs 1–4 depend on the corresponding successful optimized check.

| Work | Slurm job/array |
| --- | --- |
| Original four-GPU checks | `63171564` |
| Optimized four-GPU checks, 16 CPUs per task | `63173521` |
| Epochs 1–4 | `63171675` |
| Epochs 5–8 | `63171676` |
| Epochs 9–12 | `63171677` |
| Epochs 13–16 | `63171678` |
| Epochs 17–20 | `63171679` |
| Epochs 21–24 | `63171680` |
| Final comparison, after both runs complete | `63171682` |

```bash
squeue -u "$USER" -o '%.18i %.25j %.10T %.10M %.38E'
```

The exact submission commands and dependencies are saved in
`outputs/experiments/20261101-full-scanqa/jobs.json`.
The GPU jobs were subsequently updated to `--cpus-per-task=16` at the user's
request; the original commands and resource-update history are both retained.
The later resource-conservation hold is recorded in `scheduling_updates`.
`gpu_resource_review.json` in the same directory records allocated GPU time
from Slurm: 1.63 GPU-hours across the day's completed, failed, and cancelled
PerLA pilots, checks, and benchmarks, including 0.26 GPU-hours for failed or
cancelled attempts. Held jobs allocate no GPUs. No automatic release is
scheduled.

## Reproducibility and outputs

The run directory is `outputs/experiments/20261101-full-scanqa/`. Its `source/`
directory contains a fixed copy of the training code and ScanQA train/validation
annotations. `source_manifest.json` records source hashes, the base Git commit,
and the training plan. The shared large scene/model assets stay at their existing
paths; initialization and annotation hashes are recorded in each run manifest.

Each of `baseline/` and `kl/` contains:

- `run_manifest.json`: arguments, initialization hash, data hashes, environment,
  and a checksum of the initial trainable parameters.
- `checkpoint.pth`: latest completed epoch and optimizer state.
- `checkpoint_best.pth`: checkpoint with the highest validation CIDEr.
- `evaluations/epoch-NNN/`: full predictions, references, per-question scores,
  and aggregate metrics for that epoch.
- `stage-START-STOP.json` and `summary.json`: completed epochs, actual optimizer
  steps, memory, runtime, and best/final validation scores. The `complete` field
  remains false until all 24 epochs finish.

The final CPU job checks matched initialization/configuration and writes
`comparison.json` with best and final validation CIDEr differences. It can also
be run manually after both schedules complete:

```bash
python scripts/run_full_scanqa.py \
  --summarize outputs/experiments/20261101-full-scanqa
```

The reusable launchers are `scripts/train_full_scanqa.slurm.sh` and
`scripts/run_full_scanqa.py`. Training source is fixed before execution; the
speed update was applied while all full-training jobs were held. Changes in the
main workspace do not modify running stages. The CPU regression suite and CUDA
sampling parity tests passed, including schedule continuity, epoch RNG
continuity, checkpoint recovery, and optimizer-state restoration.
