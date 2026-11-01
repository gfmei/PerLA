# ScanQA KL-grounding pilot: 2026-11-01

Both 200-step training runs completed successfully. On the same 128 validation
questions, KL grounding increased CIDEr ×100 from **10.75 to 23.54**. This is a
short training diagnostic with seed 0 and a freshly initialized PerLA captioner.
It does not establish full-validation accuracy, an improvement over published
PerLA/LL3DA scores, or the effect of KL after generalist pretraining.

## Results

| Measurement | Baseline | Baseline + KL |
| --- | ---: | ---: |
| Initial CIDEr ×100 | 0.3919 | 0.3919 |
| Final CIDEr ×100 | 10.7519 | 23.5389 |
| Final ROUGE-L ×100 | 6.2743 | 15.3016 |
| Final METEOR ×100 | 2.9152 | 7.4978 |
| Actual optimizer steps | 200 | 200 |
| Training + final validation, seconds | 1056.33 | 1059.57 |
| Peak allocated GPU memory, GiB | 20.1434 | 20.1493 |

The CIDEr ×100 difference is **+12.7870**. A paired bootstrap of fixed
per-question scores (10,000 resamples, RNG seed 0) gives a 95% interval of
**[+0.2825, +25.1457]**. This interval does not account for training-seed
variability or correlations between questions from the same scene. These
subset scores must not be compared with the published full-validation targets.

Initial predictions match exactly. The comparison also verifies matching code,
runner snapshots, initialization hash, annotations, validation question IDs,
package versions, GPU type, and every training argument except the output
directory and the KL enable flag. Both runs used baseline fusion with Hilbert
neighbors; the instruction-conditioned fusion variant was not trained here.

The machine-readable [comparison](results/20261101_scanqa_kl_pilot.json) records
all shared arguments, validation IDs, source hashes, raw metrics, and timings.

## Configuration and artifacts

- Code: `7fdd8d2eb2535f2c81abacf6b805f23b227a516d`, with an empty working-tree patch.
- One NVIDIA A100-SXM-64GB per run; PyTorch `2.2.0+cu121`, Transformers `4.40.2`.
- Frozen pretrained detector and OPT-1.3B; BERT embedding initialization for the
  Q-Former; no trained generalist checkpoint.
- Batch 4, six partitions, 24 neighbors, 128 clusters, constant learning rate
  `1e-4`, local/global loss weights both 1, beam-search decoding.
- KL weight 1.0, smoothing 0.05. Training logs confirm grounding supervision was
  active, with 97.5–100% represented referents in the logged windows.
- Validation sampling uses the runner's fixed subset seed 1729; point sampling
  and prompts are reset before each evaluation.
- Prepared scenes: 633, covering the ScanQA train/validation annotations.

The completed Slurm tasks `63169381_0` and `63169381_1` both exited with code 0.
Their full local artifacts remain under
`outputs/experiments/20261101-kl/{baseline-tensor,kl-tensor}/`, including
`run_manifest.json`, initial/final predictions, logs, `summary.json`, and
`checkpoint.pth`. These large artifacts are ignored by Git; the results above
are retained in the repository. The checkpoints are pilot weights, not released
benchmark checkpoints.

| Checkpoint | SHA-256 |
| --- | --- |
| `baseline-tensor/checkpoint.pth` | `3711c9f2ee536c4b3de5c971a696aff2eb8ffe0d34be0af449122f88ba123b2c` |
| `kl-tensor/checkpoint.pth` | `4b7cd7309b0d0727a4f364834f88ed27c235128776b2a024ffc589a124b95f24` |

The asset manifest pins OPT-1.3B to Hugging Face revision
`3f5c25d0bc631cb57ac65913f76e22c2dfb61d62` and `CH3COOK/bert-base-embedding` to
`dad6c5c4bbc67801c70d59f8dfd6446925f07cc1`.

Regenerate the comparison without a GPU:

```bash
python scripts/compare_grounding_pilots.py \
  outputs/experiments/20261101-kl/baseline-tensor \
  outputs/experiments/20261101-kl/kl-tensor \
  --output docs/results/20261101_scanqa_kl_pilot.json
```

To repeat training inside a GPU allocation, run the command below once for the
baseline and again with a fresh output directory and
`--kl_attention_loss --kl_attention_weight 1.0 --kl_attention_smoothing 0.05`.
Java must be available for METEOR. The local assets use the paths from these runs.

```bash
python scripts/run_grounding_pilot.py \
  --steps 200 --validation_samples 128 \
  --use_color --use_normal --detector perla --captioner perla \
  --freeze_detector --freeze_llm --dataset unified_scanqa \
  --root_dir outputs/data/scannet_scanqa \
  --vocab outputs/models/opt-1.3b \
  --qformer_vocab outputs/models/bert-base-embedding \
  --pretrained_weights pretrained/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
  --checkpoint_dir outputs/experiments/scanqa-kl-repeat/baseline \
  --ngpus 1 --batchsize_per_gpu 4 --dataset_num_workers 4 \
  --n_splits 6 --n_neighs 24 --n_clus 128 --seed 0 \
  --fusion_method baseline --neighbor_selection hilbert \
  --base_lr 1e-4 --final_lr 1e-4 --local_loss_weight 1 --global_loss_weight 1 \
  --max_des_len 224 --max_prompts 1 --use_beam_search --log_every 10
```

## GPU fusion benchmark

The tensor Hilbert encoder preserves exact keys and merged features in the
recorded GPU checks. With batch 4, 1,024 global keys and 6,144 local keys on the
same GPU type, median inference time over 20 measurements was:

| Fusion configuration | Scalar Hilbert, ms | Tensor Hilbert, ms | Speedup |
| --- | ---: | ---: | ---: |
| Baseline / Hilbert neighbors | 349.06 | 13.65 | 25.57× |
| Instruction / geometry neighbors | 347.64 | 12.89 | 26.97× |

These timings cover the fusion module only, excluding the detector, Q-Former
and language model. See the [raw benchmark](results/20261101_hilbert_gpu.json).

Full-validation training from a generalist checkpoint, multiple training seeds,
ScanRefer evaluation, and the instruction-fusion/KL ablation matrix remain
research follow-ups under the [benchmark protocol](ll3da_benchmark.md).
