# PerLA: Perceptive 3D Language Assistant

Official code for **PerLA**, accepted at **CVPR 2025**. PerLA connects 3D point
clouds with a language model for question answering, dense captioning, and
scene understanding.

[Project website](https://gfmei.github.io/PerLA/) ·
[Paper](https://arxiv.org/abs/2411.19774) ·
[Training guide](docs/full_scanqa_training.md) ·
[Measured speed improvements](docs/training_speed.md)

## Measured improvements

### Training speed and GPU memory

The point-cloud sampling optimization was measured against the preceding
implementation on the **same four A100-SXM-64GB GPUs**, with **16 CPUs per task**
and an unchanged **batch size of 4 per GPU / 16 total**.

| Measurement | Reference | Optimized |
| --- | ---: | ---: |
| Mean training step | 4.644 s | **2.366 s** |
| Training examples per second | 3.445 | **6.762** |
| Peak allocated GPU memory per GPU, maximum across ranks | 20.128 GiB | **8.400 GiB** |

This gives **1.96× throughput**, **49.05% less GPU time per training step**, and
**58.3% lower peak allocated memory**. The benchmark includes data loading,
transfer, forward/backward, AdamW, and distributed synchronization. It uses
three warmup steps and 12 measured steps per variant. Initialization, validation,
and checkpoint writes are excluded.

Initial weights and input batches matched, and every measured training loss
was identical. Full-run cost savings remain to be measured.
[Methodology, source hashes, and raw results →](docs/training_speed.md)

### KL attention grounding

An optional grounding loss supervises Q-Former attention using annotated object
regions. In a **200-step, single-seed pilot on 128 ScanQA validation questions**,
CIDEr ×100 increased from **10.75 to 23.54**. These subset results do not establish
full-validation accuracy or an improvement over the published PerLA benchmark.
[Grounding method](docs/attention_grounding.md) ·
[Pilot results and uncertainty](docs/grounding_pilot_results.md)

**Full-training status, 2026-11-01:** two 24-epoch ScanQA schedules—baseline and
baseline + KL—are queued with the optimized code for the full-validation accuracy
comparison. GPU experiments are limited to these two runs; existing speed and
pilot measurements are reused. Four-GPU training, validation, and checkpoint
checks passed for both. Full-validation results remain pending.
These runs start from the pretrained detector and language components
with a fresh captioner; they do not include generalist pretraining.
[Schedule and artifacts →](docs/full_scanqa_training.md)

## Environment

The measured runs used Python **3.10.20**, PyTorch **2.2.0+cu121**, and
Transformers **4.40.2**. Run commands from the repository root.

An environment matching the core package versions is:

```bash
conda create -n perla python=3.10
conda activate perla
python -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  transformers==4.40.2 numpy==1.23.1 scipy==1.13.1 \
  addict==2.4.0 h5py==3.11.0 easydict==1.9
```

Training requires CUDA. METEOR evaluation also requires Java; the recorded runs
used OpenJDK 11. Mesh preprocessing additionally needs `plyfile`. Visualization
utilities have separate dependencies such as `trimesh`, `open3d`, and
`matplotlib`; they are not required for the training commands below.

## Data and pretrained weights

Prepare these assets before starting a run:

| Asset | Location / argument |
| --- | --- |
| Prepared ScanNet scenes with superpoints | `--root_dir` |
| ScanQA training and validation annotations | `data/ScanQA/ScanQA_v1.0_{train,val}.json` |
| OPT-1.3B model and tokenizer | `--vocab` |
| LL3DA BERT embedding checkpoint and tokenizer | `--qformer_vocab` |
| Pretrained XYZ/color/normal detector | `pretrained/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth` |

Each prepared scene must include `_aligned_vert.npy`, `_aligned_bbox.npy`,
`_ins_label.npy`, `_sem_label.npy`, and `_spt.npy` files. The commands below
enable color and normals, so vertex features must include XYZ, RGB, and normals.

<details>
<summary>Scene preparation and annotation layout</summary>

Follow the [ScanNet preparation instructions](https://github.com/ch3cook-fdu/Vote2Cap-DETR/tree/master/data/scannet).
Set `SCANNET_DIR` and `OUTPUT_FOLDER` in
[datasets/scannet/batch_load_scannet_data.py](datasets/scannet/batch_load_scannet_data.py),
then run:

```bash
cd datasets/scannet
python batch_load_scannet_data.py
cd ../..
```

Annotation sources:

- [ScanQA](https://github.com/ATR-DBI/ScanQA/blob/main/docs/dataset.md)
- [ScanRefer](https://github.com/daveredrum/ScanRefer)
- [Nr3D / ReferIt3D](https://referit3d.github.io/)
- [Included 3D-LLM annotations](data/3D_LLM) and
  [preprocessing script](data/3D_LLM/pre-process-3D-LLM.py)

The ScanQA-only commands need ScanQA annotations. Generalist training also uses
ScanRefer, Nr3D, and 3D-LLM:

```text
data/
├── ScanQA/
│   ├── ScanQA_v1.0_train.json
│   └── ScanQA_v1.0_val.json
├── ScanRefer/
│   ├── ScanRefer_filtered_train.json
│   └── ScanRefer_filtered_val.json
├── Nr3D/
│   ├── nr3d_train.json
│   └── nr3d_val.json
└── 3D_LLM/
    ├── 3d_llm_embodied_dialogue_filtered_{train,val}.json
    ├── 3d_llm_embodied_planning_filtered_{train,val}.json
    └── 3d_llm_scene_description_{train,val}.json
```

</details>

The examples use local model directories under `outputs/models/`. Obtain the
weights and tokenizers for `facebook/opt-1.3b` and
`CH3COOK/bert-base-embedding`, and adjust `--vocab` and `--qformer_vocab` to
their locations. A detector-only checkpoint initializes training; evaluation
requires a trained PerLA captioner checkpoint.

## Quickstart: a bounded ScanQA pilot

Inside a one-GPU allocation, after preparing the assets:

```bash
python scripts/run_grounding_pilot.py \
  --steps 200 --validation_samples 128 --seed 0 \
  --dataset unified_scanqa --detector perla --captioner perla \
  --use_color --use_normal --freeze_detector --freeze_llm \
  --root_dir outputs/data/scannet_scanqa \
  --vocab outputs/models/opt-1.3b \
  --qformer_vocab outputs/models/bert-base-embedding \
  --pretrained_weights pretrained/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
  --checkpoint_dir outputs/experiments/scanqa-pilot/baseline \
  --ngpus 1 --batchsize_per_gpu 4 --dataset_num_workers 4 \
  --n_splits 6 --n_neighs 24 --n_clus 128 \
  --fusion_method baseline --neighbor_selection hilbert \
  --base_lr 1e-4 --final_lr 1e-4 \
  --local_loss_weight 1 --global_loss_weight 1 \
  --max_des_len 224 --max_prompts 1 --use_beam_search
```

For the matched KL variant, use a fresh output directory and append:

```text
--kl_attention_loss --kl_attention_weight 1.0 --kl_attention_smoothing 0.05
```

The runner records initialization, data hashes, actual optimizer steps,
predictions, metrics, runtime, and peak memory. Use a new directory for every
pilot; the general trainer automatically resumes an existing
`checkpoint.pth`.

## Full training

[The full ScanQA guide](docs/full_scanqa_training.md) describes the two submitted
24-epoch runs: all **25,563 training questions**, all **4,675 validation questions**
after every epoch, four GPUs, global batch 16, and **16 CPUs per task**.

`scripts/run_full_scanqa.py` supports staged jobs using `--stop_epoch` while
preserving the full `--max_epoch` learning-rate schedule. It checks resume
configuration and optimizer-step counts, writes checkpoints atomically, and
retains per-epoch predictions. The Slurm launchers contain paths, a Python
environment, and modules for the recorded cluster; adapt them and your
account/partition before using them elsewhere.

For generalist pretraining followed by task-specific tuning, use the existing
launchers. They accept additional command-line arguments to override defaults:

| Task | Launcher |
| --- | --- |
| Generalist pretraining | `scripts/opt-1.3b/train.generalist.sh` |
| ScanQA tuning | `scripts/opt-1.3b/tuning.scanqa.sh` |
| ScanRefer tuning | `scripts/opt-1.3b/tuning.scanrefer.sh` |
| Nr3D tuning | `scripts/opt-1.3b/tuning.nr3d.sh` |

For example, with four allocated GPUs and the required generalist datasets:

```bash
bash scripts/opt-1.3b/train.generalist.sh \
  --root_dir /path/to/prepared/scannet \
  --vocab outputs/models/opt-1.3b \
  --qformer_vocab outputs/models/bert-base-embedding \
  --ngpus 4 --batchsize_per_gpu 4 --n_splits 6 --n_neighs 24 \
  --checkpoint_dir outputs/experiments/generalist
```

Point `--root_dir` at scenes covering **all selected datasets**, and initialize
task tuning with the trained generalist's `checkpoint_best.pth`.
The [accuracy protocol](docs/ll3da_benchmark.md) provides complete tuning commands,
split controls, and guidance for comparisons with published results.

## Evaluation

Evaluate a trained ScanQA checkpoint with the same architecture and fusion
settings used during training:

```bash
bash scripts/opt-1.3b/eval.scanqa.sh \
  --root_dir outputs/data/scannet_scanqa \
  --vocab outputs/models/opt-1.3b \
  --qformer_vocab outputs/models/bert-base-embedding \
  --test_ckpt outputs/experiments/scanqa-pilot/baseline/checkpoint.pth \
  --checkpoint_dir outputs/experiments/scanqa-pilot/baseline/eval \
  --ngpus 1 --batchsize_per_gpu 4 --seed 0 \
  --n_splits 6 --n_neighs 24 --n_clus 128 \
  --fusion_method baseline --neighbor_selection hilbert --max_des_len 224
```

The evaluation launchers also cover
[generalist](scripts/opt-1.3b/eval.generalist.sh),
[ScanRefer](scripts/opt-1.3b/eval.scanrefer.sh), and
[Nr3D](scripts/opt-1.3b/eval.nr3d.sh).
Evaluators return raw CIDEr; multiply by 100 when comparing with tables that use
CIDEr ×100. The command above evaluates the full ScanQA validation split, so its
score has a different scope from the pilot's 128-question subset.

Pilot checkpoints are local experiment artifacts. Full-run benchmark
checkpoints and accuracy results have not yet been released.

## Tests and documentation

The model code has been cleaned up to remove unused legacy fusion/clustering
variants, unreferenced module copies, and obsolete debug demonstrations. Active
model computations and checkpoint parameter layouts are preserved.

```bash
python -m unittest discover -s tests -v
```

The current suite contains 73 tests. On the CPU, 72 passed and the CUDA-only
sampling test was skipped; all three sampling tests passed in a GPU allocation.
Separate four-GPU checks verified baseline/KL training, validation, and
checkpoint writes with 16 CPUs per task.

| Guide | Contents |
| --- | --- |
| [Training speed](docs/training_speed.md) | Matched timings, GPU memory, cost scope, raw records |
| [Grounding pilot](docs/grounding_pilot_results.md) | Accuracy diagnostic, uncertainty, reproducibility |
| [Full ScanQA training](docs/full_scanqa_training.md) | Schedule, Slurm jobs, continuation, outputs |
| [KL attention grounding](docs/attention_grounding.md) | Supervision, masks, loss settings |
| [Adaptive fusion](docs/adaptive_fusion.md) | Experimental instruction conditioning and geometric neighbors |
| [Accuracy protocol](docs/ll3da_benchmark.md) | Generalist/tuning recipes and controlled comparisons |

Instruction-conditioned fusion remains experimental; its accuracy improvement
has not yet been measured.

## Citation

```bibtex
@inproceedings{mei2025PerLA,
  title     = {PerLA: Perceptive 3D Language Assistant},
  author    = {Guofeng Mei and Wei Lin and Luigi Riz and Yujiao Wu and Fabio Poiesi and Yiming Wang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

## Acknowledgments and contributions

PerLA builds on [LL3DA](https://github.com/Open3DA/LL3DA) and
[Vote2Cap-DETR](https://github.com/ch3cook-fdu/Vote2Cap-DETR).
We thank their authors and the dataset contributors.

Bug reports and pull requests are welcome. Include the command, environment,
and relevant logs when reporting a problem.
[Open an issue](https://github.com/gfmei/PerLA/issues) or contact
[gmei@fbk.eu](mailto:gmei@fbk.eu).

## License

The code is licensed under [Apache 2.0](LICENSE).
The project website is licensed under
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
