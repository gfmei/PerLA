# KL attention grounding

PerLA optionally uses the forward KL grounding objective from ReTok3D to
supervise attention to annotated objects. It complements instruction-conditioned
fusion and also works with baseline fusion. It is disabled by default.

Enable it by adding these options to a training command:

```bash
--kl_attention_loss --kl_attention_weight 1.0 --kl_attention_smoothing 0.05
```

The total objective adds `kl_attention_weight * KL(target || attention)` to the
existing language, detector, and weighted local/global losses. The weight is a
starting configuration, not a tuned optimum.

## Targets and attention

ScanQA uses all annotated referent IDs. ScanRefer and Nr3D captioning and
open-vocabulary detection use the described object's ID. Targets follow the
actual sampled points after cropping and augmentation. Unsupported tasks,
missing referents, and examples whose response is entirely truncated contribute
zero KL.

For each represented superpoint, compute the fraction of sampled points belonging
to a referent. Normalize these fractions and mix in the configured uniform mass
over represented superpoints. Share a superpoint's target mass equally among its
encoder keys, so repeated keys do not increase that region's total weight. Void
labels and regions absent from the encoder are excluded.

ReTok3D supervises answer-token attention to spatial tokens in its language
decoder. PerLA instead places spatial keys in the Q-Former: OPT receives learned
query embeddings without a one-to-one region assignment. This adaptation
therefore supervises Q-Former cross-attention, normalizing over valid spatial
keys for each head and latent query before averaging. Visual prompt queries are
excluded. KL is averaged over cross-attention layers and the whole batch,
including examples with zero contribution. Object labels enter only the loss;
they do not change the instruction, visual prompts, or scene features.

Target construction uses existing superpoint assignments without a dense
point-to-key distance matrix. The loss reads Q-Former's attention probabilities
before dropout; it adds no teacher model or learned parameters. Generation needs
neither annotations nor KL calculations, and existing checkpoints can be used
to initialize training. Resume requires matching KL settings; use a new output
directory when changing the objective.

Training reports the unweighted KL loss and the fraction of examples with
represented referents. Compare baseline, baseline + KL, adaptive fusion, and
adaptive fusion + KL under the same [evaluation protocol](ll3da_benchmark.md).
ScanQA and ScanRefer benchmark improvements have not yet been measured.

For a bounded ScanQA comparison, `scripts/run_grounding_pilot.py` accepts the
usual `main.py` training arguments plus `--steps` and `--validation_samples`.
Run each objective in a separate output directory with identical settings.
The runner fixes validation sampling, saves predictions before and after
training, and records actual optimizer steps and GPU memory. Use
`--validation_samples 0` for the full validation split; subset scores are
diagnostics and are not comparable to published full-split results.

Reference implementation: ReTok3D,
`models/point_mllm/attention_grounding.py`. This PerLA implementation adapts its
occupancy targets and forward-KL normalization; it does not depend on ReTok3D at
runtime.
