"""Run a bounded ScanQA training pilot with reproducible validation sampling.

Accepts the usual main.py options plus --steps and --validation_samples.
Use identical options in separate output directories, toggling only
--kl_attention_loss, to compare the objectives. Subset scores are pilot
diagnostics, not full ScanQA benchmark results.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers

from datasets.dloader import build_dataset
from libs.misc import load_checkpoint, load_model_state, seed_everything
from main import make_args_parser
from models.perla.engine import do_train
from models.perla.model_general import CaptionNet


class LimitedLoader:
    def __init__(self, loader, steps):
        if not 0 < steps <= len(loader):
            raise ValueError(f'Pilot steps must be in [1, {len(loader)}]')
        self.loader, self.steps = loader, steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        return itertools.islice(self.loader, self.steps)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--validation_samples', type=int, default=256,
                        help='0 uses the full validation split')
    pilot, remaining = parser.parse_known_args()
    args = make_args_parser(remaining)
    if args.ngpus != 1 or args.test_only or args.dataset != 'unified_scanqa':
        parser.error('This pilot requires one GPU, training mode, and --dataset unified_scanqa')
    if not args.freeze_detector or not args.freeze_llm or not args.pretrained_weights:
        parser.error('Specify --freeze_detector --freeze_llm --pretrained_weights')
    if args.optimizer != 'AdamW':
        parser.error('This pilot uses --optimizer AdamW')
    if not args.checkpoint_dir or pilot.steps < 1 or pilot.validation_samples < 0:
        parser.error('Specify --checkpoint_dir, positive steps, and nonnegative validation_samples')
    destination = Path(args.checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'run_manifest.json').exists() or (destination / 'checkpoint.pth').exists():
        raise ValueError('Use a fresh output directory for each pilot')
    if not torch.cuda.is_available():
        raise RuntimeError('Run this experiment inside a GPU allocation')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    seed_everything(args.seed)
    args.max_epoch, args.start_epoch, args.warm_lr_epochs = 1, 0, 0
    args.eval_every_iteration = args.save_every = pilot.steps + 1

    dataset_config, datasets, loaders = build_dataset(args)
    validation = datasets['test'][0]
    original_annotations = list(validation.annotations)
    indices = list(range(len(original_annotations)))
    if pilot.validation_samples:
        if pilot.validation_samples > len(indices):
            raise ValueError('Requested more validation questions than are available')
        indices = sorted(random.Random(1729).sample(indices, pilot.validation_samples))
    validation.annotations = [original_annotations[index] for index in indices]
    loaders['train'] = LimitedLoader(loaders['train'], pilot.steps)

    manifest = dict(
        args=vars(args), pilot=vars(pilot), command=sys.argv,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint_sha256=sha256(args.pretrained_weights),
        torch=torch.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(),
        validation_question_ids=[a['question_id'] for a in validation.annotations],
        train_annotations_sha256=hashlib.sha256(json.dumps(
            datasets['train'].datasets[0].annotations, sort_keys=True).encode()).hexdigest(),
        validation_annotations_sha256=hashlib.sha256(json.dumps(
            validation.annotations, sort_keys=True).encode()).hexdigest(),
        score_scope='full validation' if not pilot.validation_samples else 'validation subset pilot',
    )
    (destination / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    (destination / 'working_tree.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD']))
    (destination / 'pilot_runner.py').write_bytes(Path(__file__).read_bytes())
    model = CaptionNet(args, dataset_config, nlatent_query=args.nlatent_query)
    checkpoint = load_checkpoint(args.pretrained_weights)
    load_model_state(model, checkpoint.get('model', checkpoint), require_detector=True)
    del checkpoint
    model.cuda()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.base_lr, weight_decay=args.weight_decay)
    completed_steps = []
    step_hook = optimizer.register_step_post_hook(lambda *_: completed_steps.append(1))

    evaluate = validation.eval_func
    evaluations = []

    def recorded_evaluation(*eval_args, **kwargs):
        # Fix the point sampling and token prompts independently of training RNG.
        seed_everything(args.seed + 10000)
        start = time.perf_counter()
        metrics = evaluate(*eval_args, **kwargs)
        evaluations.append(dict(metrics=metrics, seconds=time.perf_counter()-start,
                                train_iteration=kwargs.get('curr_train_iter', -1)))
        (destination / 'evaluation_metrics.json').write_text(json.dumps(evaluations, indent=2))
        phase = 'initial' if len(evaluations) == 1 else 'final'
        for name in ('qa_corpus_val.json', 'qa_pred_val.json', 'qa_pred_gt_val.json'):
            path = destination / name
            if path.exists():
                (destination / f'{phase}_{name}').write_bytes(path.read_bytes())
        return metrics

    validation.eval_func = recorded_evaluation
    model.eval()
    recorded_evaluation(args, -1, model, dataset_config, loaders['test'][0])
    seed_everything(args.seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    do_train(args, model, model, optimizer, dataset_config, loaders, {})
    step_hook.remove()
    summary = dict(evaluations=evaluations, train_and_final_eval_seconds=time.perf_counter()-start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   optimizer_steps=len(completed_steps), requested_steps=pilot.steps,
                   score_scope=manifest['score_scope'])
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if len(completed_steps) != pilot.steps:
        raise RuntimeError('Some optimizer steps were skipped; this pilot is not a matched comparison')


if __name__ == '__main__':
    main()
