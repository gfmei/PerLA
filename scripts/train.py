"""Resumable training with full validation after every epoch.

Wraps main.py for ScanQA, ScanRefer and Nr3D (or a multi-task mixture validated on one
of them). Training can run in stages (--stop_epoch) while keeping the full --max_epoch
learning-rate schedule; each stage checks that it resumes where the previous one ended.
Every epoch's predictions and metrics are kept under <checkpoint_dir>/evaluations/,
and summary.json records the best and final epochs.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import transformers

import main as entrypoint
from datasets.dloader import build_dataloader_func
from libs.dist import get_rank, is_primary
from libs.misc import seed_everything


TRAIN_ANNOTATIONS = {
    'unified_scanqa': 'data/ScanQA/ScanQA_v1.0_train.json',
    'unified_densecap_scanrefer': 'data/ScanRefer/ScanRefer_filtered_train.json',
    'unified_densecap_nr3d': 'data/Nr3D/nr3d_train.json',
    'unified_3dllm_scene_description': 'data/3D_LLM/3d_llm_scene_description_train.json',
    'unified_3dllm_embodied_dialogue': 'data/3D_LLM/3d_llm_embodied_dialogue_filtered_train.json',
    'unified_3dllm_embodied_planning': 'data/3D_LLM/3d_llm_embodied_planning_filtered_train.json',
}
# Validation annotations and the prefix of the prediction files each evaluator writes.
VALIDATION = {
    'unified_scanqa': ('data/ScanQA/ScanQA_v1.0_val.json', 'qa_'),
    'unified_densecap_scanrefer': ('data/ScanRefer/ScanRefer_filtered_val.json', 'scanrefer_densecap_'),
    'unified_densecap_nr3d': ('data/Nr3D/nr3d_val.json', 'nr3d_densecap_'),
}


def validation_setup(name, test_min_iou):
    """Return validation annotations, evaluator output files and selection criterion."""
    annotations, prefix = VALIDATION[name]
    outputs = tuple(f'{prefix}{kind}_val.json' for kind in ('corpus', 'pred', 'pred_gt'))
    # Dense-captioning metrics are keyed by IoU threshold, e.g. 'CiDEr@0.5'. A missing
    # key is swallowed by the training loop and no best checkpoint is ever saved.
    criterion = 'CiDEr' if name == 'unified_scanqa' else f'CiDEr@{test_min_iou}'
    return annotations, outputs, criterion


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class LimitedLoader:
    def __init__(self, loader, steps):
        self.loader, self.steps = loader, steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        return itertools.islice(self.loader, self.steps)


class RecordedEvaluation:
    """A picklable evaluator, so spawned DataLoader workers can copy the dataset."""

    def __init__(self, evaluate, outputs):
        self.evaluate, self.outputs = evaluate, outputs

    def __call__(self, args, epoch, model, config, loader, logout=print, curr_train_iter=-1):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        start = time.perf_counter()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                seed_everything(args.seed + 10000 + get_rank())
                metrics = self.evaluate(args, epoch, model, config, loader, logout,
                                        curr_train_iter=curr_train_iter)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        if is_primary():
            root = Path(args.checkpoint_dir)
            destination = root / 'evaluations' / f'epoch-{epoch + 1:03d}'
            destination.mkdir(parents=True, exist_ok=True)
            for name in self.outputs:
                shutil.copyfile(root / name, destination / name)
            write_json(destination / 'metrics.json', dict(
                epoch=epoch, train_iteration=curr_train_iter, metrics=metrics,
                seconds=time.perf_counter() - start,
                validation_questions=len(loader.dataset.annotations),
            ))
        return metrics


def stable_args(args):
    transient = {'start_epoch', 'stop_epoch', 'expected_start_epoch', 'dist_url',
                 'optimizer_steps', 'skipped_steps'}
    return {key: value for key, value in vars(args).items() if key not in transient}


def worker(rank, args):
    torch.set_num_threads(4)
    original_train = entrypoint.do_train

    def recorded_train(args, model, model_no_ddp, optimizer, config, loaders, best):
        if not args.expected_start_epoch <= args.start_epoch <= args.stop_epoch:
            raise ValueError(f'Expected resume in [{args.expected_start_epoch}, {args.stop_epoch}], '
                             f'got {args.start_epoch}')
        root = Path(args.checkpoint_dir)
        validation = loaders['test'][0].dataset
        if args.preflight_steps:
            loaders['train'] = LimitedLoader(loaders['train'], args.preflight_steps)
            validation.annotations = validation.annotations[:16]
            # DistributedSampler caches its size at construction. Rebuild it
            # after subsetting so the preflight does not repeat the subset.
            _, loaders['test'][0] = build_dataloader_func(args, validation, split='test')
        steps_per_epoch = len(loaders['train'])
        args.eval_every_iteration = steps_per_epoch
        args.save_every = args.max_epoch * steps_per_epoch + 1
        validation_annotations, outputs, _ = validation_setup(
            args.eval_dataset or args.dataset, args.test_min_iou)
        validation.eval_func = RecordedEvaluation(validation.eval_func, outputs)
        manifest_path = root / 'run_manifest.json'
        if is_primary():
            if manifest_path.exists():
                saved = json.loads(manifest_path.read_text())
                if saved['args'] != stable_args(args):
                    raise ValueError('Training arguments changed between stages')
            else:
                train_dataset = (loaders['train'].loader if args.preflight_steps
                                 else loaders['train']).dataset
                digest = hashlib.sha256()
                for name, parameter in model_no_ddp.named_parameters():
                    if parameter.requires_grad:
                        digest.update(name.encode())
                        digest.update(parameter.detach().cpu().numpy().tobytes())
                write_json(manifest_path, dict(
                    args=stable_args(args), initial_trainable_sha256=digest.hexdigest(),
                    pretrained_sha256=sha256(args.pretrained_weights),
                    train_annotations_sha256={
                        path: sha256(path) for path in
                        (TRAIN_ANNOTATIONS[name] for name in args.dataset.split(','))
                    },
                    validation_annotations_sha256=sha256(validation_annotations),
                    train_questions=len(train_dataset),
                    train_examples_by_dataset={
                        name: len(dataset) for name, dataset in
                        zip(args.dataset.split(','), train_dataset.datasets)
                    },
                    validation_questions=len(validation.annotations),
                    steps_per_epoch=steps_per_epoch,
                    torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(),
                ))
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        original_train(args, model, model_no_ddp, optimizer, config, loaders, best)
        stop = args.stop_epoch or args.max_epoch
        expected_steps = stop * steps_per_epoch
        if args.optimizer_steps != expected_steps or args.skipped_steps:
            raise RuntimeError(f'Expected {expected_steps} optimizer steps; got '
                               f'{args.optimizer_steps}, skipped {args.skipped_steps}')
        if is_primary():
            evaluations = [json.loads(path.read_text()) for path in
                           sorted((root / 'evaluations').glob('epoch-*/metrics.json'))]
            if [value['epoch'] for value in evaluations] != list(range(stop)):
                raise RuntimeError('Missing full-epoch validation results')
            best_evaluation = max(evaluations, key=lambda value: value['metrics'][args.criterion])
            summary = dict(
                complete=stop == args.max_epoch, completed_epochs=stop,
                requested_epochs=args.max_epoch, optimizer_steps=args.optimizer_steps,
                skipped_steps=args.skipped_steps, steps_per_epoch=steps_per_epoch,
                validation_questions=len(validation.annotations),
                stage_seconds=time.perf_counter() - start,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                best=best_evaluation, final=evaluations[-1],
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            )
            if summary['complete']:
                summary['checkpoint_sha256'] = {
                    name: sha256(root / name) for name in ('checkpoint.pth', 'checkpoint_best.pth')
                }
            write_json(root / f'stage-{args.start_epoch:03d}-{stop:03d}.json', summary)
            write_json(root / 'summary.json', summary)
            print(json.dumps(summary, indent=2), flush=True)

    entrypoint.do_train = recorded_train
    entrypoint.main(rank, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog='All other arguments are passed to main.py.')
    parser.add_argument('--expected_start_epoch', type=int, default=0,
                        help='Fail unless the run resumes at this epoch (0 for a new run)')
    parser.add_argument('--preflight_steps', type=int, default=0,
                        help='Smoke test: train this many steps and validate on 16 samples')
    options, remaining = parser.parse_known_args()
    args = entrypoint.make_args_parser(remaining)
    validation_name = args.eval_dataset or args.dataset
    if validation_name not in VALIDATION or args.test_only or not args.reseed_each_epoch:
        parser.error('Use training mode, one validation set from '
                     f'{sorted(VALIDATION)}, and --reseed_each_epoch')
    criterion = validation_setup(validation_name, args.test_min_iou)[2]
    if args.criterion != criterion:
        parser.error(f'Use --criterion {criterion} for {validation_name}')
    unknown = set(args.dataset.split(',')) - TRAIN_ANNOTATIONS.keys()
    if unknown:
        parser.error(f'No recorded annotations for training datasets {sorted(unknown)}')
    if not args.freeze_detector or not args.freeze_llm or not args.pretrained_weights:
        parser.error('Specify frozen detector/LLM and pretrained detector weights')
    args.expected_start_epoch = options.expected_start_epoch
    args.preflight_steps = options.preflight_steps
    if options.preflight_steps:
        args.max_epoch, args.stop_epoch, args.warm_lr_epochs = 1, 1, 0
    if args.stop_epoch is None:
        args.stop_epoch = args.max_epoch
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn', force=True)
    if args.ngpus == 1:
        worker(0, args)
    else:
        torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.ngpus, join=True)


if __name__ == '__main__':
    main()
