"""Measure real ScanQA optimizer steps, including data, transfer, and DDP.

Run reference and candidate sequentially in the same GPU allocation with
identical main.py arguments. Validation/checkpoint I/O and startup are excluded.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch


def worker(rank, args, options):
    sys.path.insert(0, options.source_root)
    os.chdir(options.source_root)
    import main as entrypoint
    from libs.dist import all_reduce_average, barrier, is_primary
    from libs.misc import seed_everything
    from models.perla.engine import adjust_learning_rate

    torch.set_num_threads(4)

    def benchmark(args, model, model_no_ddp, optimizer, config, loaders, best):
        model.train()
        seed_everything(args.seed + rank)
        if hasattr(loaders['train_sampler'], 'set_epoch'):
            loaders['train_sampler'].set_epoch(0)
        initial = hashlib.sha256()
        for name, parameter in model_no_ddp.named_parameters():
            if parameter.requires_grad:
                initial.update(name.encode())
                initial.update(parameter.detach().cpu().numpy().tobytes())
        events = {name: [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                  for name in ('detector', 'captioner', 'backward', 'optimizer')}
        handles = []
        for name in ('detector', 'captioner'):
            module = getattr(model_no_ddp, name)
            handles.append(module.register_forward_pre_hook(
                lambda module, inputs, name=name: events[name][0].record()))
            handles.append(module.register_forward_hook(
                lambda module, inputs, outputs, name=name: events[name][1].record()))
        iterator = iter(loaders['train'])
        records = []
        batches = hashlib.sha256()
        torch.cuda.synchronize()
        barrier()
        for index in range(options.warmup + options.steps):
            if index == options.warmup:
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            batch = next(iterator)
            loaded = time.perf_counter()
            batch = {key: value.cuda() for key, value in batch.items()}
            lr = adjust_learning_rate(args, optimizer, index / (args.max_epoch * len(loaders['train'])))
            optimizer.zero_grad()
            outputs = model(batch, is_eval=False)
            local = outputs.get('l_loss', outputs['loss'].new_zeros(()))
            global_loss = outputs.get('g_loss', outputs['loss'].new_zeros(()))
            loss = outputs['loss'] + args.local_loss_weight * local + args.global_loss_weight * global_loss
            metrics = [loss.detach(), local.detach(), global_loss.detach()]
            if args.kl_attention_loss:
                metrics.extend([outputs['kl_attention_loss'].detach(),
                                outputs['kl_attention_supervised_fraction'].detach()])
            reduced = all_reduce_average(torch.stack(metrics))
            if not math.isfinite(reduced[0].item()):
                raise RuntimeError('Nonfinite loss in benchmark')
            events['backward'][0].record()
            loss.backward()
            events['backward'][1].record()
            events['optimizer'][0].record()
            if args.clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()
            if getattr(model_no_ddp.captioner, 'reference_point', None):
                model_no_ddp.captioner.update_reference_point()
            events['optimizer'][1].record()
            barrier()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            record = dict(index=index, seconds=seconds, data_wait_seconds=loaded-start,
                          loss=reduced[0].item(), learning_rate=lr,
                          phase_seconds={key: pair[0].elapsed_time(pair[1])/1000
                                         for key, pair in events.items()})
            # Hash outside the timed region; includes augmentation and tokenization.
            for key in sorted(batch):
                batches.update(key.encode())
                batches.update(batch[key].cpu().numpy().tobytes())
            if index >= options.warmup:
                records.append(record)
            if is_primary():
                print('BENCHMARK ' + json.dumps(record), flush=True)
        for handle in handles:
            handle.remove()
        result = dict(rank=rank, records=records, batch_sha256=batches.hexdigest(),
                      initial_trainable_sha256=initial.hexdigest(),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        results = [result]
        if args.ngpus > 1:
            results = [None] * args.ngpus
            torch.distributed.all_gather_object(results, result)
        if is_primary():
            steps = [max(r['records'][i]['seconds'] for r in results) for i in range(options.steps)]
            report = dict(label=options.label, args=vars(args), ranks=results,
                          warmup_steps=options.warmup, measured_steps=options.steps,
                          mean_step_seconds=statistics.mean(steps),
                          median_step_seconds=statistics.median(steps),
                          examples_per_second=args.batchsize_per_gpu*args.ngpus/statistics.mean(steps),
                          peak_allocated_gib=max(r['peak_allocated_gib'] for r in results),
                          gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                          source_sha256={str(path.relative_to(options.source_root)):
                                         hashlib.sha256(path.read_bytes()).hexdigest()
                                         for directory in ('models', 'libs', 'datasets')
                                         for path in sorted((Path(options.source_root)/directory).rglob('*.py'))},
                          slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                          cpus_per_task=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)),
                          scope='Real data, transfer, forward, backward, AdamW, and DDP barrier; '
                                'excludes initialization, batch hashing, validation, and checkpoints')
            destination = Path(options.result)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2) + '\n')
            print('RESULT ' + json.dumps({key: value for key, value in report.items()
                                         if key not in ('args', 'ranks', 'source_sha256')}), flush=True)

    entrypoint.do_train = benchmark
    entrypoint.main(rank, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=lambda p: str(Path(p).resolve()), required=True)
    parser.add_argument('--result', type=lambda p: str(Path(p).resolve()), required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=12)
    options, remaining = parser.parse_known_args()
    if options.steps < 1 or options.warmup < 1:
        parser.error('Use positive warmup and measured steps')
    sys.path.insert(0, options.source_root)
    os.chdir(options.source_root)
    import main as entrypoint
    args = entrypoint.make_args_parser(remaining)
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn', force=True)
    if args.ngpus == 1:
        worker(0, args, options)
    else:
        torch.multiprocessing.spawn(worker, nprocs=args.ngpus, args=(args, options), join=True)


if __name__ == '__main__':
    main()
