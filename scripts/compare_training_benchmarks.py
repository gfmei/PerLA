"""Validate a matched training-speed comparison and report GPU-time savings."""

import argparse
import json
import math
from pathlib import Path
import statistics


def compare(reference, candidate):
    for key in ('gpu', 'torch', 'slurm_job_id', 'cpus_per_task', 'warmup_steps', 'measured_steps', 'scope'):
        if reference[key] != candidate[key]:
            raise ValueError(f'Mismatched {key}; compare within one allocation')
    for key in reference['args'].keys() | candidate['args'].keys():
        if key != 'checkpoint_dir' and reference['args'].get(key) != candidate['args'].get(key):
            raise ValueError(f'Mismatched training argument: {key}')
    if len(reference['ranks']) != len(candidate['ranks']) or len(reference['ranks']) != reference['args']['ngpus']:
        raise ValueError('Missing distributed rank results')
    loss_deltas = []
    for before, after in zip(reference['ranks'], candidate['ranks']):
        for key in ('rank', 'batch_sha256', 'initial_trainable_sha256'):
            if before[key] != after[key]:
                raise ValueError(f'Mismatched rank {before["rank"]} {key}')
        if len(before['records']) != reference['measured_steps'] or len(after['records']) != reference['measured_steps']:
            raise ValueError('Missing optimizer steps')
        for first, second in zip(before['records'], after['records']):
            if first['index'] != second['index'] or first['learning_rate'] != second['learning_rate']:
                raise ValueError('Mismatched optimizer schedule')
            if not math.isclose(first['loss'], second['loss'], rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError(f'Loss mismatch: {first["loss"]} versus {second["loss"]}')
            loss_deltas.append(abs(first['loss']-second['loss']))
    before, after = reference['mean_step_seconds'], candidate['mean_step_seconds']
    if not 0 < before < math.inf or not 0 < after < math.inf:
        raise ValueError('Invalid timing')
    projections = {}
    for name, result in (('reference', reference), ('optimized', candidate)):
        projections[name] = dict(
            mean_step_seconds=result['mean_step_seconds'],
            median_step_seconds=result['median_step_seconds'],
            examples_per_second=result['examples_per_second'],
            peak_allocated_gib=result['peak_allocated_gib'],
            mean_phase_seconds={key: statistics.mean(record['phase_seconds'][key]
                for rank in result['ranks'] for record in rank['records'])
                for key in result['ranks'][0]['records'][0]['phase_seconds']},
            projected_training_gpu_hours_38352_steps=result['mean_step_seconds']*38352*result['args']['ngpus']/3600,
        )
    return dict(
        scope=reference['scope'], gpu=reference['gpu'], ngpus=reference['args']['ngpus'],
        batch_per_gpu=reference['args']['batchsize_per_gpu'],
        global_batch=reference['args']['batchsize_per_gpu']*reference['args']['ngpus'],
        slurm_job_id=reference['slurm_job_id'], warmup_steps=reference['warmup_steps'],
        cpus_per_task=reference['cpus_per_task'],
        measured_steps=reference['measured_steps'], **projections,
        speedup=before/after, training_gpu_time_reduction_percent=100*(1-after/before),
        max_absolute_loss_difference=max(loss_deltas),
        matching_batches_and_initialization=True,
        changed_source_files=[key for key in reference['source_sha256'].keys() | candidate['source_sha256'].keys()
                              if reference['source_sha256'].get(key) != candidate['source_sha256'].get(key)],
        limitations='Short single-allocation throughput measurement. GPU-hour projection excludes '
                    'validation, checkpoint I/O, initialization, and queueing. Monetary savings '
                    'assume an unchanged GPU-hour price; no full-run cost or accuracy claim.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
