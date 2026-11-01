"""Compare a matched baseline/KL pair produced by run_grounding_pilot.py."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(directory, filename):
    return json.loads((directory / filename).read_text())


def compare(baseline, grounded, bootstrap_samples=10000, bootstrap_seed=0):
    require(bootstrap_samples > 0, 'bootstrap_samples must be positive')
    runs = [Path(baseline), Path(grounded)]
    manifests = [read_json(run, 'run_manifest.json') for run in runs]
    summaries = [read_json(run, 'summary.json') for run in runs]
    matched_fields = (
        'commit', 'checkpoint_sha256', 'validation_question_ids',
        'train_annotations_sha256', 'validation_annotations_sha256',
        'torch', 'transformers', 'gpu', 'pilot', 'score_scope',
    )
    for key in matched_fields:
        require(manifests[0][key] == manifests[1][key], f'Mismatched {key}')
    args = [manifest['args'] for manifest in manifests]
    require(args[0]['kl_attention_loss'] is False and args[1]['kl_attention_loss'] is True,
            'Expected baseline without KL and candidate with KL')
    require(args[1]['kl_attention_weight'] > 0, 'KL weight must be positive')
    for key in args[0].keys() | args[1].keys():
        if key not in ('checkpoint_dir', 'kl_attention_loss'):
            require(key in args[0] and key in args[1] and args[0][key] == args[1][key],
                    f'Mismatched argument: {key}')
    source_hashes = {}
    for filename in ('working_tree.patch', 'pilot_runner.py'):
        contents = [(run / filename).read_bytes() for run in runs]
        require(contents[0] == contents[1], f'Mismatched {filename}')
        source_hashes[filename] = hashlib.sha256(contents[0]).hexdigest()

    steps = manifests[0]['pilot']['steps']
    question_ids = manifests[0]['validation_question_ids']
    require(steps > 0 and len(question_ids) > 0, 'Empty pilot')
    require(len(set(question_ids)) == len(question_ids), 'Duplicate validation question IDs')
    requested_samples = manifests[0]['pilot']['validation_samples']
    require(requested_samples == 0 or requested_samples == len(question_ids),
            'Validation sample count does not match manifest')
    for summary in summaries:
        require(summary['optimizer_steps'] == summary['requested_steps'] == steps,
                'Incomplete optimizer steps')
        require(summary['score_scope'] == manifests[0]['score_scope'], 'Mismatched score scope')
        evaluations = summary['evaluations']
        require(len(evaluations) == 2 and evaluations[0]['train_iteration'] == -1
                and evaluations[1]['train_iteration'] == steps - 1,
                'Expected initial and final evaluations')
    require(summaries[0]['evaluations'][0]['metrics'] == summaries[1]['evaluations'][0]['metrics'],
            'Initial metrics differ')
    initial = [read_json(run, 'initial_qa_pred_val.json') for run in runs]
    require(initial[0] == initial[1], 'Initial predictions differ')
    predictions = [read_json(run, 'final_qa_pred_gt_val.json') for run in runs]
    corpus = [read_json(run, 'final_qa_corpus_val.json') for run in runs]
    require(corpus[0] == corpus[1], 'Validation references differ')
    keys = list(corpus[0])
    require(len(keys) == len(question_ids), 'Validation coverage differs from manifest')
    for run_index, prediction in enumerate(predictions):
        require(prediction.keys() == corpus[0].keys() == initial[run_index].keys(),
                'Missing or extra predictions')
        for key in keys:
            require(prediction[key]['gt'] == corpus[0][key], f'References differ for {key}')
        # BLEU and METEOR aggregate corpus statistics rather than averaging
        # sentence scores. Only the mean-based metrics can be checked here.
        for metric, score_key in (('CiDEr', 'CiDEr'), ('Rouge', 'rouge')):
            values = np.array([prediction[key]['score'][score_key] for key in keys])
            reported = summaries[run_index]['evaluations'][-1]['metrics'][metric]
            require(np.isfinite(values).all() and np.isclose(values.mean(), reported, atol=1e-10),
                    f'Per-question scores do not match reported {metric}')

    delta = np.array([predictions[1][key]['score']['CiDEr']
                      - predictions[0][key]['score']['CiDEr'] for key in keys]) * 100
    rng = np.random.default_rng(bootstrap_seed)
    # Bound memory even when the whole validation split is compared.
    bootstrap = np.empty(bootstrap_samples)
    for start in range(0, bootstrap_samples, 256):
        count = min(256, bootstrap_samples - start)
        indices = rng.integers(0, len(keys), size=(count, len(keys)))
        bootstrap[start:start + count] = delta[indices].mean(axis=1)
    return dict(
        score_scope=manifests[0]['score_scope'],
        optimizer_steps=steps, validation_questions=len(keys), seed=args[0]['seed'],
        matched_provenance={key: manifests[0][key] for key in matched_fields},
        source_sha256=source_hashes,
        shared_args={key: value for key, value in args[0].items()
                     if key not in ('checkpoint_dir', 'kl_attention_loss')},
        initial_predictions_identical=True,
        baseline=dict(directory=str(runs[0]), **summaries[0]),
        kl=dict(directory=str(runs[1]), **summaries[1]),
        cider_x100_delta=float(delta.mean()),
        paired_question_bootstrap_95ci=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
        caution='Subset scores are not comparable to full ScanQA results. The interval resamples '
                'fixed per-question scores; it does not measure training-seed variability or '
                'account for correlations between questions from the same scene.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('grounded', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap_samples', type=int, default=10000)
    parser.add_argument('--bootstrap_seed', type=int, default=0)
    args = parser.parse_args()
    report = compare(args.baseline, args.grounded, args.bootstrap_samples, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"CIDEr x100 delta: {report['cider_x100_delta']:.4f}; "
          f"paired question bootstrap 95% interval: {report['paired_question_bootstrap_95ci']}")
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
