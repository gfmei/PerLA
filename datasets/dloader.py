import importlib
import os
import sys

import torch

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, 'libs'))
sys.path.append(os.path.join(cwd, 'datasets'))

from libs.dist import is_distributed
from libs.misc import my_worker_init_fn


def build_dataloader_func(args, dataset, split):
    if is_distributed():
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=(split == 'train')
        )
    else:
        if split == "train":
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batchsize_per_gpu,
        num_workers=args.dataset_num_workers,
        worker_init_fn=my_worker_init_fn,
    )
    return sampler, dataloader


SCENE_FILES = ('_aligned_vert.npy', '_ins_label.npy', '_sem_label.npy', '_aligned_bbox.npy', '_spt.npy')


def skip_unprepared_scenes(dataset, root_dir):
    """Drop training annotations whose scene files are absent, reporting the count."""
    files = set(os.listdir(root_dir))
    scenes = {anno['scene_id'] for anno in dataset.annotations}
    missing = {scene for scene in scenes
               if any(scene + suffix not in files for suffix in SCENE_FILES)}
    if missing:
        kept = [anno for anno in dataset.annotations if anno['scene_id'] not in missing]
        print(f"skipped {len(dataset.annotations) - len(kept)} {dataset.task_name} annotations "
              f"from unprepared scenes {sorted(missing)}")
        dataset.annotations = kept


def build_dataset(args):
    from datasets.data_config import DatasetConfig

    dataset_config = DatasetConfig(args)
    datasets = {'train': None, 'test': []}

    train_datasets = []
    if not args.test_only:
        for dataset in args.dataset.split(','):
            dataset_module = importlib.import_module(f'datasets.{dataset}')
            train_datasets.append(
                dataset_module.Dataset(
                    args,
                    dataset_config,
                    split_set="train",
                    use_color=args.use_color,
                    use_normal=args.use_normal,
                    use_multiview=args.use_multiview,
                    use_height=args.use_height,
                    augment=True
                )
            )
            if getattr(args, 'skip_unprepared_scenes', False):
                skip_unprepared_scenes(train_datasets[-1], args.root_dir)
    # Validation may cover fewer tasks than training, e.g. ScanQA for a generalist.
    for dataset in (getattr(args, 'eval_dataset', None) or args.dataset).split(','):
        dataset_module = importlib.import_module(f'datasets.{dataset}')
        datasets['test'].append(
            dataset_module.Dataset(
                args,
                dataset_config,
                split_set="val",
                use_color=args.use_color,
                use_normal=args.use_normal,
                use_multiview=args.use_multiview,
                use_height=args.use_height,
                augment=False
            )
        )
    train_sampler, train_loader = None, None
    if not args.test_only:
        datasets['train'] = torch.utils.data.ConcatDataset(train_datasets)
        train_sampler, train_loader = build_dataloader_func(args, datasets['train'], split='train')

    dataloaders = {
        'train': train_loader,
        'test': [],
        'train_sampler': train_sampler,
    }
    for dataset in datasets['test']:
        _, test_loader = build_dataloader_func(args, dataset, split='test')
        dataloaders['test'].append(test_loader)

    return dataset_config, datasets, dataloaders
