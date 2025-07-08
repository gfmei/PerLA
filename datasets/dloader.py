import importlib
import os
import sys

import torch

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, 'libs'))
sys.path.append(os.path.join(cwd, 'datasets'))

from libs.dist import is_distributed
from datasets.data_config import DatasetConfig
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


def build_dataset(args):
    dataset_config = DatasetConfig(args)
    datasets = {'train': None, 'test': []}

    train_datasets = []
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