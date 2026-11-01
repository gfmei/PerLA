import argparse
import os, sys

import torch
from torch.multiprocessing import set_start_method

cwd = os.getcwd()
sys.path.append(cwd)

sys.path.append(os.path.join(cwd, 'datasets'))
sys.path.append(os.path.join(cwd, 'models'))
sys.path.append(os.path.join(cwd, 'libs'))

from datasets.dloader import build_dataset
from models.perla.engine import do_train
from models.perla.model_general import CaptionNet
from models.perla.attention_grounding import validate_grounding_settings
from libs.dist import init_distributed, is_distributed, get_rank
from libs.misc import (load_checkpoint, load_model_state, resume_if_possible,
                       seed_everything, validate_model_configuration)


def make_args_parser():
    parser = argparse.ArgumentParser("PerLA", add_help=False)
    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float,
        help="Max L2 norm of the gradient"
    )
    # DISABLE warmup learning rate during dense caption training
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    # only ACTIVATE during dense caption training
    parser.add_argument("--pretrained_params_lr", default=None, type=float)
    parser.add_argument("--pretrained_weights", default=None, type=str)

    ##### Model #####
    # input based parameters
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--no_height", default=False, action="store_true")
    parser.add_argument("--use_multiview", default=False, action="store_true")
    parser.add_argument("--n_neighs", default=8, type=int)
    parser.add_argument("--n_splits", default=4, type=int)
    parser.add_argument("--n_clus", default=128, type=int)
    parser.add_argument("--nlatent_query", default=32, type=int)
    parser.add_argument("--fusion_method", choices=("baseline", "gated", "instruction"), default="baseline",
                        help="Local/global fusion: original, adaptive gate, or instruction-conditioned gate")
    parser.add_argument("--neighbor_selection", choices=("hilbert", "geometry"), default="hilbert",
                        help="Hilbert neighbors or geometric reranking with spatial attention bias")

    parser.add_argument("--detector", default="perla", help="folder of the detector")
    parser.add_argument("--captioner", default=None, type=str, help="folder of the captioner")
    # training strategy
    parser.add_argument(
        "--freeze_detector", default=False, action='store_true',
        help="freeze all parameters other than the caption head"
    )
    parser.add_argument(
        "--freeze_llm", default=False, action='store_true',
        help="freeze the llm for caption generation"
    )
    # caption related hyper parameters
    parser.add_argument(
        "--use_beam_search", default=False, action='store_true',
        help='whether use beam search during caption generation.'
    )
    parser.add_argument(
        "--max_des_len", default=128, type=int,
        help="maximum length of object descriptions."
    )
    parser.add_argument(
        "--max_gen_len", default=32, type=int,
        help="maximum length of object descriptions."
    )

    ##### Dataset #####
    parser.add_argument("--max_prompts", default=16, type=int, help="number of visual interactions")
    parser.add_argument("--dataset", default='scannet', help="dataset list split by ','")
    # root_dir = '/storage2/TEV/datasets/ScanNet/perla'
    parser.add_argument("--root_dir", default='/data/disk1/data/scannet/scannet_llm', help="dataset root")
    parser.add_argument("--grid_size_3d", default=255, type=int, help="grid size of the 3D scene")
    parser.add_argument('--vocab', default="llama-hf/7B", type=str, help="The LLM backend")
    parser.add_argument('--qformer_vocab', default="bert-base-uncased", type=str, help="The QFormer backend")

    parser.add_argument("--dataset_num_workers", default=2, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=1080, type=int)
    parser.add_argument("--start_eval_after", default=-1, type=int)
    parser.add_argument("--eval_every_iteration", default=4000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--local_loss_weight", default=1.0, type=float,
                        help="Weight of the local auxiliary loss (0 disables its gradient)")
    parser.add_argument("--global_loss_weight", default=1.0, type=float,
                        help="Weight of the global auxiliary loss (0 disables its gradient)")
    parser.add_argument("--kl_attention_loss", action="store_true",
                        help="Supervise Q-Former spatial attention with annotated object regions")
    parser.add_argument("--kl_attention_weight", default=1.0, type=float)
    parser.add_argument("--kl_attention_smoothing", default=0.05, type=float)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument(
        "--test_min_iou", default=0.50, type=float,
        help='minimum iou for evaluating dense caption performance'
    )
    parser.add_argument(
        "--criterion", default='CiDEr', type=str,
        help='metrics for saving the best model'
    )
    parser.add_argument("--test_ckpt", default="", type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--save_every", default=4000, type=int)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--filter_name", default='captioner.transformer.', type=str)

    ##### Distributed #####
    parser.add_argument("--ngpus", default=1, type=int, help='number of gpus')
    parser.add_argument("--dist_url", default='tcp://localhost:12345', type=str)

    args = parser.parse_args()
    args.use_height = not args.no_height
    validate_grounding_settings(args.kl_attention_weight, args.kl_attention_smoothing)
    if args.kl_attention_loss and args.captioner is None:
        parser.error("--kl_attention_loss requires --captioner perla")

    return args


def main(local_rank, args):
    if args.test_only and not args.test_ckpt:
        raise ValueError("Please specify a checkpoint using --test_ckpt for evaluation")
    if args.checkpoint_dir is None:
        if not args.test_ckpt:
            raise ValueError("Please specify an output directory using --checkpoint_dir")
        args.checkpoint_dir = os.path.dirname(args.test_ckpt) or "."
        print(f'testing directory: {args.checkpoint_dir}')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    resuming = os.path.isfile(os.path.join(args.checkpoint_dir, "checkpoint.pth"))
    if not args.test_only and not resuming:
        if args.pretrained_weights is not None:
            if not os.path.isfile(args.pretrained_weights):
                raise FileNotFoundError(f"Pretrained checkpoint does not exist: {args.pretrained_weights}")
        elif args.freeze_detector and args.detector is not None:
            raise ValueError("--freeze_detector requires --pretrained_weights or a resumable checkpoint")

    if args.ngpus > 1:
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    torch.cuda.set_device(local_rank)
    seed_everything(args.seed + get_rank())

    # build datasets and dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)
    model = CaptionNet(args, dataset_config, nlatent_query=args.nlatent_query)

    # testing phase
    if args.test_only:
        ckpt = load_checkpoint(args.test_ckpt)
        validate_model_configuration(args, ckpt)
        load_model_state(model, ckpt.get("model", ckpt),
                         require_detector=getattr(args, "freeze_detector", False), require_captioner=True)

        # model_no_ddp = model.cuda()
        model = model.cuda(local_rank)

        if is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )

        for test_loader in dataloaders['test']:
            test_loader.dataset.eval_func(
                args,
                -1,
                model,
                dataset_config,
                test_loader
            )

    # training phase
    else:

        assert (
                args.checkpoint_dir is not None
        ), "Please specify a checkpoint dir using --checkpoint_dir"
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # whether or not use pretrained weights
        if resuming:
            print("Resuming model and optimizer from the output directory checkpoint")
        elif args.pretrained_weights is not None:
            checkpoint = load_checkpoint(args.pretrained_weights)
            load_model_state(model, checkpoint.get("model", checkpoint),
                             require_detector=args.freeze_detector)
        else:
            print("No pre-trained weights provided. Starting training from scratch...")

        model_no_ddp = model.cuda()
        model = model.cuda(local_rank)

        if is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                # Box/click prompt branches are optional and differ by task.
                model, device_ids=[local_rank], find_unused_parameters=True
            )

        if args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                filter(lambda params: params.requires_grad, model_no_ddp.parameters()),
                lr=args.base_lr,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                filter(lambda params: params.requires_grad, model_no_ddp.parameters()),
                lr=args.base_lr,
                weight_decay=args.weight_decay
            )
        else:
            raise NotImplementedError

        print('====                                          ====')
        print('====  Only training the following parameters  ====')
        print('====                                          ====')
        for name, param in model_no_ddp.named_parameters():
            if param.requires_grad is True:
                print('\t', name, param.shape)

        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer, args=args
        )
        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )

    if is_distributed():
        torch.distributed.destroy_process_group()


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        # Create a temporary directory
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,), join=True)


if __name__ == "__main__":
    args = make_args_parser()

    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    torch.backends.cudnn.benchmark = True

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
