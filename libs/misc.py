# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import random
import tempfile

import torch
import numpy as np
from collections import deque
from typing import List
from libs.dist import is_distributed, barrier, all_reduce_sum, is_primary


def save_checkpoint(
        checkpoint_dir,
        model_no_ddp,
        optimizer,
        epoch,
        args,
        best_val_metrics,
        filename=None,
):
    if not is_primary():
        return
    if filename is None:
        filename = f"checkpoint_{epoch:04d}.pth"
    checkpoint_name = os.path.join(checkpoint_dir, filename)

    weight_ckpt = model_no_ddp.state_dict()
    parameter_names = list(weight_ckpt.keys())
    for name in parameter_names:
        if args.filter_name is not None and args.filter_name in name:
            weight_ckpt.pop(name)

    sd = {
        "model": weight_ckpt,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "best_val_metrics": best_val_metrics,
        "optimizer_steps": getattr(args, "optimizer_steps", 0),
        "skipped_steps": getattr(args, "skipped_steps", 0),
    }
    # A time limit or node failure must not destroy the last complete epoch.
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=checkpoint_dir, prefix=".checkpoint-", delete=False) as target:
            temporary_name = target.name
            torch.save(sd, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, checkpoint_name)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_checkpoint(path):
    """Load a trusted PerLA checkpoint, including CLI arguments and metric scalars."""
    if hasattr(torch.serialization, "safe_globals"):
        # CIDEr and ROUGE return NumPy scalars in existing training checkpoints.
        numpy_core = getattr(np, "_core", None)
        if numpy_core is None:
            numpy_core = np.core
        allowed_globals = [
            argparse.Namespace, numpy_core.multiarray.scalar, np.dtype,
            type(np.dtype(np.float32)), type(np.dtype(np.float64)),
        ]
        if numpy_core.multiarray.scalar.__module__ != "numpy.core.multiarray":
            # NumPy 2 renamed the module; checkpoints from NumPy 1 use this name.
            allowed_globals.append((numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"))
        with torch.serialization.safe_globals(allowed_globals):
            return torch.load(path, map_location="cpu", weights_only=True)
    # Older PyTorch cannot allowlist Namespace in weights-only mode. Preserve
    # its legacy loading behavior for trusted PerLA training checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_state(model, state_dict, require_detector=False, require_captioner=False):
    """Allow partial captioner weights, but never silently freeze a random detector."""
    target = model.state_dict()
    matching = target.keys() & state_dict.keys()
    if not matching:
        raise ValueError("Checkpoint has no matching model keys; check the architecture and key prefixes")
    if require_detector:
        required = {name for name in target if name.startswith("detector.")
                    and not name.endswith("num_batches_tracked")}
        missing = required - state_dict.keys()
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(f"Frozen detector is missing {len(missing)} checkpoint tensors: {preview}")
    if require_captioner:
        required = {name for name, parameter in model.named_parameters()
                    if name.startswith("captioner.") and parameter.requires_grad}
        missing = required - state_dict.keys()
        unexpected = {name for name in state_dict.keys() - target.keys() if name.startswith("captioner.")}
        if missing or unexpected:
            raise ValueError(f"Captioner checkpoint does not match the selected method: "
                             f"{len(missing)} missing, {len(unexpected)} unexpected parameters")
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {len(matching)} model tensors; "
          f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
    return result


def seed_everything(seed):
    """Seed model initialization, prompt selection, and point sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_model_configuration(args, checkpoint):
    """Evaluation/resume must use the architecture saved with the checkpoint."""
    saved = checkpoint.get("args")
    if saved is None:
        return
    saved = vars(saved) if isinstance(saved, argparse.Namespace) else saved
    defaults = {"fusion_method": "baseline", "neighbor_selection": "hilbert"}
    for name in ("fusion_method", "neighbor_selection", "n_splits", "n_neighs", "nlatent_query"):
        if hasattr(args, name) and (name in saved or name in defaults):
            expected = saved.get(name, defaults.get(name))
            if getattr(args, name) != expected:
                raise ValueError(f"Checkpoint requires --{name} {expected}; got {getattr(args, name)}. "
                                 "To change architecture, warm-start in a fresh output directory.")


def validate_grounding_configuration(args, checkpoint):
    """Resume the same objective; warm-starts and inference remain unrestricted."""
    saved = checkpoint.get('args')
    saved = vars(saved) if isinstance(saved, argparse.Namespace) else (saved or {})
    defaults = {'kl_attention_loss': False, 'kl_attention_weight': 1.0,
                'kl_attention_smoothing': 0.05}
    names = ['kl_attention_loss']
    if getattr(args, 'kl_attention_loss', False):
        names.extend(['kl_attention_weight', 'kl_attention_smoothing'])
    for name in names:
        expected = saved.get(name, defaults[name])
        if getattr(args, name, defaults[name]) != expected:
            raise ValueError(f"Checkpoint requires --{name} {expected}; "
                             "warm-start in a fresh output directory to change the KL objective.")


def resume_if_possible(checkpoint_dir, model_no_ddp, optimizer, args=None):
    """
    Resume if checkpoint is available.
    Return
    - epoch of loaded checkpoint.
    """
    epoch = -1
    best_val_metrics = {}
    if not os.path.isdir(checkpoint_dir):
        return epoch, best_val_metrics

    last_checkpoint = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.isfile(last_checkpoint):
        return epoch, best_val_metrics

    sd = load_checkpoint(last_checkpoint)
    if args is not None:
        validate_model_configuration(args, sd)
        validate_grounding_configuration(args, sd)
    epoch = sd["epoch"]
    best_val_metrics = sd["best_val_metrics"]
    print(f"Found checkpoint at {epoch}. Resuming.")

    load_model_state(model_no_ddp, sd["model"],
                     require_detector=getattr(model_no_ddp, "freeze_detector", False), require_captioner=True)
    optimizer.load_state_dict(sd["optimizer"])
    if args is not None:
        args.optimizer_steps = sd.get("optimizer_steps", 0)
        args.skipped_steps = sd.get("skipped_steps", 0)
    print(
        f"Loaded model and optimizer state at {epoch}. Loaded best val metrics so far."
    )
    return epoch, best_val_metrics

def my_worker_init_fn(worker_id):
    # DataLoader gives each worker a fresh seed each epoch and per rank.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.jit.ignore
def to_list_1d(arr) -> List[float]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr


@torch.jit.ignore
def to_list_3d(arr) -> List[List[List[float]]]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr


def huber_loss(error, delta=1.0):
    """
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss


# From https://github.com/facebookresearch/detr/blob/master/util/misc.py
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_distributed():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        barrier()
        all_reduce_sum(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
