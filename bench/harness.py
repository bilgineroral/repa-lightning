""" Shared setup for bench/step_time.py and bench/profile_step.py: runs the real LitREPA training step through the
Lightning Trainer (same H2D copy, logging, gradient clipping and optimizer as pretrain.py) on one GPU. """
import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from config import PERF_FLAGS, PretrainConfig, PretrainConfigSmall, parse_perf_flags
from dataset import IMAGE_MEAN, IMAGE_STD, ImageNetDataModule, PretrainDataset
from repa import LitREPA

def add_common_args(p):
    p.add_argument("--model", choices=["S", "B"], default="S")
    p.add_argument("--flags", default="none", help="comma-separated perf flags, 'all', or 'none' (baseline)")
    p.add_argument("--batch", type=int, default=None, help="per-device micro-batch (default: the config's)")
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--data-root", default=None, help="ImageFolder root with train/ (needed for --data real)")
    p.add_argument("--num-workers", type=int, default=8, help="dataloader workers (real data)")
    p.add_argument("--optimizer", dest="optimizer", action="store_true", default=None,
                   help="include the optimizer step (default for real data)")
    p.add_argument("--no-optimizer", dest="optimizer", action="store_false",
                   help="forward + backward only (default for synthetic data)")
    p.add_argument("--accum", type=int, default=1,
                   help="micro-batches per optimizer step with --optimizer (pretraining uses 8-16)")
    p.add_argument("--warmup", type=int, default=20, help="untimed steps first (covers torch.compile)")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--precision", default="bf16-mixed")
    return p

def flag_names(spec: str):
    if spec in ("", "none", "baseline"):
        return []
    return sorted(k[len("perf_"):] for k in parse_perf_flags(spec))

def build_config(args):
    Config = {"S": PretrainConfigSmall, "B": PretrainConfig}[args.model]
    batch = args.batch or Config.batch_size
    world = torch.cuda.device_count() # the config checks global batch % (batch * GPUs); the bench runs on one GPU
    flags = {} if not flag_names(args.flags) else parse_perf_flags(",".join(flag_names(args.flags)))
    cfg = Config(batch_size=batch, global_batch_size=batch * world, compile=not args.no_compile, wandb=False, **flags)
    torch.set_float32_matmul_precision(cfg.matmul_precision) # as pretrain.py
    return cfg

class _SyntheticBatches(torch.utils.data.Dataset):
    """ one fixed batch, returned every step (no decode cost): fp32 normalized images, or uint8 with perf_uint8_images """
    def __init__(self, cfg, n):
        g = torch.Generator().manual_seed(0)
        B, N = cfg.batch_size, (cfg.image_size // cfg.patch_size) ** 2
        shape = (B, cfg.num_channels, cfg.image_size, cfg.image_size)
        if cfg.perf_uint8_images:
            img = torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
        else:
            mean, std = torch.tensor(IMAGE_MEAN).view(-1, 1, 1), torch.tensor(IMAGE_STD).view(-1, 1, 1)
            img = (torch.rand(shape, generator=g) - mean) / std
        self.batch = {
            "img": img,
            "position_ids": torch.stack([torch.randperm(N, generator=g) for _ in range(B)]),
            "s": torch.full((B,), cfg.sl, dtype=torch.long),
        }
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.batch

    def set_epoch(self, epoch):
        pass

class _SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, cfg, n):
        super().__init__()
        self.train_ds = self.val_ds = _SyntheticBatches(cfg, n)
        self.pin = cfg.perf_uint8_images

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=None, num_workers=0, pin_memory=self.pin)

def build_datamodule(args, cfg, n_batches):
    if args.data == "synthetic":
        return _SyntheticDataModule(cfg, n_batches)
    if not args.data_root:
        raise SystemExit("--data real needs --data-root")
    kw = dict(image_size=cfg.image_size, patch_size=cfg.patch_size, sl=cfg.sl, sh=cfg.sh,
              use_permutation=cfg.use_permutation, uint8_images=cfg.perf_uint8_images)
    train = PretrainDataset(args.data_root, "train", base_seed=cfg.seed, **kw)
    return ImageNetDataModule(train_ds=train, val_ds=train, seed=cfg.seed, batch_size=cfg.batch_size,
                              num_workers=args.num_workers, pin_memory=cfg.perf_uint8_images)

class StopAfter(Callback):
    """ stops after n training batches (spans epochs if the dataset is small); keeps each step's loss on the GPU """
    def __init__(self, n):
        self.n, self.seen, self.losses = n, 0, []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.seen += 1
        self.losses.append(outputs["loss"].detach().float())
        if self.seen >= self.n:
            trainer.should_stop = True

    def loss_summary(self):
        losses = torch.stack(self.losses).cpu() if self.losses else torch.empty(0)
        return {"loss_first": losses[0].item() if len(losses) else None,
                "loss_last": losses[-1].item() if len(losses) else None,
                "loss_all_finite": bool(torch.isfinite(losses).all()) if len(losses) else None}

def build_trainer(args, cfg, callbacks):
    optimizer = args.optimizer if args.optimizer is not None else (args.data == "real")
    total = args.warmup + args.steps + 1
    # without the optimizer step: accumulate past the end of the run, so no optimizer step ever happens
    accum = args.accum if optimizer else total + 1
    trainer = pl.Trainer(
        max_epochs=10_000, accelerator="gpu", devices=1, precision=args.precision,
        gradient_clip_val=cfg.grad_clip, accumulate_grad_batches=accum,
        limit_val_batches=0, num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, use_distributed_sampler=False,
        callbacks=[StopAfter(total), *callbacks],
    )
    return trainer, optimizer, accum, total

def stop_callback(trainer):
    return next(c for c in trainer.callbacks if isinstance(c, StopAfter))

def environment():
    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", REPO, *a], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "gpu_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "lightning": pl.__version__,
        "python": sys.version.split()[0],
        "git_hash": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "hostname": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

def run_tag(args, flags):
    gpu = torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace(" ", "-")
    name = "all" if set(flags) == set(PERF_FLAGS) else ("+".join(flags) or "baseline")
    return f"{args.model}_{args.data}_{name}_b{args.batch}_{gpu}_{time.strftime('%Y%m%d-%H%M%S')}"
