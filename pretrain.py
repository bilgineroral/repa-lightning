import argparse
from dataclasses import asdict
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EMAWeightAveraging, EarlyStopping
from pytorch_lightning.utilities import rank_zero_only

from dataset import PretrainDataset, ImageNetDataModule
from config import PretrainConfig, PretrainConfigSmall
from repa import LitREPA

torch.serialization.add_safe_globals([PretrainConfig, PretrainConfigSmall]) # configs are pickled into checkpoints

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)

def main():
    p = argparse.ArgumentParser(description="Run REPA pretraining on ImageNet")
    p.add_argument("--data-root", type=str, required=True, help="ImageNet root with train/ and val/ (see prepare_imagenet.py)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--early-stop", type=int, default=-1, help="Enable early stopping")
    p.add_argument("--use-rope", type=int, default=1, help="Use RoPE (1) or APE (0)")
    p.add_argument("--use-qk-norm", type=int, default=1, help="Use QK-norm (1) or not (0)")
    p.add_argument("--use-layerscale", type=int, default=1, help="Use LayerScale (1) or not (0)")
    p.add_argument("--run-name", type=str, default=None, help="Override run name")
    p.add_argument("--blr", type=float, default=None, help="Override base learning rate (lr = embed_lr = blr * global batch size / 256)")
    p.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    p.add_argument("--warmup-ratio", type=float, default=None, help="Override warmup ratio (the recipe's 40 warmup epochs: 40 / max epochs)")
    p.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    p.add_argument("--global-batch-size", type=int, default=None, help="Override global batch size (sets gradient accumulation)")
    p.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers per device")
    p.add_argument("--val-every", type=int, default=None, help="Override validation (and checkpoint) interval in epochs")
    p.add_argument("--wandb-group", type=str, default=None, help="Override wandb group")
    p.add_argument("--ema-decay", type=float, default=None, help="Override EMA decay")
    p.add_argument("--use-permutation", type=int, choices=[0, 1], default=None, help="Use permutation sampling (1) or next-token sampling (0)")
    p.add_argument("--use-query-stream", type=int, choices=[0, 1], default=None, help="Predict with the query stream (1) or, as NEPA, from the content stream (0, needs --use-permutation 0)")
    p.add_argument("--model", type=str, choices=["base", "small"], default="base", help="ViT-B/16 (base) or ViT-S/16 (small) recipe")
    args = p.parse_args()

    Config = {"base": PretrainConfig, "small": PretrainConfigSmall}[args.model]
    config = Config(
        use_rope=bool(args.use_rope),
        use_qk_norm=bool(args.use_qk_norm),
        use_layerscale=bool(args.use_layerscale),
        use_permutation=bool(args.use_permutation) if args.use_permutation is not None else Config.use_permutation,
        use_query_stream=bool(args.use_query_stream) if args.use_query_stream is not None else Config.use_query_stream,
        batch_size=args.batch_size if args.batch_size is not None else Config.batch_size,
        global_batch_size=args.global_batch_size if args.global_batch_size is not None else Config.global_batch_size,
        num_workers=args.num_workers if args.num_workers is not None else Config.num_workers,
        val_every=args.val_every if args.val_every is not None else Config.val_every,
        wandb_group=args.wandb_group if args.wandb_group is not None else Config.wandb_group,
        ema_decay=args.ema_decay if args.ema_decay is not None else Config.ema_decay,
        blr=args.blr if args.blr is not None else Config.blr,
        max_epochs=args.max_epochs if args.max_epochs is not None else Config.max_epochs,
        warmup_ratio=args.warmup_ratio if args.warmup_ratio is not None else Config.warmup_ratio,
        run=args.run_name if args.run_name is not None else Config.run
    )

    rank_zero_print("Pretraining configuration:")
    for key, value in config.__dict__.items():
        rank_zero_print(f"  {key}: {value}")
    rank_zero_print("")

    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision(config.matmul_precision)

    train_dataset = PretrainDataset(
        args.data_root, "train",
        image_size=config.image_size,
        patch_size=config.patch_size,
        sl=config.sl,
        sh=config.sh,
        base_seed=config.seed,
        use_permutation=config.use_permutation
    )
    val_dataset = PretrainDataset(
        args.data_root, "val",
        image_size=config.image_size,
        patch_size=config.patch_size,
        sl=config.sl,
        sh=config.sh,
        base_seed=(config.seed + 10_000_000_000),
        use_permutation=config.use_permutation
    )
    dm = ImageNetDataModule(
        train_ds=train_dataset,
        val_ds=val_dataset,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    rank_zero_print(f"Use permutation: {config.use_permutation}, use query stream: {config.use_query_stream}")
    
    model = LitREPA(config=config)

    if config.wandb:
        logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name,
            save_dir="./wandb",
            group=config.wandb_group,
            config=asdict(config), # one W&B config entry per field (Lightning would log the dataclass as one string)
        )
    else:
        logger = True

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename=f"repa_epoch_{{epoch:03d}}_{{val/loss:.4f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=config.save_top_k,
        every_n_epochs=config.val_every,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )

    callbacks = [checkpoint_callback]
    if config.save_last:
        # separate unmonitored callback: Lightning 2.6 only refreshes save_last when a new top-k checkpoint is saved
        callbacks.append(ModelCheckpoint(
            dirpath=config.output_dir,
            filename="last",
            every_n_epochs=config.val_every,
            save_on_train_epoch_end=False,
        ))

    if config.ema_decay > 0.0:
        ema_callback = EMAWeightAveraging( # parameters only, as NEPA's EMA
            decay=config.ema_decay,
            update_every_n_steps=1,
            use_buffers=False,
        )
        callbacks.append(ema_callback)

    early_stop = EarlyStopping(
        monitor="val/loss",
        min_delta=0.0,
        patience=args.early_stop,
        verbose=True,
        mode="min"
    )
    
    if args.early_stop > 0:
        callbacks.append(early_stop)

    prec = {
        "bf16": "bf16-mixed",
        "fp16": "16-mixed",
        "fp32": "32-true", 
    }

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=prec[config.amp_dtype],
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.grad_accum_steps,
        log_every_n_steps=1,
        check_val_every_n_epoch=config.val_every,
        logger=logger,
        callbacks=callbacks,
        use_distributed_sampler=False,
        enable_progress_bar=(not config.wandb)
    )

    trainer.fit(
        model, datamodule=dm,
        ckpt_path=(args.resume if args.resume is not None else None)
    )

if __name__ == "__main__":
    main()
