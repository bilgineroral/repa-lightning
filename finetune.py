import argparse
from dataclasses import asdict
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EMAWeightAveraging
from pytorch_lightning.utilities import rank_zero_only

from dataset import ImageNetDataset, ImageNetDataModule, build_finetune_transform
from config import PretrainConfig, PretrainConfigSmall, ClassificationConfig, FinetuneConfigSmall
from repa import load_backbone
from classifier import LitClassifier

torch.serialization.add_safe_globals([PretrainConfig, PretrainConfigSmall, ClassificationConfig, FinetuneConfigSmall]) # configs are pickled into checkpoints

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)

def main():
    p = argparse.ArgumentParser(description="Finetune a REPA backbone for ImageNet classification")
    p.add_argument("--backbone-ckpt", type=str, default=None, help="Pretraining checkpoint (trains from scratch if omitted)")
    p.add_argument("--data-root", type=str, required=True, help="ImageNet root with train/ and val/ (see prepare_imagenet.py)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--run-name", type=str, default=None, help="Override run name")
    p.add_argument("--output-root", type=str, default=None, help="Override checkpoint output root")
    p.add_argument("--backbone-weights", type=str, choices=["raw", "ema"], default=None, help="Finetune from the raw or the EMA pretraining weights")
    p.add_argument("--pooling", type=str, choices=["mean", "cls"], default=None, help="Classify from the mean of the patch tokens or the CLS token")
    p.add_argument("--blr", type=float, default=None, help="Override base learning rate (lr = blr * global batch size / 256)")
    p.add_argument("--head-lr", type=float, default=None, help="Override classification head learning rate")
    p.add_argument("--llrd", type=float, default=None, help="Override layer-wise lr decay")
    p.add_argument("--drop-path", type=float, default=None, help="Override drop path rate")
    p.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    p.add_argument("--warmup-ratio", type=float, default=None, help="Override warmup ratio")
    p.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    p.add_argument("--global-batch-size", type=int, default=None, help="Override global batch size (sets gradient accumulation)")
    p.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers per device")
    p.add_argument("--ema-decay", type=float, default=None, help="Override EMA decay (0 disables EMA)")
    p.add_argument("--val-every", type=int, default=None, help="Override validation (and checkpoint) interval in epochs")
    p.add_argument("--wandb-group", type=str, default=None, help="Override wandb group")
    p.add_argument("--model", type=str, choices=["base", "small"], default="base", help="ViT-B/16 (base) or ViT-S/16 (small) recipe, matching the backbone checkpoint")
    args = p.parse_args()

    Config, BackboneConfig = {"base": (ClassificationConfig, PretrainConfig), "small": (FinetuneConfigSmall, PretrainConfigSmall)}[args.model]
    config = Config(
        backbone_weights=args.backbone_weights if args.backbone_weights is not None else Config.backbone_weights,
        pooling=args.pooling if args.pooling is not None else Config.pooling,
        blr=args.blr if args.blr is not None else Config.blr,
        head_lr=args.head_lr if args.head_lr is not None else Config.head_lr,
        llrd=args.llrd if args.llrd is not None else Config.llrd,
        drop_path=args.drop_path if args.drop_path is not None else Config.drop_path,
        max_epochs=args.max_epochs if args.max_epochs is not None else Config.max_epochs,
        warmup_ratio=args.warmup_ratio if args.warmup_ratio is not None else Config.warmup_ratio,
        batch_size=args.batch_size if args.batch_size is not None else Config.batch_size,
        global_batch_size=args.global_batch_size if args.global_batch_size is not None else Config.global_batch_size,
        num_workers=args.num_workers if args.num_workers is not None else Config.num_workers,
        ema_decay=args.ema_decay if args.ema_decay is not None else Config.ema_decay,
        freeze_patch_embed=Config.freeze_patch_embed and bool(args.backbone_ckpt), # keep it trainable from scratch
        val_every=args.val_every if args.val_every is not None else Config.val_every,
        wandb_group=args.wandb_group if args.wandb_group is not None else Config.wandb_group,
        run=args.run_name if args.run_name is not None else Config.run
    )
    if args.output_root is not None:
        config.output_dir = str(Path(args.output_root) / config.run)

    rank_zero_print("Finetuning configuration:")
    for key, value in config.__dict__.items():
        rank_zero_print(f"  {key}: {value}")
    rank_zero_print("")

    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision(config.matmul_precision)

    if args.backbone_ckpt:
        rank_zero_print(f"Loading backbone from checkpoint: {args.backbone_ckpt}")
        backbone_config, backbone = load_backbone(args.backbone_ckpt, config.backbone_weights)
        if backbone_config.hidden_size != BackboneConfig.hidden_size:
            raise ValueError(f"--model {args.model} is the recipe for {BackboneConfig.hidden_size}-wide backbones, but the checkpoint's is {backbone_config.hidden_size}-wide")
        rank_zero_print(f"Using the {config.backbone_weights} pretraining weights")
    else:
        rank_zero_print("[WARNING] No backbone checkpoint provided, training from scratch!")
        backbone_config, backbone = BackboneConfig(), None

    model = LitClassifier(config=config, backbone_config=backbone_config, backbone=backbone)

    train_dataset = ImageNetDataset(
        args.data_root, "train",
        transform=build_finetune_transform(config, backbone_config.image_size, train=True),
        base_seed=config.seed,
    )
    val_dataset = ImageNetDataset(
        args.data_root, "val",
        transform=build_finetune_transform(config, backbone_config.image_size, train=False),
    )
    dm = ImageNetDataModule(
        train_ds=train_dataset,
        val_ds=val_dataset,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    if config.wandb:
        logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name,
            group=config.wandb_group,
            save_dir="./wandb",
            config={**asdict(config), "backbone": asdict(backbone_config)}, # one W&B config entry per field
        )
    else:
        logger = True

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename=f"cls_epoch_{{epoch:03d}}_{{val/acc1:.2f}}",
        monitor="val/acc1",
        mode="max",
        save_top_k=config.save_top_k,
        every_n_epochs=config.val_every,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )

    callbacks = [checkpoint_callback]
    if config.ema_decay > 0.0: # validation and checkpoints use the EMA weights
        callbacks.append(EMAWeightAveraging(decay=config.ema_decay, update_every_n_steps=1, use_buffers=False))
    if config.save_last:
        # separate unmonitored callback: Lightning 2.6 only refreshes save_last when a new top-k checkpoint is saved
        callbacks.append(ModelCheckpoint(
            dirpath=config.output_dir,
            filename="last",
            every_n_epochs=config.val_every,
            save_on_train_epoch_end=False,
        ))

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
