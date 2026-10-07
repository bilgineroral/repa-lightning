import argparse
from dataclasses import asdict
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from ade20k import ADE20KDataset, ADE20KDataModule
from config import PretrainConfig, PretrainConfigSmall, SegmentationConfig, SegmentationConfigSmall
from repa import load_backbone
from segmentation import LitSegmenter

torch.serialization.add_safe_globals([PretrainConfig, PretrainConfigSmall, SegmentationConfig, SegmentationConfigSmall]) # configs are pickled into checkpoints

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)

def main():
    p = argparse.ArgumentParser(description="Finetune a REPA backbone with UPerNet for ADE20K semantic segmentation")
    p.add_argument("--backbone-ckpt", type=str, default=None, help="Pretraining checkpoint (trains from scratch if omitted)")
    p.add_argument("--backbone-weights", type=str, choices=["raw", "ema"], default=None, help="Finetune from the raw or the EMA pretraining weights")
    p.add_argument("--data-root", type=str, required=True, help="ADE20K root with images/ and annotations/ (see prepare_ade20k.py)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--run-name", type=str, default=None, help="Override run name")
    p.add_argument("--output-root", type=str, default=None, help="Override checkpoint output root")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--layer-decay", type=float, default=None, help="Override layer-wise lr decay")
    p.add_argument("--max-steps", type=int, default=None, help="Override the number of training iterations")
    p.add_argument("--warmup-steps", type=int, default=None, help="Override warmup iterations")
    p.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    p.add_argument("--global-batch-size", type=int, default=None, help="Override global batch size (sets gradient accumulation)")
    p.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers per device")
    p.add_argument("--val-every", type=int, default=None, help="Override validation (and checkpoint) interval in steps")
    p.add_argument("--wandb-group", type=str, default=None, help="Override wandb group")
    p.add_argument("--model", type=str, choices=["base", "small"], default="base", help="ViT-B/16 (base) or ViT-S/16 (small) recipe, matching the backbone checkpoint")
    args = p.parse_args()

    Config, BackboneConfig = {"base": (SegmentationConfig, PretrainConfig), "small": (SegmentationConfigSmall, PretrainConfigSmall)}[args.model]
    config = Config(
        backbone_weights=args.backbone_weights if args.backbone_weights is not None else Config.backbone_weights,
        lr=args.lr if args.lr is not None else Config.lr,
        layer_decay=args.layer_decay if args.layer_decay is not None else Config.layer_decay,
        max_steps=args.max_steps if args.max_steps is not None else Config.max_steps,
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else Config.warmup_steps,
        batch_size=args.batch_size if args.batch_size is not None else Config.batch_size,
        global_batch_size=args.global_batch_size if args.global_batch_size is not None else Config.global_batch_size,
        num_workers=args.num_workers if args.num_workers is not None else Config.num_workers,
        val_every=args.val_every if args.val_every is not None else Config.val_every,
        wandb_group=args.wandb_group if args.wandb_group is not None else Config.wandb_group,
        run=args.run_name if args.run_name is not None else Config.run
    )
    if args.output_root is not None:
        config.output_dir = str(Path(args.output_root) / config.run)

    rank_zero_print("Segmentation configuration:")
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

    model = LitSegmenter(config=config, backbone_config=backbone_config, backbone=backbone)

    dm = ADE20KDataModule(
        train_ds=ADE20KDataset(args.data_root, "training", config, base_seed=config.seed),
        val_ds=ADE20KDataset(args.data_root, "validation", config),
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_batches=config.max_steps * config.grad_accum_steps,
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

    callbacks = [ModelCheckpoint(
        dirpath=config.output_dir,
        filename=f"seg_step_{{step:06d}}_{{val/mIoU:.2f}}",
        monitor="val/mIoU",
        mode="max",
        save_top_k=config.save_top_k,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )]
    if config.save_last:
        # separate unmonitored callback: Lightning 2.6 only refreshes save_last when a new top-k checkpoint is saved
        callbacks.append(ModelCheckpoint(
            dirpath=config.output_dir,
            filename="last",
            save_on_train_epoch_end=False,
        ))

    prec = {
        "bf16": "bf16-mixed",
        "fp16": "16-mixed",
        "fp32": "32-true",
    }

    # iteration-based (mmsegmentation's IterBasedTrainLoop): one epoch spanning max_steps, validation every val_every steps
    # and after the last one (LitSegmenter.on_train_batch_end); training metrics logged every 50 steps (mmseg's LoggerHook)
    trainer = pl.Trainer(
        max_steps=config.max_steps,
        max_epochs=-1,
        accelerator="auto",
        devices="auto",
        precision=prec[config.amp_dtype],
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.grad_accum_steps,
        log_every_n_steps=50,
        val_check_interval=config.val_every * config.grad_accum_steps,
        check_val_every_n_epoch=None,
        sync_batchnorm=True,
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
