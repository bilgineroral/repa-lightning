import math
from collections import defaultdict
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from config import PretrainConfig, ClassificationConfig
from dataset import seed_training_batch
from modules import REPA, SingleStreamREPAEncoder, add_qk_norm_affine
from optim import get_decay_parameter_names, get_llrd_cosine_schedule_with_warmup, hf_accumulation_scale
from rope import prepend_prefix_tokens

class LitClassifier(pl.LightningModule):
    """ ImageNet classification finetuning of a REPA backbone (NEPA SFT recipe), from the mean of the patch tokens or the CLS token """
    def __init__(self, config: ClassificationConfig, backbone_config: PretrainConfig, backbone: Optional[REPA] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"], logger=False) # finetune.py logs the configs to W&B field by field
        if backbone is None:
            backbone = REPA(backbone_config)
        backbone = getattr(backbone, "_orig_mod", backbone) # compiled pretraining model

        self.patch_embed = backbone.patch_embed
        self.rope_embed = backbone.rope_embed
        self.pos_embed = backbone.pos_embed
        self.cls_token = backbone.cls_token
        self.encoder = SingleStreamREPAEncoder(backbone.encoder, drop_path=config.drop_path)
        self.norm = backbone.norm
        self.fc_norm = nn.LayerNorm(backbone_config.hidden_size, eps=config.layer_norm_eps) if config.pooling == "mean" else None
        self.head = nn.Linear(backbone_config.hidden_size, config.num_classes)
        nn.init.zeros_(self.head.weight) # as NEPA's init_nepa_cls_from_pretrain.py
        nn.init.zeros_(self.head.bias)

        if self.rope_embed is not None:
            self.rope_embed.rescale = config.pos_embed_rescale
        if config.qk_norm_affine:
            add_qk_norm_affine(self.encoder.layers, backbone_config)
        if config.freeze_patch_embed:
            self.patch_embed.requires_grad_(False)
        if config.compile:
            self.encoder.compile()

        if config.mixup > 0 or config.cutmix > 0:
            self.mixup_fn = Mixup(
                mixup_alpha=config.mixup, cutmix_alpha=config.cutmix,
                prob=config.mixup_prob, switch_prob=config.mixup_switch_prob, mode="batch",
                label_smoothing=config.smoothing, num_classes=config.num_classes,
            )
            self.criterion = SoftTargetCrossEntropy()
        else:
            self.mixup_fn = None
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.smoothing)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """ img: [B,C_in,H,W] -> logits [B,num_classes] """
        x = self.patch_embed(img) # [B,N,D], raster order

        # Position embeddings: RoPE applied in attention (CLS unrotated), APE added to patch tokens
        if self.rope_embed is not None:
            position_embeds = prepend_prefix_tokens(self.rope_embed(img))
        else:
            x = x + self.pos_embed
            position_embeds = None

        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.norm(self.encoder(x, position_embeddings=position_embeds))
        x = self.fc_norm(x[:, 1:].mean(dim=1)) if self.fc_norm is not None else x[:, 0]
        return self.head(x)

    def configure_optimizers(self):
        # NEPA SFT (run_image_classification.py EnhancedTrainer.create_optimizer): lr * llrd ** (num_layers - 1 - i)
        # for layer i and ** num_layers for the embeddings; the head and final norm use head_lr without decay
        cfg = self.hparams.config
        num_layers = len(self.encoder.layers)
        decay = set(get_decay_parameter_names(self))
        grouped = defaultdict(list) # (lr, weight_decay, llrd_scale) -> params

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(("head.", "norm.")):
                lr, scale = cfg.head_lr, 0
            elif name.startswith("encoder.layers."):
                lr, scale = cfg.lr, num_layers - 1 - int(name.split(".")[2])
            elif name.startswith(("patch_embed", "pos_embed", "cls_token")):
                lr, scale = cfg.lr, num_layers
            else: # fc_norm
                lr, scale = cfg.lr, 0
            wd = 0.0 if p.ndim <= 1 or name not in decay else cfg.weight_decay
            grouped[(lr, wd, scale)].append(p)

        param_groups = [
            {"params": params, "lr": lr, "weight_decay": wd, "llrd": cfg.llrd, "llrd_scale": scale}
            for (lr, wd, scale), params in grouped.items()
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas)

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = math.ceil(total_steps * cfg.warmup_ratio)
        scheduler = get_llrd_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        img, label = batch["img"], batch["label"]
        if self.mixup_fn is not None:
            img, label = self.mixup_fn(img, label)
        loss = self.criterion(self(img), label)

        lr = max(pg["lr"] for pg in self.trainer.optimizers[0].param_groups)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train/lr", lr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return loss * hf_accumulation_scale(batch_idx, self.trainer.num_training_batches, self.trainer.accumulate_grad_batches)

    def validation_step(self, batch, batch_idx):
        img, label = batch["img"], batch["label"]
        logits = self(img)
        top5 = logits.topk(5, dim=-1).indices

        metrics = {
            "val/loss": F.cross_entropy(logits.float(), label),
            "val/acc1": (top5[:, 0] == label).float().mean() * 100,
            "val/acc5": (top5 == label[:, None]).any(dim=-1).float().mean() * 100,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=img.shape[0])

    def on_train_epoch_start(self):
        self.trainer.datamodule.train_ds.set_epoch(self.current_epoch)

    def on_train_batch_start(self, batch, batch_idx):
        # per-batch random draws (mixup / cutmix, RoPE rescaling, drop path), so a resumed run matches an uninterrupted one
        seed_training_batch(self.hparams.config.seed, self.current_epoch, batch_idx, self.global_rank)

if __name__ == "__main__":
    pt_config = PretrainConfig()
    ft_config = ClassificationConfig()
    model = LitClassifier(config=ft_config, backbone_config=pt_config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params/1e6:.2f}M")

    img = torch.randn(2, pt_config.num_channels, pt_config.image_size, pt_config.image_size)
    label = torch.randint(0, ft_config.num_classes, (2,))
    img, target = model.mixup_fn(img, label)
    logits = model(img)
    loss = model.criterion(logits, target)
    loss.backward()
    print(f"logits shape: {logits.shape}, loss: {loss.item():.4f}")
