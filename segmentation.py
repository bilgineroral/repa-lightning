"""
UPerNet semantic segmentation on ADE20K: mmsegmentation's MAE recipe (MAE backbone + Feature2Pyramid neck + UPerHead
+ FCNHead auxiliary head, configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-512x512.py), which NEPA follows, with a
REPA backbone. Modules keep mmsegmentation's names, so its layer-wise lr decay rule applies as is.
Inputs use ImageNet mean / std, matching the MAE recipe.

Changes from the MAE recipe, as implied by NEPA:
  - backbone: the REPA content stream with bidirectional attention (NEPA: "we disable causal masking")
  - positions: RoPE on the [-1, 1] patch coordinates of each crop instead of MAE's interpolated absolute position
    embedding plus zero-initialized relative position bias; learnable QK-norm scales and RoPE coordinate rescaling as
    NEPA's classification finetuning
"""
import math
import os
from collections import deque
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from config import PretrainConfig, SegmentationConfig
from ade20k import CLASSES, IGNORE_INDEX
from dataset import seed_training_batch
from modules import REPA, SingleStreamREPAEncoder, add_qk_norm_affine
from rope import prepend_prefix_tokens

def resize(x: torch.Tensor, size, align_corners: bool = False) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)

class SegBackbone(nn.Module):
    """ REPA content stream with bidirectional attention, returning the patch-token feature maps after the out_indices
    blocks (as mmsegmentation's MAE backbone: CLS token kept in the sequence, no final norm) """
    def __init__(self, backbone: REPA, config: SegmentationConfig, backbone_config: PretrainConfig):
        super().__init__()
        self.patch_size = backbone_config.patch_size
        assert config.crop_size % self.patch_size == 0, "crop_size must be a multiple of the patch size (predictions would be stretched)"
        self.out_indices = tuple(config.out_indices)
        self.patch_embed = backbone.patch_embed
        self.rope_embed = backbone.rope_embed
        self.pos_embed = backbone.pos_embed
        self.cls_token = backbone.cls_token
        self.layers = SingleStreamREPAEncoder(backbone.encoder, drop_path=config.drop_path).layers

        if self.rope_embed is not None:
            self.rope_embed.rescale = config.pos_embed_rescale
        if config.qk_norm_affine:
            add_qk_norm_affine(self.layers, backbone_config)

    def _pos_embed(self, grid: tuple) -> torch.Tensor:
        """ APE resized to the input grid (bicubic, as mmsegmentation's resize_abs_pos_embed) """
        n = int(self.pos_embed.shape[1] ** 0.5)
        if (n, n) == grid:
            return self.pos_embed
        pos = self.pos_embed.reshape(1, n, n, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=grid, mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).flatten(1, 2)

    def forward(self, img: torch.Tensor) -> tuple:
        x = self.patch_embed(img) # [B,N,D], raster order
        B, _, D = x.shape
        grid = (img.shape[-2] // self.patch_size, img.shape[-1] // self.patch_size)

        # Position embeddings: RoPE applied in attention (CLS unrotated), APE added to patch tokens
        if self.rope_embed is not None:
            position_embeds = prepend_prefix_tokens(self.rope_embed(img))
        else:
            x = x + self._pos_embed(grid)
            position_embeds = None

        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, position_embeds)
            if i in self.out_indices:
                outs.append(x[:, 1:].reshape(B, *grid, D).permute(0, 3, 1, 2).contiguous())
        return tuple(outs)

class ConvModule(nn.Module):
    """ mmcv ConvModule: conv (no bias) -> BatchNorm (SyncBatchNorm under DDP) -> ReLU, kaiming (fan_out) init """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, dilation: int = 1, inplace: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=inplace)
        nn.init.kaiming_normal_(self.conv.weight, a=0, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))

class Feature2Pyramid(nn.Module):
    """ mmseg Feature2Pyramid: rescales the four backbone feature maps by 4, 2, 1, 0.5 """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.upsample_4x = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.upsample_2x = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
        self.identity = nn.Identity()
        self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs: tuple) -> tuple:
        ops = [self.upsample_4x, self.upsample_2x, self.identity, self.downsample_2x]
        return tuple(op(x) for op, x in zip(ops, inputs))

class DecodeHead(nn.Module):
    """ mmseg BaseDecodeHead: dropout and the 1x1 classifier (normal(std=0.01) init) """
    def __init__(self, channels: int, num_classes: int, dropout_ratio: float):
        super().__init__()
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        nn.init.normal_(self.conv_seg.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv_seg.bias, 0)

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)

class PPM(nn.ModuleList):
    """ mmseg Pooling Pyramid Module """
    def __init__(self, pool_scales: tuple, in_channels: int, channels: int):
        super().__init__()
        for pool_scale in pool_scales:
            self.append(nn.Sequential(nn.AdaptiveAvgPool2d(pool_scale), ConvModule(in_channels, channels, 1)))

    def forward(self, x):
        return [resize(ppm(x), x.shape[2:]) for ppm in self]

class UPerHead(DecodeHead):
    """ mmseg UPerHead: PPM on the coarsest level + FPN """
    def __init__(self, in_channels: list, channels: int, num_classes: int, dropout_ratio: float = 0.1, pool_scales=(1, 2, 3, 6)):
        super().__init__(channels, num_classes, dropout_ratio)
        self.psp_modules = PPM(pool_scales, in_channels[-1], channels)
        self.bottleneck = ConvModule(in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for c in in_channels[:-1]: # skip the top layer
            self.lateral_convs.append(ConvModule(c, channels, 1, inplace=False))
            self.fpn_convs.append(ConvModule(channels, channels, 3, padding=1, inplace=False))
        self.fpn_bottleneck = ConvModule(len(in_channels) * channels, channels, 3, padding=1)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        return self.bottleneck(torch.cat(psp_outs, dim=1))

    def forward(self, inputs: tuple) -> torch.Tensor:
        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))
        # build top-down path
        levels = len(laterals)
        for i in range(levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + resize(laterals[i], laterals[i - 1].shape[2:])
        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(levels - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(levels - 1, 0, -1):
            fpn_outs[i] = resize(fpn_outs[i], fpn_outs[0].shape[2:])
        feats = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        return self.cls_seg(feats)

class FCNHead(DecodeHead):
    """ mmseg FCNHead with num_convs=1, concat_input=False, on one pyramid level """
    def __init__(self, in_channels: int, channels: int, num_classes: int, in_index: int = 2, dropout_ratio: float = 0.1):
        super().__init__(channels, num_classes, dropout_ratio)
        self.in_index = in_index
        self.convs = nn.Sequential(ConvModule(in_channels, channels, 3, padding=1))

    def forward(self, inputs: tuple) -> torch.Tensor:
        return self.cls_seg(self.convs(inputs[self.in_index]))

def seg_loss(logits: torch.Tensor, label: torch.Tensor) -> tuple:
    """ mmseg BaseDecodeHead.loss_by_feat: logits upsampled to the label size, then the CrossEntropyLoss (avg_non_ignore=False:
    summed over the non-ignored pixels and divided by all pixels) and the pixel accuracy acc_seg in % (mmseg accuracy:
    over the non-ignored pixels, eps keeps an all-ignored batch finite) """
    logits = resize(logits, label.shape[-2:])
    eps = torch.finfo(torch.float32).eps
    loss = F.cross_entropy(logits.float(), label, ignore_index=IGNORE_INDEX, reduction="none")
    with torch.no_grad():
        valid = label != IGNORE_INDEX
        acc = (((logits.argmax(1) == label) & valid).sum() + eps) / (valid.sum() + eps) * 100
    return loss.sum() / (label.numel() + eps), acc

def intersect_and_union(pred: torch.Tensor, label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ mmseg IoUMetric.intersect_and_union -> [4, num_classes]: intersection, union, prediction and label areas """
    mask = label != IGNORE_INDEX
    pred, label = pred[mask], label[mask]
    intersect = pred[pred == label]
    hist = lambda t: torch.histc(t.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_intersect, area_pred, area_label = hist(intersect), hist(pred), hist(label)
    return torch.stack([area_intersect, area_pred + area_label - area_intersect, area_pred, area_label])

def iou_metrics(areas: torch.Tensor) -> dict:
    """ mmseg IoUMetric.total_area_to_metrics for summed areas: mIoU, mAcc (nan-mean over classes), aAcc, in % """
    intersect, union, _, label = areas.double()
    return {
        "mIoU": torch.nanmean(intersect / union).item() * 100,
        "mAcc": torch.nanmean(intersect / label).item() * 100,
        "aAcc": (intersect.sum() / label.sum()).item() * 100,
    }

def get_layer_id_for_vit(var_name: str, max_layer_id: int) -> int:
    """ mmseg layer_decay_optimizer_constructor.get_layer_id_for_vit """
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.layers"):
        return int(var_name.split(".")[2]) + 1
    else:
        return max_layer_id - 1

def poly_warmup_factor(step: int, warmup_steps: int, max_steps: int, warmup_factor: float) -> float:
    """ mmengine LinearLR(start_factor=warmup_factor, end=warmup_steps) then PolyLR(power=1, eta_min=0, end=max_steps):
    the lr of optimizer step `step` (0-based) relative to the base lr, in closed form """
    if step < warmup_steps:
        return warmup_factor + (1.0 - warmup_factor) * step / (warmup_steps - 1)
    return max(0.0, 1.0 - (step - warmup_steps) / (max_steps - warmup_steps - 1))

class LitSegmenter(pl.LightningModule):
    def __init__(self, config: SegmentationConfig, backbone_config: PretrainConfig, backbone: Optional[REPA] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"], logger=False) # segment.py logs the configs to W&B field by field
        if backbone is None:
            backbone = REPA(backbone_config)
        backbone = getattr(backbone, "_orig_mod", backbone) # compiled pretraining model

        dim = backbone_config.hidden_size
        self.backbone = SegBackbone(backbone, config, backbone_config)
        self.neck = Feature2Pyramid(dim)
        self.decode_head = UPerHead([dim] * 4, config.channels, config.num_classes, dropout_ratio=config.dropout)
        self.auxiliary_head = FCNHead(dim, config.aux_channels, config.num_classes, in_index=2, dropout_ratio=config.dropout)
        if config.compile:
            self.backbone.compile()
        self._recent_losses = deque(maxlen=10) # mmengine LogProcessor's window_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """ img: [B,3,H,W] -> logits [B,num_classes,H,W] (mmseg EncoderDecoder.encode_decode) """
        logits = self.decode_head(self.neck(self.backbone(img)))
        return resize(logits, img.shape[2:])

    @torch.no_grad()
    def slide_inference(self, img: torch.Tensor) -> torch.Tensor:
        """ mmseg EncoderDecoder.slide_inference: average of the logits of overlapping crop_size windows """
        cfg = self.hparams.config
        stride, crop = cfg.slide_stride, cfg.crop_size
        B, _, H, W = img.shape
        h_grids = max(H - crop + stride - 1, 0) // stride + 1
        w_grids = max(W - crop + stride - 1, 0) // stride + 1
        preds = img.new_zeros((B, cfg.num_classes, H, W), dtype=torch.float32)
        count = img.new_zeros((B, 1, H, W), dtype=torch.float32)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1, x1 = h_idx * stride, w_idx * stride
                y2, x2 = min(y1 + crop, H), min(x1 + crop, W)
                y1, x1 = max(y2 - crop, 0), max(x2 - crop, 0)
                crop_logits = self(img[:, :, y1:y2, x1:x2]).float()
                preds += F.pad(crop_logits, (x1, W - x2, y1, H - y2))
                count[:, :, y1:y2, x1:x2] += 1
        return preds / count

    def predict(self, img: torch.Tensor, size) -> torch.Tensor:
        """ class map at the original image size (mmseg slide inference + postprocess_result) """
        return resize(self.slide_inference(img), size).argmax(dim=1)

    def configure_optimizers(self):
        # mmseg LearningRateDecayOptimizerConstructor (layer_wise): lr * layer_decay ** (num_layers + 1 - layer_id),
        # layer_id 0 for the embeddings, i + 1 for block i and num_layers + 1 for the neck and heads
        cfg = self.hparams.config
        num_layers = len(self.backbone.layers) + 2
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # as mmseg: 1-D parameters and biases (the full names never equal "pos_embed" / "cls_token")
            if p.ndim == 1 or name.endswith(".bias") or name in ("pos_embed", "cls_token"):
                group_name, weight_decay = "no_decay", 0.0
            else:
                group_name, weight_decay = "decay", cfg.weight_decay
            layer_id = get_layer_id_for_vit(name, num_layers)
            group_name = f"layer_{layer_id}_{group_name}"
            if group_name not in groups:
                scale = cfg.layer_decay ** (num_layers - layer_id - 1)
                groups[group_name] = {"params": [], "weight_decay": weight_decay, "lr": scale * cfg.lr, "lr_scale": scale}
            groups[group_name]["params"].append(p)

        optimizer = torch.optim.AdamW(list(groups.values()), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: poly_warmup_factor(step, cfg.warmup_steps, cfg.max_steps, cfg.warmup_factor)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        img, label = batch["img"], batch["label"]
        feats = self.neck(self.backbone(img))
        loss_decode, acc_decode = seg_loss(self.decode_head(feats), label)
        loss_aux, acc_aux = seg_loss(self.auxiliary_head(feats), label)
        loss_aux = self.hparams.config.aux_loss_weight * loss_aux
        loss = loss_decode + loss_aux

        # as mmseg's LoggerHook (written every log_every_n_steps = 50 iterations) with mmengine's LogProcessor: rank 0's
        # losses averaged over the last 10 iterations, accuracies and lr of the current one
        self._recent_losses.append(torch.stack([loss, loss_decode, loss_aux]).detach())
        avg_loss, avg_loss_decode, avg_loss_aux = torch.stack(tuple(self._recent_losses)).mean(0)
        lr = max(pg["lr"] for pg in self.trainer.optimizers[0].param_groups) # mmseg's base_lr: neck and heads have lr scale 1
        self.log_dict({"train/loss": avg_loss, "train/loss_decode": avg_loss_decode, "train/loss_aux": avg_loss_aux,
                       "train/acc_seg": acc_decode, "train/acc_seg_aux": acc_aux, "train/lr": lr}, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        # per-batch random draws (dropout, RoPE rescaling, drop path), so a resumed run matches an uninterrupted one;
        # batch_idx keeps counting across a mid-epoch resume (one epoch spans the whole schedule)
        seed_training_batch(self.hparams.config.seed, self.current_epoch, batch_idx, self.global_rank)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # as mmseg's IterBasedTrainLoop: validate (and so checkpoint) after the last iteration too, also when max_steps is
        # not a multiple of val_every; Lightning runs validation when a stop is requested
        if self.trainer.global_step >= self.trainer.max_steps:
            self.trainer.should_stop = True

    def setup(self, stage):
        # with W&B, fail now rather than at the first validation, val_every steps into training
        if isinstance(self.logger, WandbLogger):
            n = len(self.trainer.datamodule.val_ds)
            if not all(0 <= i < n for i in self.hparams.config.vis_indices):
                raise ValueError(f"vis_indices {self.hparams.config.vis_indices} must index the {n} validation images")

    def on_validation_epoch_start(self):
        self._areas = torch.zeros(4, self.hparams.config.num_classes, dtype=torch.float64, device=self.device)

    def validation_step(self, batch, batch_idx):
        img, label = batch["img"], batch["label"]
        pred = self.predict(img, label.shape[-2:])
        for p, l in zip(pred, label):
            self._areas += intersect_and_union(p, l, self.hparams.config.num_classes).double()

    def on_validation_epoch_end(self):
        areas = self.trainer.strategy.reduce(self._areas, reduce_op="sum") # each image is on exactly one rank
        for name, value in iou_metrics(areas).items():
            self.log(f"val/{name}", value, prog_bar=(name == "mIoU"), sync_dist=False, rank_zero_only=False)
        # on rank 0, which logs to W&B, and not for the sanity check before training (Lightning doesn't log its metrics)
        if isinstance(self.logger, WandbLogger) and self.trainer.is_global_zero and not self.trainer.sanity_checking:
            self._log_predictions()

    def _log_predictions(self):
        """ The config's vis_indices validation images as W&B images, with their ground truth and predicted masks at the
        original size (as evaluated). Rank 0 predicts them again, whichever rank validated them """
        cfg, val_ds = self.hparams.config, self.trainer.datamodule.val_ds
        if not cfg.vis_indices:
            return
        class_labels = dict(enumerate(CLASSES)) | {IGNORE_INDEX: "other (ignored)"}
        images = []
        for i in cfg.vis_indices:
            item = val_ds[i]
            with self.trainer.precision_plugin.forward_context(): # the autocast of the validation steps
                pred = self.predict(item["img"][None].to(self.device), item["label"].shape)[0]
            img_path = val_ds.samples[i][0]
            images.append(wandb.Image(img_path, caption=os.path.basename(img_path), masks={
                "prediction": {"mask_data": pred.to(torch.uint8).cpu().numpy(), "class_labels": class_labels},
                "ground_truth": {"mask_data": item["label"].to(torch.uint8).numpy(), "class_labels": class_labels},
            }))
        # commit=False: W&B adds them to its next row, the val metrics that Lightning logs right after this hook
        self.logger.experiment.log({"val/predictions": images}, commit=False)

if __name__ == "__main__":
    pt_config = PretrainConfig()
    seg_config = SegmentationConfig()
    model = LitSegmenter(config=seg_config, backbone_config=pt_config)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    img = torch.randn(2, 3, seg_config.crop_size, seg_config.crop_size)
    label = torch.randint(0, seg_config.num_classes, (2, seg_config.crop_size, seg_config.crop_size))
    logits = model(img)
    loss, _ = seg_loss(logits, label)
    loss.backward()
    print(f"logits shape: {logits.shape}, loss: {loss.item():.4f}")
