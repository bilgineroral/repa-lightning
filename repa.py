import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import math

from config import PretrainConfig
from dataset import IMAGE_MEAN, IMAGE_STD, seed_training_batch
from modules import REPA
from optim import get_decay_parameter_names, hf_accumulation_scale

def prediction_loss(tgt, pred, s=None, sync_free: bool = False):
    """ tgt:  [B, T, D], pred: [B, T, D], s: [B] or None (targets before s are excluded)
    sync_free: the same mean as a masked sum, without boolean indexing, which waits for the GPU (perf_no_cpu_sync) """
    tgt = tgt.detach()
    p = F.normalize(pred, dim=-1)
    z = F.normalize(tgt, dim=-1)

    loss = 1.0 - (p * z).sum(dim=-1) # [B, N]
    if s is None: # every target is predicted, the first one from the CLS token (as NEPA)
        return loss.mean()

    s = s.to(device=loss.device)
    include = torch.arange(loss.shape[1], device=loss.device)[None, :] >= s[:, None]
    if sync_free:
        weight = include.to(loss.dtype)
        return (loss * weight).sum() / weight.sum()
    return loss[include].mean()

def fp8_supported() -> bool:
    """ torchao float8 training needs FP8 tensor cores: compute capability >= 8.9 (Ada, Hopper or newer) """
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)

def convert_to_fp8(model):
    """ perf_fp8: swap nn.Linear for torchao's Float8Linear (its default recipe: dynamic tensorwise scaling). FP8 GEMMs
    need both matrix dims divisible by 16; the patch embedding is a Conv2d and stays bf16. Parameter names and
    checkpoints are unchanged. """
    from torchao.float8 import convert_to_float8_training
    return convert_to_float8_training(
        model, module_filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and m.in_features % 16 == 0 and m.out_features % 16 == 0
    )

class LitREPA(pl.LightningModule):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.save_hyperparameters(logger=False) # pretrain.py logs the config to W&B field by field
        model = REPA(config)
        if getattr(config, "perf_fp8", False): # before torch.compile, as torchao requires
            if not fp8_supported():
                cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
                raise RuntimeError(f"perf_fp8 needs a GPU with compute capability >= 8.9 (Ada / Hopper); found {cap}")
            model = convert_to_fp8(model)
        if config.compile:
            model = torch.compile(model)
        self.model = model

        # speed flags; getattr: configs pickled into older checkpoints don't have them
        self.perf_no_cpu_sync = getattr(config, "perf_no_cpu_sync", False)
        self.perf_fused_adamw = getattr(config, "perf_fused_adamw", False)
        if self.perf_fused_adamw and config.amp_dtype == "fp16":
            raise ValueError("perf_fused_adamw supports bf16 / fp32 only: fp16 needs the GradScaler path (see configure_gradient_clipping)")
        self._lr_sum, self._lr_count = 0.0, 0 # perf_no_cpu_sync: train/lr epoch mean, accumulated on the host
        # perf_uint8_images: ToTensor's scaling and Normalize on the GPU (not saved in checkpoints)
        self.register_buffer("img_mean", torch.tensor(IMAGE_MEAN).view(-1, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGE_STD).view(-1, 1, 1), persistent=False)
        # a tensor, not the Python 255: CUDA divides by a Python scalar via its reciprocal, which isn't bit-identical
        self.register_buffer("img_255", torch.tensor(255.0), persistent=False)

    def _create_optimizer(self):
        # NEPA pretraining (run_nepa.py EnhancedTrainer.create_optimizer): weight decay except for
        # biases, norms and layer scales; embeddings (patch embedding, CLS, query token, APE) at embed_lr
        cfg = self.hparams.config
        decay = set(get_decay_parameter_names(self))
        embed = {id(p) for p in self.model.patch_embed.parameters()} | {id(self.model.cls_token)}
        if self.model.w is not None:
            embed.add(id(self.model.w))
        if self.model.pos_embed is not None:
            embed.add(id(self.model.pos_embed))

        groups = [
            {"params": [], "weight_decay": cfg.weight_decay, "lr": cfg.embed_lr},
            {"params": [], "weight_decay": 0.0, "lr": cfg.embed_lr},
            {"params": [], "weight_decay": cfg.weight_decay, "lr": cfg.lr},
            {"params": [], "weight_decay": 0.0, "lr": cfg.lr},
        ]
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            groups[2 * (id(p) not in embed) + (name not in decay)]["params"].append(p)

        fused = {"fused": True} if self.perf_fused_adamw else {}
        return torch.optim.AdamW([g for g in groups if g["params"]], lr=cfg.lr, betas=cfg.betas, **fused)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if self.perf_fused_adamw and gradient_clip_val:
            # Lightning's AMP plugin refuses to clip for fused optimizers (a guard for fp16 loss scaling,
            # pytorch_lightning/plugins/precision/amp.py). bf16 / fp32 have no scaler, so this runs the clip Lightning
            # would otherwise run (Precision.clip_grad_by_norm: clip_grad_norm_ over the optimizer's parameters)
            if gradient_clip_algorithm not in (None, "norm"):
                raise ValueError(f"perf_fused_adamw supports norm clipping only, got {gradient_clip_algorithm}")
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], gradient_clip_val)
            return
        super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def configure_optimizers(self):
        optimizer = self._create_optimizer()

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = math.ceil(total_steps * self.hparams.config.warmup_ratio) # as HF Trainer's warmup_ratio

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    def forward(self, img, position_ids=None):
        return self.model(img, position_ids=position_ids)
    
    def training_step(self, batch, batch_idx):
        img, position_ids, s = batch["img"], batch["position_ids"], batch["s"]
        tgt, pred = self.model(img, position_ids=position_ids)
        s = s if self.hparams.config.use_permutation else None
        loss = prediction_loss(tgt, pred, s=s, sync_free=self.perf_no_cpu_sync)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        if self.perf_no_cpu_sync:
            # logging a Python float copies it to the GPU and waits: keep the epoch mean on the host instead
            self._lr_sum += lr
            self._lr_count += 1
        else:
            self.log("train/lr", lr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return loss * hf_accumulation_scale(batch_idx, self.trainer.num_training_batches, self.trainer.accumulate_grad_batches)

    def validation_step(self, batch, batch_idx):
        img, position_ids, s = batch["img"], batch["position_ids"], batch["s"]
        tgt, pred = self.model(img, position_ids=position_ids)
        s = s if self.hparams.config.use_permutation else None
        loss = prediction_loss(tgt, pred, s=s, sync_free=self.perf_no_cpu_sync)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def setup(self, stage):
        # same sampling for every validation; set here since Lightning creates the val
        # iterator (workers start prefetching) before on_validation_epoch_start runs
        self.trainer.datamodule.val_ds.set_epoch(0)

    def on_train_epoch_start(self):
        self.trainer.datamodule.train_ds.set_epoch(self.current_epoch)

    def on_train_epoch_end(self):
        if self.perf_no_cpu_sync and self._lr_count > 0: # the same epoch mean the per-batch log gives
            self.log("train/lr", self._lr_sum / self._lr_count, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            self._lr_sum, self._lr_count = 0.0, 0

    def on_after_batch_transfer(self, batch, dataloader_idx):
        img = batch["img"]
        if img.dtype == torch.uint8: # perf_uint8_images: the ops of ToTensor (x / 255) and Normalize, on the GPU
            batch["img"] = img.float().div(self.img_255).sub_(self.img_mean).div_(self.img_std)
        return batch

    def on_train_batch_start(self, batch, batch_idx):
        # per-batch random draws (RoPE rescaling, dropout, drop path), so a resumed run matches an uninterrupted one
        seed_training_batch(self.hparams.config.seed, self.current_epoch, batch_idx, self.global_rank)

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        mapped_state_dict = {}
        
        is_compiled = hasattr(self.model, "_orig_mod")
        
        for key, value in state_dict.items():
            # Checkpoint has _orig_mod, but model does not (compiled ckpt -> uncompiled model)
            if "model._orig_mod" in key and not is_compiled:
                new_key = key.replace("model._orig_mod", "model")
                mapped_state_dict[new_key] = value
            # Checkpoint does not have _orig_mod, but model does (uncompiled ckpt -> compiled model)
            elif "model." in key and "model._orig_mod" not in key and is_compiled:
                new_key = key.replace("model.", "model._orig_mod.")
                mapped_state_dict[new_key] = value
            else:
                mapped_state_dict[key] = value
                
        checkpoint["state_dict"] = mapped_state_dict

def load_backbone(ckpt_path: str, weights: str = "raw"):
    """ Pretrained REPA model and its config from a LitREPA checkpoint. With EMA, a checkpoint's state_dict holds the
    EMA weights and current_model_state the raw ones; NEPA finetunes from the raw weights """
    pretrained = LitREPA.load_from_checkpoint(ckpt_path, map_location="cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if weights == "raw" and "current_model_state" in ckpt:
        raw = {"state_dict": ckpt["current_model_state"]}
        pretrained.on_load_checkpoint(raw)
        pretrained.load_state_dict(raw["state_dict"])
    return pretrained.hparams.config, pretrained.model

if __name__ == "__main__":
    config = PretrainConfig()
    model = LitREPA(config=config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params/1e6:.2f}M")

    B, N = 2, (config.image_size // config.patch_size) ** 2
    img = torch.randn(B, config.num_channels, config.image_size, config.image_size)
    position_ids = torch.stack([torch.randperm(N) for _ in range(B)])
    with torch.no_grad():
        tgt, pred = model(img, position_ids=position_ids)
    print(f"tgt shape: {tgt.shape}, pred shape: {pred.shape}")
