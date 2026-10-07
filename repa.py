import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import math

from config import PretrainConfig
from dataset import seed_training_batch
from modules import REPA
from optim import get_decay_parameter_names, hf_accumulation_scale

def prediction_loss(tgt, pred, s=None):
    """ tgt:  [B, T, D], pred: [B, T, D], s: [B] or None (targets before s are excluded) """
    tgt = tgt.detach()
    p = F.normalize(pred, dim=-1)
    z = F.normalize(tgt, dim=-1)

    loss = 1.0 - (p * z).sum(dim=-1) # [B, N]
    if s is None: # every target is predicted, the first one from the CLS token (as NEPA)
        return loss.mean()

    s = s.to(device=loss.device)
    include = torch.arange(loss.shape[1], device=loss.device)[None, :] >= s[:, None]
    return loss[include].mean()

class LitREPA(pl.LightningModule):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.save_hyperparameters(logger=False) # pretrain.py logs the config to W&B field by field
        model = REPA(config)
        if config.compile:
            model = torch.compile(model)
        self.model = model

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

        return torch.optim.AdamW([g for g in groups if g["params"]], lr=cfg.lr, betas=cfg.betas)

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
        loss = prediction_loss(tgt, pred, s=s)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return loss * hf_accumulation_scale(batch_idx, self.trainer.num_training_batches, self.trainer.accumulate_grad_batches)

    def validation_step(self, batch, batch_idx):
        img, position_ids, s = batch["img"], batch["position_ids"], batch["s"]
        tgt, pred = self.model(img, position_ids=position_ids)
        s = s if self.hparams.config.use_permutation else None
        loss = prediction_loss(tgt, pred, s=s)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def setup(self, stage):
        # same sampling for every validation; set here since Lightning creates the val
        # iterator (workers start prefetching) before on_validation_epoch_start runs
        self.trainer.datamodule.val_ds.set_epoch(0)

    def on_train_epoch_start(self):
        self.trainer.datamodule.train_ds.set_epoch(self.current_epoch)

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
