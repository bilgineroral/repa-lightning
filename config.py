import torch
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

def _grad_accum_steps(global_batch_size: int, batch_size: int, world_size: int) -> int:
    if global_batch_size % (batch_size * world_size) != 0:
        raise ValueError(f"global_batch_size {global_batch_size} is not divisible by batch_size {batch_size} x {world_size} devices")
    return global_batch_size // (batch_size * world_size)

@dataclass
class PretrainConfig:
    # ------------- Model Configuration -------------
    # -----------------------------------------------
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    use_gated_mlp: bool = False
    hidden_act: Literal["gelu", "swish"] = "gelu"

    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    rope_theta: float = 100.0
    pos_embed_rescale: Optional[float] = 2.0 # RoPE coordinates scaled by a log-uniform factor in [1/x, x] (training only)
    layerscale_value: float = 1e-5

    qkv_bias: bool = True
    qk_norm: bool = True
    qk_norm_bias: bool = False
    qk_norm_affine: bool = False

    use_rope: bool = True
    use_qk_norm: bool = True
    use_layerscale: bool = True

    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0

    # ------------- Training Configuration -------------
    # --------------------------------------------------
    max_epochs: int = 800 # NEPA: 1600 (800 is MAE's ablation schedule)
    warmup_ratio: float = 0.05 # 40 epochs, as MAE and NEPA (0.025 of 1600)

    batch_size: int = 128 # per device; NEPA uses 256 (same global batch either way)
    global_batch_size: int = 4096
    grad_accum_steps: int = field(init=False) # global_batch_size / (batch_size * world_size)
    blr: float = 3e-4
    lr: float = field(init=False) # blr * global_batch_size / 256
    embed_lr: float = field(init=False)
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    world_size: int = field(init=False)
    num_workers: int = 8
    seed: int = 1337
    use_permutation: bool = True # random generation order (raster order otherwise)
    use_query_stream: bool = True # off (with use_permutation off): NEPA, the content stream predicts the next patch
    sl: int = 1 # minimum prefix size excluded from the loss in random-order prediction
    sh: int = 1 # maximum prefix size (inclusive)

    compile: bool = True
    matmul_precision: Literal["highest", "high", "medium"] = "highest" # PyTorch default, which NEPA (HF Trainer) keeps
    amp_dtype: Literal["fp16", "bf16", "fp32"] = "bf16"

    ema_decay: float = 0.0 # finetuning will start with the last checkpoint with raw weights
    save_top_k: int = 1
    save_last: bool = True
    val_every: int = 10 # epochs; checkpoints are saved after each validation
    output_dir: str = field(init=False)

    wandb: bool = True
    wandb_run_name: str = field(init=False)
    wandb_group: str = None
    wandb_project: str = "REPA"
    run: str = "repa-b-patch16-224-pretrain"

    # Derived attributes
    def __post_init__(self):
        # Override qk_norm / layerscale based on ablation flags
        if not self.use_qk_norm:
            self.qk_norm = False
        if not self.use_layerscale:
            self.layerscale_value = None
        if self.use_permutation and not self.use_query_stream:
            raise ValueError("random-order prediction needs the query stream (use_permutation requires use_query_stream)")

        self.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.grad_accum_steps = _grad_accum_steps(self.global_batch_size, self.batch_size, self.world_size)
        self.lr = self.blr * self.global_batch_size / 256
        self.embed_lr = self.lr
        self.output_dir = f"checkpoints/{self.run}"
        self.wandb_run_name = self.run

@dataclass
class PretrainConfigSmall(PretrainConfig):
    hidden_size: int = 384
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    batch_size: int = 256 # per device, as NEPA-B (same global batch)
    run: str = "repa-s-patch16-224-pretrain"

@dataclass
class ClassificationConfig:
    # ------------- Model Configuration -------------
    # -----------------------------------------------
    num_classes: int = 1000
    pooling: Literal["mean", "cls"] = "mean" # mean of the patch tokens (NEPA's add_pooling_layer) or the CLS token
    drop_path: float = 0.1
    layer_norm_eps: float = 1e-12 # fc_norm
    qk_norm_affine: bool = True # learnable QK-norm scales, initialized to 1
    pos_embed_rescale: Optional[float] = 2.0
    freeze_patch_embed: bool = True
    backbone_weights: Literal["raw", "ema"] = "raw" # NEPA finetunes from the raw weights (init_nepa_cls_from_pretrain.py without --use_ema)

    # ------------- Data Configuration -------------
    # ----------------------------------------------
    auto_augment: str = "rand-m9-mstd0.5-inc1"
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1
    crop_pct: float = 0.875 # eval: resize to image_size / crop_pct, then center crop

    mixup: float = 0.8
    cutmix: float = 1.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    smoothing: float = 0.1

    # ------------- Training Configuration -------------
    # --------------------------------------------------
    max_epochs: int = 100
    warmup_ratio: float = 0.05 # MAE uses 0.05, NEPA paper says 0.05 but code defaults to 0.2

    batch_size: int = 128 # per device
    global_batch_size: int = 1024
    grad_accum_steps: int = field(init=False) # global_batch_size / (batch_size * world_size)
    blr: float = 1.5e-3
    lr: float = field(init=False) # blr * global_batch_size / 256
    head_lr: float = 1e-3
    llrd: float = 0.65
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0

    world_size: int = field(init=False)
    num_workers: int = 8
    seed: int = 1337

    compile: bool = True
    matmul_precision: Literal["highest", "high", "medium"] = "highest" # PyTorch default, which NEPA (HF Trainer) keeps
    amp_dtype: Literal["fp16", "bf16", "fp32"] = "bf16"

    ema_decay: float = 0.9999
    save_top_k: int = 1
    save_last: bool = True
    val_every: int = 1 # epochs; checkpoints are saved after each validation
    output_dir: str = field(init=False)

    wandb: bool = True
    wandb_run_name: str = field(init=False)
    wandb_group: str = None
    wandb_project: str = "REPA"
    run: str = "repa-b-patch16-224-finetune-cls"

    # Derived attributes
    def __post_init__(self):
        self.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.grad_accum_steps = _grad_accum_steps(self.global_batch_size, self.batch_size, self.world_size)
        self.lr = self.blr * self.global_batch_size / 256
        self.output_dir = f"checkpoints/{self.run}"
        self.wandb_run_name = self.run

@dataclass
class FinetuneConfigSmall(ClassificationConfig):
    # ViT-S/16: iBOT's changes from its ViT-B to its ViT-S finetuning (evaluation/README.md: 200 instead of 100 epochs,
    # layer decay 0.75 instead of 0.65, drop path 0.1 as NEPA-B), keeping NEPA-B's 20 warmup epochs
    max_epochs: int = 200
    llrd: float = 0.75
    run: str = "repa-s-patch16-224-finetune-cls"

@dataclass
class SegmentationConfig:
    # UPerNet on ADE20K: mmsegmentation's MAE recipe (configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-512x512.py),
    # which NEPA follows. Changes implied by NEPA are marked "NEPA"
    # ------------- Model Configuration -------------
    # -----------------------------------------------
    num_classes: int = 150
    out_indices: Tuple[int, ...] = (3, 5, 7, 11) # backbone blocks feeding the 4x, 2x, 1x, 0.5x pyramid levels
    channels: int = 768 # UPerHead
    aux_channels: int = 256 # FCNHead on the third pyramid level
    aux_loss_weight: float = 0.4
    dropout: float = 0.1
    drop_path: float = 0.1
    qk_norm_affine: bool = True # NEPA: learnable QK-norm scales when finetuning (MAE adds zero-init relative position bias instead)
    pos_embed_rescale: Optional[float] = 2.0 # NEPA: RoPE coordinate rescaling when finetuning
    backbone_weights: Literal["raw", "ema"] = "raw" # NEPA finetunes from the raw pretraining weights

    # ------------- Data Configuration -------------
    # ----------------------------------------------
    crop_size: int = 512
    img_scale: Tuple[int, int] = (2048, 512) # resize to fit (long, short) side
    ratio_range: Tuple[float, float] = (0.5, 2.0) # training: random rescaling of img_scale
    cat_max_ratio: float = 0.75 # training: resample crops dominated by one class
    slide_stride: int = 341 # inference: sliding 512 windows
    # ImageNet RGB normalization on [0, 255] pixels, as in mmsegmentation's MAE recipe
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)

    # ------------- Training Configuration -------------
    # --------------------------------------------------
    max_steps: int = 160_000
    warmup_steps: int = 1500 # linear from warmup_factor * lr, then polynomial decay (power 1) to 0
    warmup_factor: float = 1e-6

    batch_size: int = 4 # per device
    global_batch_size: int = 16
    grad_accum_steps: int = field(init=False) # global_batch_size / (batch_size * world_size)
    lr: float = 1e-4
    layer_decay: float = 0.65
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)
    grad_clip: Optional[float] = None

    world_size: int = field(init=False)
    num_workers: int = 4
    seed: int = 1337

    compile: bool = True
    matmul_precision: Literal["highest", "high", "medium"] = "highest"
    amp_dtype: Literal["fp16", "bf16", "fp32"] = "bf16"

    save_top_k: int = 1
    save_last: bool = True
    val_every: int = 4_000 # steps (mmseg's MAE recipe: 16k); checkpoints are saved after each validation
    vis_indices: Tuple[int, ...] = (0, 500, 1000, 1500) # validation images logged to W&B with their predictions (none: ())
    output_dir: str = field(init=False)

    wandb: bool = True
    wandb_run_name: str = field(init=False)
    wandb_group: str = None
    wandb_project: str = "REPA"
    run: str = "repa-b-patch16-224-finetune-seg"

    # Derived attributes
    def __post_init__(self):
        self.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.grad_accum_steps = _grad_accum_steps(self.global_batch_size, self.batch_size, self.world_size)
        self.output_dir = f"checkpoints/{self.run}"
        self.wandb_run_name = self.run

@dataclass
class SegmentationConfigSmall(SegmentationConfig):
    # ViT-S/16 UPerNet: iBOT's ViT-S config (evaluation/semantic_segmentation/configs/upernet/vit_small_512_ade20k_160k.py;
    # its ViT-B config has the base recipe's layer decay 0.65 and 768 channels)
    channels: int = 384 # UPerHead, the embedding width as for ViT-B
    lr: float = 3e-5
    layer_decay: float = 0.9
    run: str = "repa-s-patch16-224-finetune-seg"

if __name__ == "__main__":
    for config in (PretrainConfig(), ClassificationConfig(), SegmentationConfig(),
                   PretrainConfigSmall(), FinetuneConfigSmall(), SegmentationConfigSmall()):
        print(type(config).__name__)
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")
