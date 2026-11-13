import torch
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

PERF_FLAGS = (
    "skip_last_content", "causal_no_mask", "query_proj_once", "no_cpu_sync",
    "uint8_images", "fused_adamw", "ddp_static_graph",
)

EXTRA_PERF_FLAGS = ("fp8",) # accepted by name, not part of "all" (FP8 changes numerics and needs a capability >= 8.9 GPU)

def parse_perf_flags(spec: str) -> dict:
    """ --perf value ("a,b", "all", or e.g. "all,fp8") -> {"perf_a": True, "perf_b": True} """
    names = []
    for n in (n.strip() for n in (spec or "").split(",")):
        if n == "all":
            names += PERF_FLAGS
        elif n:
            names.append(n)
    unknown = [n for n in names if n not in PERF_FLAGS + EXTRA_PERF_FLAGS]
    if unknown:
        raise ValueError(f"unknown --perf flags {unknown}; choose from {list(PERF_FLAGS + EXTRA_PERF_FLAGS)} or 'all'")
    return {f"perf_{n}": True for n in names}

# Architecture speed flags (change the model; for speed benchmarks and later pilots). --arch "name[=value],..."
# "width" values accept "d" / "2d" (multiples of hidden_size) or an integer.
ARCH_FLAGS = {
    "single_stream_ref": bool, "query_top_k": int, "single_stream_target_pos": bool, "target_pos_type": str,
    "target_pos_layers": int, "target_pos_null_p": float, "query_mlp_width": "width", "train_res": int,
    "predict_frac": float,
}

def parse_arch_flags(spec: str, hidden_size: int) -> dict:
    """ --arch value, e.g. "query_top_k=4" or "single_stream_target_pos,target_pos_type=sincos" -> {"arch_...": value} """
    out = {}
    for item in (x.strip() for x in (spec or "").split(",")):
        if not item or item == "none":
            continue
        name, _, value = item.partition("=")
        kind = ARCH_FLAGS.get(name)
        if kind is None:
            raise ValueError(f"unknown --arch flag {name!r}; choose from {list(ARCH_FLAGS)}")
        if kind is bool:
            if value not in ("", "1", "true", "True"):
                raise ValueError(f"--arch {name} takes no value")
            out[f"arch_{name}"] = True
        elif kind == "width":
            out[f"arch_{name}"] = (int(value[:-1] or 1) * hidden_size) if value.endswith("d") else int(value)
        else:
            out[f"arch_{name}"] = kind(value)
    return out

# Tier 3 kernel flags (same model and loss; ~/repa-notes/kernel_research.md). --kern "patch_linear,attn_fused=gqa"
KERN_FLAGS = {"patch_linear": bool, "attn_fused": ("gqa", "flex")}

def parse_kern_flags(spec: str) -> dict:
    """ --kern value, e.g. "patch_linear" or "patch_linear,attn_fused=flex" -> {"kern_...": value} """
    out = {}
    for item in (x.strip() for x in (spec or "").split(",")):
        if not item or item == "none":
            continue
        name, _, value = item.partition("=")
        kind = KERN_FLAGS.get(name)
        if kind is None:
            raise ValueError(f"unknown --kern flag {name!r}; choose from {list(KERN_FLAGS)}")
        if kind is bool:
            if value not in ("", "1", "true", "True"):
                raise ValueError(f"--kern {name} takes no value")
            out[f"kern_{name}"] = True
        else:
            if value not in kind:
                raise ValueError(f"--kern {name} must be one of {list(kind)}, got {value!r}")
            out[f"kern_{name}"] = value
    return out

def arch_single_stream(config) -> bool:
    """ the arch modes that run the content stream alone (no query stream) """
    return getattr(config, "arch_single_stream_ref", False) or getattr(config, "arch_single_stream_target_pos", False)

def arch_predict_offset(config) -> int:
    """ arch_predict_frac: number of leading positions of each order that are neither computed nor predicted """
    frac = getattr(config, "arch_predict_frac", 0.0)
    if not frac:
        return 0
    n = (config.image_size // config.patch_size) ** 2
    return n - int(round(frac * n))

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

    # ------------- Speed flags (False = original behaviour; see bench/README.md) -------------
    # --------------------------------------------------------------------------------------------
    perf_skip_last_content: bool = False # last layer: content stream computes only K/V (the rest never reaches the loss)
    perf_causal_no_mask: bool = False # drop the last permuted patch from the content stream; is_causal SDPA, no masks
    perf_query_proj_once: bool = False # layer 1: LN + Q projection of the query token w on one row, then broadcast
    perf_no_cpu_sync: bool = False # masked-mean loss (no boolean indexing); train/lr logged once per epoch
    perf_uint8_images: bool = False # uint8 images from pinned memory, normalized on the GPU
    perf_fused_adamw: bool = False # AdamW(fused=True)
    perf_ddp_static_graph: bool = False # DDP static_graph=True (multi-GPU only)
    perf_fp8: bool = False # torchao float8 training for the Linear layers (GPU capability >= 8.9; not bit-exact)

    # ------------- Architecture flags (CHANGE the model; off = original; see bench/README.md) -------------
    # --------------------------------------------------------------------------------------------------------
    arch_single_stream_ref: bool = False # no query stream: content stream alone, causal over the permuted order
    arch_query_top_k: int = 0 # query stream only in the top K layers (0 = every layer)
    arch_single_stream_target_pos: bool = False # option B (~/repa-notes/option_b_design.md): single stream + e_tgt
    arch_target_pos_type: Literal["learned", "sincos"] = "learned" # option B: learned table or fixed 2D sin-cos
    arch_target_pos_layers: int = 0 # option B: 0 = add e_tgt at the input, k = before each of the last k layers
    arch_target_pos_null_p: float = 0.0 # option B: fraction of rows given a null token instead, excluded from the loss
    arch_query_mlp_width: int = 0 # separate query-stream MLP of this hidden width (0 = the shared MLP)
    arch_train_res: int = 0 # pretrain at this image size (0 = image_size); RoPE models only
    arch_predict_frac: float = 0.0 # predict (and compute query rows for) only the last fraction of each order (0 = all)

    # ------------- Kernel flags (same model and loss; off = original; see bench/README.md) -------------
    # ---------------------------------------------------------------------------------------------------
    kern_patch_linear: bool = False # patch embedding as reshape + one GEMM instead of Conv2d (same weights)
    # both streams' attention in one call over the shared K/V: "gqa" (SDPA enable_gqa, interleaved heads) or "flex"
    # (FlexAttention, interleaved rows, mask q // 2 >= kv); "" = two calls. Takes effect with perf_causal_no_mask only
    kern_attn_fused: str = ""

    # Derived attributes
    def __post_init__(self):
        # Override qk_norm / layerscale based on ablation flags
        if not self.use_qk_norm:
            self.qk_norm = False
        if not self.use_layerscale:
            self.layerscale_value = None
        single_stream = self.arch_single_stream_ref or self.arch_single_stream_target_pos
        if self.arch_single_stream_ref and self.arch_single_stream_target_pos:
            raise ValueError("choose one of arch_single_stream_ref and arch_single_stream_target_pos")
        if single_stream and self.use_query_stream:
            raise ValueError("arch_single_stream_* runs without the query stream: set use_query_stream=False")
        if self.use_permutation and not self.use_query_stream and not single_stream:
            raise ValueError("random-order prediction needs the query stream (use_permutation requires use_query_stream)")
        if (self.arch_query_top_k or self.arch_query_mlp_width or self.arch_predict_frac) and not self.use_query_stream:
            raise ValueError("arch_query_top_k / arch_query_mlp_width / arch_predict_frac need the query stream")
        if not 0 <= self.arch_query_top_k <= self.num_hidden_layers:
            raise ValueError(f"arch_query_top_k must be in [0, {self.num_hidden_layers}]")
        if not 0.0 <= self.arch_predict_frac < 1.0:
            raise ValueError("arch_predict_frac must be in [0, 1)")
        if not 0 <= self.arch_target_pos_layers <= self.num_hidden_layers:
            raise ValueError(f"arch_target_pos_layers must be in [0, {self.num_hidden_layers}]")
        if self.arch_train_res:
            if not self.use_rope:
                raise ValueError("arch_train_res needs RoPE (the APE table is sized for image_size)")
            if self.arch_train_res % self.patch_size:
                raise ValueError("arch_train_res must be a multiple of patch_size")
            self.image_size = self.arch_train_res
        if self.perf_fp8 and self.batch_size % 16:
            # FP8's grad_weight matmul contracts over batch x tokens rows (197 with CLS), which torch._scaled_mm needs
            # divisible by 16: only a batch that is a multiple of 16 guarantees it
            raise ValueError(f"perf_fp8 needs per-GPU batch size divisible by 16 (torch._scaled_mm), got {self.batch_size}")
        if self.kern_attn_fused not in ("", "gqa", "flex"):
            raise ValueError(f"kern_attn_fused must be '', 'gqa' or 'flex', got {self.kern_attn_fused!r}")

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
