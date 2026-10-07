"""
Cost profiler for the pretraining (two-stream REPA) and finetuning
(single-stream REPA + classification head) stages.

Reports:
  - Parameter count (total + breakdown)
  - FLOPs per training step (forward + backward) at the configured batch
  - FLOPs per inference step (forward only) at the configured batch
  - Peak GPU memory in train mode (fwd + bwd + AdamW step)
  - Peak GPU memory in inference mode (forward only)

Memory is reported under bf16-mixed autocast (the actual training precision
per PretrainConfig/FinetuneConfig.amp_dtype).
"""

import gc
import torch
import torch.nn.functional as F
from lightning.fabric.utilities.throughput import measure_flops

from config import PretrainConfig, ClassificationConfig
from repa import LitREPA, prediction_loss
from classifier import LitClassifier

# -----------------------
# CONFIG
# -----------------------
BATCH = 128
H = W = PretrainConfig.image_size
C = PretrainConfig.num_channels
PATCH = PretrainConfig.patch_size
NUM_PATCHES = (H // PATCH) * (W // PATCH)
DEVICE = "cuda"
AMP_DTYPE = torch.bfloat16  # matches *Config.amp_dtype = "bf16"

torch.manual_seed(0)

# -----------------------
# helpers
# -----------------------
def n_params(m):
    return sum(p.numel() for p in m.parameters())

def n_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def reset_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def make_inputs(B=BATCH, mask_ratio=0.95):
    img = torch.randn(B, C, H, W, device=DEVICE)
    pos = torch.arange(NUM_PATCHES, device=DEVICE).expand(B, NUM_PATCHES).contiguous()
    s_val = int(NUM_PATCHES * (1.0 - mask_ratio))
    s = torch.full((B,), s_val, device=DEVICE, dtype=torch.long)
    return img, pos, s

# ============================================================
# PRETRAIN  (LitREPA  — two-stream content/query encoder)
# ============================================================
print("=" * 64)
print("PRETRAIN  (LitREPA / two-stream REPA)")
print("=" * 64)

pre_cfg = PretrainConfig(
    drop_path=0.0,
    compile=False,
)

pre_model = LitREPA(config=pre_cfg).to(DEVICE)
print(f"Parameters (total)       : {n_params(pre_model)/1e6:7.2f} M")
print(f"  patch_embed            : {n_params(pre_model.model.patch_embed)/1e6:7.2f} M")
print(f"  encoder ({pre_cfg.num_hidden_layers} REPA layers): {n_params(pre_model.model.encoder)/1e6:7.2f} M")
print(f"  other (cls + w + norm)  : {(n_params(pre_model.model) - n_params(pre_model.model.patch_embed) - n_params(pre_model.model.encoder))/1e6:7.2f} M")

img, pos, s = make_inputs()

# ---- FLOPs: training (fwd + bwd) ----
pre_model.train()

def _pre_fwd():
    return pre_model(img, position_ids=pos)

def _pre_loss(out):
    tgt, pred = out
    return prediction_loss(tgt, pred, s=s)

pre_flops_train = measure_flops(pre_model, _pre_fwd, _pre_loss)
print(f"FLOPs / step  train  (fwd+bwd) @ B={BATCH}: {pre_flops_train/1e12:7.3f} TFLOPs  "
      f"({pre_flops_train/BATCH/1e9:7.3f} GFLOPs / sample)")

# ---- FLOPs: inference (fwd only) ----
pre_model.eval()
pre_flops_infer = measure_flops(pre_model, _pre_fwd)
print(f"FLOPs / step  infer  (fwd)     @ B={BATCH}: {pre_flops_infer/1e12:7.3f} TFLOPs  "
      f"({pre_flops_infer/BATCH/1e9:7.3f} GFLOPs / sample)")

# ---- Peak memory: training under bf16-mixed ----
pre_model.train()
opt = torch.optim.AdamW(pre_model.parameters(), lr=1e-4)

# warm-up step to allocate optimizer state / select cudnn kernels
for _ in range(2):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        tgt, pred = pre_model(img, position_ids=pos)
        loss = prediction_loss(tgt, pred, s=s)
    loss.backward()
    opt.step()
torch.cuda.synchronize()

reset_mem()
opt.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
    tgt, pred = pre_model(img, position_ids=pos)
    loss = prediction_loss(tgt, pred, s=s)
loss.backward()
opt.step()
torch.cuda.synchronize()
print(f"Peak GPU memory  train (bf16-mixed, fwd+bwd+AdamW step) @ B={BATCH}: {peak_mb():7.1f} MB")

# ---- Peak memory: inference under bf16-mixed ----
# free the training state (AdamW state, gradients, outputs) so that only the weights stay resident
del opt, tgt, pred, loss
pre_model.zero_grad(set_to_none=True)
pre_model.eval()
reset_mem()
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
    _ = pre_model(img, position_ids=pos)
torch.cuda.synchronize()
print(f"Peak GPU memory  infer (bf16, fwd only)                  @ B={BATCH}: {peak_mb():7.1f} MB")

# free GPU before next stage
del pre_model
torch.cuda.empty_cache()

# ============================================================
# FINETUNE  (LitClassifier: single-stream REPA + classification head)
# ============================================================
print()
print("=" * 64)
print("FINETUNE  (LitClassifier: single-stream REPA + classification head)")
print("=" * 64)

ft_cfg = ClassificationConfig(
    drop_path=0.0,
    compile=False,
)
ft_model = LitClassifier(config=ft_cfg, backbone_config=pre_cfg).to(DEVICE)

print(f"Parameters (total)              : {n_params(ft_model)/1e6:7.2f} M")
print(f"  encoder (single-stream REPA) : {(n_params(ft_model.patch_embed) + n_params(ft_model.encoder) + ft_model.cls_token.numel())/1e6:7.2f} M")
print(f"  norms + head ({ft_cfg.pooling} pooling)  : {(n_params(ft_model) - n_params(ft_model.patch_embed) - n_params(ft_model.encoder) - ft_model.cls_token.numel())/1e6:7.2f} M")

img = torch.randn(BATCH, C, H, W, device=DEVICE)
labels = torch.randint(0, ft_cfg.num_classes, (BATCH,), device=DEVICE)

# ---- FLOPs: training (fwd + bwd) ----
ft_model.train()

def _ft_fwd():
    return ft_model(img)

def _ft_loss(logits):
    return F.cross_entropy(logits, labels)

ft_flops_train = measure_flops(ft_model, _ft_fwd, _ft_loss)
print(f"FLOPs / step  train  (fwd+bwd) @ B={BATCH}: {ft_flops_train/1e12:7.3f} TFLOPs  "
      f"({ft_flops_train/BATCH/1e9:7.3f} GFLOPs / sample)")

# ---- FLOPs: inference (fwd only) ----
ft_model.eval()
ft_flops_infer = measure_flops(ft_model, _ft_fwd)
print(f"FLOPs / step  infer  (fwd)     @ B={BATCH}: {ft_flops_infer/1e12:7.3f} TFLOPs  "
      f"({ft_flops_infer/BATCH/1e9:7.3f} GFLOPs / sample)")

# ---- Peak memory: training under bf16-mixed ----
ft_model.train()
opt = torch.optim.AdamW(ft_model.parameters(), lr=1e-4)

for _ in range(2):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        loss = F.cross_entropy(ft_model(img), labels)
    loss.backward()
    opt.step()
torch.cuda.synchronize()

reset_mem()
opt.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
    loss = F.cross_entropy(ft_model(img), labels)
loss.backward()
opt.step()
torch.cuda.synchronize()
print(f"Peak GPU memory  train (bf16-mixed, fwd+bwd+AdamW step) @ B={BATCH}: {peak_mb():7.1f} MB")

# ---- Peak memory: inference under bf16-mixed ----
# free the training state (AdamW state, gradients, outputs) so that only the weights stay resident
del opt, loss
ft_model.zero_grad(set_to_none=True)
ft_model.eval()
reset_mem()
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
    _ = ft_model(img)
torch.cuda.synchronize()
print(f"Peak GPU memory  infer (bf16, fwd only)                  @ B={BATCH}: {peak_mb():7.1f} MB")
