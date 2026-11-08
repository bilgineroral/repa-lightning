"""
Equivalence tests for the perf_* speed flags (config.py): each flag alone and all together must give the same loss and
parameter gradients as the original code, from the same weights, batch and seed.

    pytest tests/test_perf_equivalence.py -v          # CPU tests always run; GPU tests skip without CUDA

fp32: per tensor, max |flag - baseline| <= 1e-5 * max |baseline| (rtol 1e-5 relative to the tensor's scale; parameters
whose baseline gradient is exactly zero must stay exactly zero).
bf16 (GPU): both versions are compared to an fp32 run of the original code; the flagged version passes if its error is
at most 1.5x the original's own bf16 error. The max abs diff between the two bf16 runs is reported.
Exact equivalence assumes drop path = 0 (the default); with drop path the flags change the RNG draw order.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import DIFFS
from config import PERF_FLAGS, PretrainConfig, PretrainConfigSmall, parse_perf_flags
from dataset import PretrainDataset, build_pretrain_transform
from modules import REPA
from repa import LitREPA, prediction_loss

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")

# flags that change the forward / loss; "all" also turns on the rest (they don't touch the forward pass)
FORWARD_FLAGS = ["skip_last_content", "causal_no_mask", "query_proj_once", "no_cpu_sync", "all"]
RTOL = 1e-5

def make_config(size: str, perf: str = "", **kw) -> PretrainConfig:
    flags = parse_perf_flags(perf)
    if size == "small": # the real REPA-S shapes (N = 196 patches)
        return PretrainConfigSmall(compile=False, wandb=False, **flags, **kw)
    return PretrainConfig( # tiny: 16 patches, 3 layers
        image_size=64, patch_size=16, hidden_size=64, num_attention_heads=4, num_hidden_layers=3,
        intermediate_size=128, compile=False, wandb=False, **flags, **kw,
    )

def make_batch(cfg: PretrainConfig, batch: int, device, s_high: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    N = (cfg.image_size // cfg.patch_size) ** 2
    img = torch.randn(batch, cfg.num_channels, cfg.image_size, cfg.image_size, generator=g)
    pos = torch.stack([torch.randperm(N, generator=g) for _ in range(batch)])
    s = torch.randint(1, s_high + 1, (batch,), generator=g)
    return img.to(device), pos.to(device), s.to(device)

def loss_and_grads(model, cfg, img, pos, s, amp_dtype=None):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1234) # the RoPE coordinate rescale draws one random number per forward
    with torch.autocast(img.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        tgt, pred = model(img, position_ids=pos)
        loss = prediction_loss(tgt, pred, s=s, sync_free=getattr(cfg, "perf_no_cpu_sync", False))
    loss.backward()
    grads = {n: (p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32))
             for n, p in model.named_parameters()}
    return loss.detach().float(), grads

def model_pair(size: str, perf: str, device, compile_models: bool = False):
    torch.manual_seed(0)
    cfg_ref = make_config(size)
    cfg_new = make_config(size, perf)
    ref = REPA(cfg_ref).to(device).train()
    new = REPA(cfg_new).to(device).train()
    new.load_state_dict(ref.state_dict()) # same weights: the flags add no parameters
    if compile_models:
        ref, new = torch.compile(ref), torch.compile(new)
    return (cfg_ref, ref), (cfg_new, new)

def max_rel(a: torch.Tensor, b: torch.Tensor):
    """ max |a - b| and that max relative to max |b| """
    diff = (a - b).abs().max().item()
    scale = b.abs().max().item()
    return diff, (diff / scale if scale > 0 else (0.0 if diff == 0 else float("inf")))

def check_fp32(test_id, size, perf, device, s_high=1, batch=4, compile_models=False, rtol=RTOL):
    (cfg_ref, ref), (cfg_new, new) = model_pair(size, perf, device, compile_models)
    img, pos, s = make_batch(cfg_ref, batch, device, s_high=s_high)
    loss_ref, g_ref = loss_and_grads(ref, cfg_ref, img, pos, s)
    loss_new, g_new = loss_and_grads(new, cfg_new, img, pos, s)

    d_loss, r_loss = max_rel(loss_new, loss_ref)
    worst_diff, worst_rel, worst_name = 0.0, 0.0, None
    for name in g_ref:
        d, r = max_rel(g_new[name], g_ref[name])
        if r > worst_rel or (worst_name is None):
            worst_diff, worst_rel, worst_name = d, r, name
        assert r <= rtol, f"{name}: max abs diff {d:.3e}, relative {r:.3e} > {rtol}"
    DIFFS.append((test_id, "loss", d_loss, f"rtol {rtol:g}"))
    DIFFS.append((test_id, f"grad (worst rel {worst_rel:.1e})", worst_diff, f"rtol {rtol:g}"))
    assert r_loss <= rtol, f"loss: {loss_new.item()} vs {loss_ref.item()}"

# ---------------------------------------------------------------- forward / backward, fp32

@pytest.mark.parametrize("perf", FORWARD_FLAGS)
def test_fp32_cpu_tiny(perf):
    check_fp32(f"fp32 cpu tiny [{perf}]", "tiny", perf, "cpu")

@needs_cuda
@pytest.mark.parametrize("perf", FORWARD_FLAGS)
def test_fp32_cuda_small(perf):
    """ real REPA-S shapes on the GPU """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check_fp32(f"fp32 cuda small [{perf}]", "small", perf, "cuda", batch=4)

def test_skip_last_content_keeps_zero_grads():
    """ the 4 last-layer parameters that only fed the skipped content output get exactly zero gradient, as before """
    (cfg_ref, ref), (cfg_new, new) = model_pair("tiny", "skip_last_content", "cpu")
    img, pos, s = make_batch(cfg_ref, 2, "cpu")
    _, g_ref = loss_and_grads(ref, cfg_ref, img, pos, s)
    _, g_new = loss_and_grads(new, cfg_new, img, pos, s)
    last = f"encoder.layers.{cfg_ref.num_hidden_layers - 1}"
    for name in (f"{last}.ln2_h.weight", f"{last}.ln2_h.bias", f"{last}.ls1_h.lambda1", f"{last}.ls2_h.lambda1"):
        assert torch.count_nonzero(g_ref[name]) == 0 and torch.count_nonzero(g_new[name]) == 0, name
        assert dict(new.named_parameters())[name].grad is not None, f"{name} left the graph (DDP would complain)"

def test_no_cpu_sync_loss_per_sample_s():
    """ masked mean == boolean-index mean when s varies per sample """
    g = torch.Generator().manual_seed(0)
    tgt, pred = torch.randn(8, 16, 32, generator=g), torch.randn(8, 16, 32, generator=g)
    s = torch.randint(1, 6, (8,), generator=g)
    ref = prediction_loss(tgt, pred, s=s)
    new = prediction_loss(tgt, pred, s=s, sync_free=True)
    d, r = max_rel(new, ref)
    DIFFS.append(("loss, per-sample s in [1, 5]", "loss", d, f"rtol {RTOL:g}"))
    assert r <= RTOL

# ---------------------------------------------------------------- bf16 autocast and torch.compile (GPU)

@needs_cuda
@pytest.mark.parametrize("perf", FORWARD_FLAGS)
def test_bf16_cuda_small(perf):
    torch.backends.cuda.matmul.allow_tf32 = False
    (cfg_ref, ref), (cfg_new, new) = model_pair("small", perf, "cuda")
    img, pos, s = make_batch(cfg_ref, 4, "cuda")
    loss_32, g_32 = loss_and_grads(ref, cfg_ref, img, pos, s)
    loss_ref, g_ref = loss_and_grads(ref, cfg_ref, img, pos, s, torch.bfloat16)
    loss_new, g_new = loss_and_grads(new, cfg_new, img, pos, s, torch.bfloat16)

    err_ref = max(max_rel(g_ref[n], g_32[n])[1] for n in g_32)
    err_new = max(max_rel(g_new[n], g_32[n])[1] for n in g_32)
    d_loss = (loss_new - loss_ref).abs().item()
    d_grad = max((g_new[n] - g_ref[n]).abs().max().item() for n in g_32)
    DIFFS.append((f"bf16 cuda small [{perf}]", "loss vs bf16 baseline", d_loss, "see grad"))
    DIFFS.append((f"bf16 cuda small [{perf}]", "grad vs bf16 baseline", d_grad, "<=1.5x base"))
    assert err_new <= 1.5 * err_ref + 1e-6, f"bf16 grad error {err_new:.3e} vs original's {err_ref:.3e} (fp32 reference)"
    assert abs(loss_new.item() - loss_32.item()) <= 1.5 * abs(loss_ref.item() - loss_32.item()) + 1e-4

@needs_cuda
def test_fp32_cuda_compiled_all():
    """ both models under torch.compile (as in training): Inductor reorders float ops, hence rtol 1e-4 """
    torch.backends.cuda.matmul.allow_tf32 = False
    check_fp32("fp32 cuda tiny compiled [all]", "tiny", "all", "cuda", compile_models=True, rtol=1e-4)

@needs_cuda
def test_no_cpu_sync_has_no_sync():
    """ the masked mean runs without a device->host sync; the original boolean index syncs """
    tgt, pred = torch.randn(4, 16, 8, device="cuda"), torch.randn(4, 16, 8, device="cuda")
    s = torch.ones(4, dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        prediction_loss(tgt, pred, s=s, sync_free=True)
        with pytest.raises(RuntimeError):
            prediction_loss(tgt, pred, s=s)
    finally:
        torch.cuda.set_sync_debug_mode("default")

# ---------------------------------------------------------------- uint8 images

def _fake_imagefolder(root, n=6):
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        d = os.path.join(root, split, "n00000000")
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            h, w = int(rng.integers(150, 300)), int(rng.integers(150, 300))
            Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(os.path.join(d, f"{i}.JPEG"), quality=90)
    return root

@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
@pytest.mark.parametrize("split", ["train", "val"])
def test_uint8_images_bit_identical(tmp_path, device, split):
    """ same seeded crop; GPU-side x/255 then Normalize gives bit-identical inputs to ToTensor + Normalize """
    root = _fake_imagefolder(str(tmp_path))
    ref_ds = PretrainDataset(root, split, image_size=64, patch_size=16, base_seed=1)
    new_ds = PretrainDataset(root, split, image_size=64, patch_size=16, base_seed=1, uint8_images=True)
    lit = LitREPA(make_config("tiny", "uint8_images")).to(device)
    for i in range(len(ref_ds)):
        ref, new = ref_ds[i], new_ds[i]
        assert new["img"].dtype == torch.uint8 and torch.equal(ref["position_ids"], new["position_ids"])
        out = lit.on_after_batch_transfer({"img": new["img"][None].to(device)}, 0)["img"][0].cpu()
        assert torch.equal(out, ref["img"]), f"sample {i}: max abs diff {(out - ref['img']).abs().max().item():.3e}"
    DIFFS.append((f"uint8 images [{split}, {device}]", "image", 0.0, "exact"))

def test_uint8_transform_flag_off_unchanged():
    t = build_pretrain_transform(64, train=False)
    assert [type(x).__name__ for x in t.transforms] == ["Resize", "CenterCrop", "ToTensor", "Normalize"]

# ---------------------------------------------------------------- optimizer and DDP

def _optimizer_step(perf, device):
    torch.manual_seed(0)
    lit = LitREPA(make_config("tiny", perf)).to(device)
    for p in lit.parameters():
        p.grad = torch.randn_like(p) * 1e-2
    opt = lit._create_optimizer()
    opt.step()
    return lit, opt

@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
def test_fused_adamw(device):
    ref, opt_ref = _optimizer_step("", device)
    new, opt_new = _optimizer_step("fused_adamw", device)
    assert opt_new.defaults["fused"] is True and not opt_ref.defaults.get("fused")
    assert [len(g["params"]) for g in opt_ref.param_groups] == [len(g["params"]) for g in opt_new.param_groups]
    worst = 0.0
    for (n, a), (_, b) in zip(new.named_parameters(), ref.named_parameters()):
        d, r = max_rel(a.detach(), b.detach())
        worst = max(worst, d)
        assert r <= RTOL, f"{n}: {d:.3e}"
    DIFFS.append((f"fused AdamW one step [{device}]", "params", worst, f"rtol {RTOL:g}"))

def test_ddp_static_graph_strategy():
    from pretrain import build_strategy
    from pytorch_lightning.strategies import DDPStrategy
    off = make_config("tiny")
    on = make_config("tiny", "ddp_static_graph")
    on.world_size = off.world_size = 1
    assert build_strategy(off) == "auto" and build_strategy(on) == "auto" # no-op on one GPU
    on.world_size = 2
    strat = build_strategy(on)
    assert isinstance(strat, DDPStrategy) and strat._ddp_kwargs.get("static_graph") is True
    off.world_size = 2
    assert build_strategy(off) == "auto"

def test_perf_flag_parsing():
    assert parse_perf_flags("") == {}
    assert parse_perf_flags("all") == {f"perf_{n}": True for n in PERF_FLAGS}
    assert parse_perf_flags("causal_no_mask, fused_adamw") == {"perf_causal_no_mask": True, "perf_fused_adamw": True}
    with pytest.raises(ValueError):
        parse_perf_flags("not_a_flag")

def test_old_config_without_perf_fields():
    """ configs pickled into older checkpoints have no perf_* attributes: everything falls back to off """
    cfg = make_config("tiny")
    for n in PERF_FLAGS:
        delattr(cfg, f"perf_{n}")
    model = REPA(cfg)
    lit = LitREPA(cfg)
    assert not (model.perf_skip_last_content or model.perf_causal_no_mask or model.perf_query_proj_once)
    assert not (lit.perf_no_cpu_sync or lit.perf_fused_adamw)

# ---------------------------------------------------------------- end to end through the Lightning Trainer

@needs_cuda
def test_lightning_all_flags_matches_baseline(tmp_path):
    """ a few real training steps (fake JPEGs) with every flag vs none: same logged loss and lr, same weights """
    import pytorch_lightning as pl
    from dataset import ImageNetDataModule

    root = _fake_imagefolder(str(tmp_path), n=16)

    def fit(perf, precision="32-true"):
        cfg = make_config("tiny", perf, batch_size=4, global_batch_size=4 * torch.cuda.device_count())
        torch.manual_seed(0)
        lit = LitREPA(cfg)
        flags = dict(image_size=64, patch_size=16, base_seed=1, uint8_images=cfg.perf_uint8_images)
        dm = ImageNetDataModule(PretrainDataset(root, "train", **flags), PretrainDataset(root, "val", **flags),
                                seed=1, batch_size=4, num_workers=0, pin_memory=cfg.perf_uint8_images)
        trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1, precision=precision, logger=False,
                             gradient_clip_val=cfg.grad_clip, # as pretrain.py
                             enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0,
                             enable_model_summary=False, use_distributed_sampler=False)
        trainer.fit(lit, datamodule=dm)
        return trainer.callback_metrics, {n: p.detach().cpu() for n, p in lit.named_parameters()}

    m_ref, p_ref = fit("")
    m_new, p_new = fit("all")
    for key in ("train/loss", "train/lr"):
        d, r = max_rel(m_new[key].cpu(), m_ref[key].cpu())
        DIFFS.append(("lightning 1 epoch [all]", key, d, "rtol 1e-4"))
        assert r <= 1e-4, key
    worst = max(max_rel(p_new[n], p_ref[n])[1] for n in p_ref)
    DIFFS.append(("lightning 1 epoch [all]", "weights (max rel)", worst, "rtol 1e-4"))
    assert worst <= 1e-4

    # bf16-mixed with gradient clipping: the path where Lightning rejects fused optimizers (see LitREPA)
    m_ref, p_ref = fit("", "bf16-mixed")
    m_new, p_new = fit("fused_adamw", "bf16-mixed")
    d, _ = max_rel(m_new["train/loss"].cpu(), m_ref["train/loss"].cpu())
    worst = max(max_rel(p_new[n], p_ref[n])[1] for n in p_ref)
    DIFFS.append(("lightning 1 epoch bf16 [fused_adamw]", "train/loss", d, "rtol 1e-4"))
    DIFFS.append(("lightning 1 epoch bf16 [fused_adamw]", "weights (max rel)", worst, "rtol 1e-4"))
    assert worst <= 1e-4

@needs_cuda
def test_fused_adamw_clipping_with_accumulation(tmp_path, monkeypatch):
    """ bf16-mixed, accumulate_grad_batches=4, 8 batches: clipping must run once per optimizer step (after batch 3
    and 7, i.e. after accumulation), on unscaled gradients, and fused / non-fused must end with the same weights """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    from dataset import ImageNetDataModule

    root = _fake_imagefolder(str(tmp_path), n=32) # 32 images / batch 4 = 8 micro-batches = 2 optimizer steps
    clip_val, accum = 0.05, 4 # small, so clipping really scales the gradients
    real_clip = torch.nn.utils.clip_grad_norm_
    calls, state = [], {"batch": None}

    def spy(parameters, max_norm, *args, **kwargs):
        params = [p for p in parameters if p.grad is not None]
        pre = real_clip(params, max_norm, *args, **kwargs) # returns the norm before clipping
        post = torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm() for p in params]))
        calls.append({"batch_idx": state["batch"], "pre": pre.item(), "post": post.item()})
        return pre
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

    class Track(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            state["batch"] = batch_idx

    def fit(perf):
        calls.clear()
        cfg = make_config("tiny", perf, batch_size=4, global_batch_size=4 * accum * torch.cuda.device_count(),
                          grad_clip=clip_val)
        assert cfg.grad_accum_steps == accum
        torch.manual_seed(0)
        lit = LitREPA(cfg)
        kw = dict(image_size=64, patch_size=16, base_seed=1)
        dm = ImageNetDataModule(PretrainDataset(root, "train", **kw), PretrainDataset(root, "val", **kw),
                                seed=1, batch_size=4, num_workers=0)
        trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1, precision="bf16-mixed", logger=False,
                             gradient_clip_val=cfg.grad_clip, accumulate_grad_batches=cfg.grad_accum_steps,
                             enable_checkpointing=False, enable_progress_bar=False, limit_val_batches=0,
                             enable_model_summary=False, use_distributed_sampler=False, callbacks=[Track()])
        trainer.fit(lit, datamodule=dm)
        weights = {n: p.detach().cpu() for n, p in lit.named_parameters()}
        return list(calls), trainer.precision_plugin.scaler, trainer.global_step, weights

    c_ref, scaler_ref, steps_ref, w_ref = fit("")
    c_new, scaler_new, steps_new, w_new = fit("fused_adamw")
    print(f"\nnon-fused clip calls: {c_ref}\nfused clip calls:     {c_new}\n"
          f"scaler (non-fused, fused): {scaler_ref}, {scaler_new}; optimizer steps: {steps_ref}, {steps_new}")

    assert scaler_ref is None and scaler_new is None # bf16-mixed has no GradScaler: gradients are never scaled
    assert steps_ref == steps_new == 2
    for calls_ in (c_ref, c_new):
        assert [c["batch_idx"] for c in calls_] == [accum - 1, 2 * accum - 1] # once per step, after the 4th backward
        for c in calls_:
            assert c["pre"] > clip_val and abs(c["post"] - clip_val) <= 1e-3 * clip_val # clipping took effect
    pre_diff = max(abs(a["pre"] - b["pre"]) / b["pre"] for a, b in zip(c_new, c_ref))
    worst = max(max_rel(w_new[n], w_ref[n])[1] for n in w_ref)
    DIFFS.append(("fused AdamW + clip, accum 4, bf16", "pre-clip norm (rel)", pre_diff, "rtol 1e-4"))
    DIFFS.append(("fused AdamW + clip, accum 4, bf16", "weights (max rel)", worst, "rtol 1e-4"))
    assert pre_diff <= 1e-4 and worst <= 1e-4
