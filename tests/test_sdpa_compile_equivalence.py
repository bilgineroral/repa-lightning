"""
Equivalence tests the RTX 3060 can run for two Tier 3 runner settings (bench/KERNEL_PLAN.md ranks 1 and 5):

SDPA backends (bench/kern_step.py --sdpa cudnn / efficient): forward and backward of our attention calls (REPA-S
heads, N = 196 causal, and the 197 x 197 masked call of the original path) against the math backend.
  - fp32: rtol 1e-5 against math in fp32. cuDNN has no fp32 kernel (skipped where a backend isn't available).
  - bf16: a bf16 result can't match a reference to 1e-4 (one bf16 ulp is ~4e-3 relative: every backend, math too,
    measures ~5e-3 against fp64 on the 3060). Criterion as tests/test_perf_equivalence.py's bf16 tests: the worst error
    against an fp64 reference is at most 1.5x the math backend's own bf16 error.

torch.compile mode="reduce-overhead" (CUDA graphs) with gradient accumulation, through the Lightning Trainer as the
runner uses it, with the same grad-buffer fix bench/kern_step.py applies (stable_grad_buffers): accumulated gradients
at each optimizer step and the losses match eager, rtol 1e-4 (compiled code reorders float ops, as
test_fp32_cuda_compiled_all). Inductor draws random numbers with its own generator, so the RoPE rescale factor inside
the compiled model differs from eager's (default compile measured ~100% gradient differences against eager on the
3060, with or without CUDA graphs); the test sets torch._inductor.config.fallback_random so compiled code uses eager's
RNG. FP8 and max-autotune need sm_89 / >= 68 SMs and are not tested here.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from conftest import DIFFS # noqa: E402
from test_perf_equivalence import make_config, max_rel # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
BACKENDS = {"cudnn": SDPBackend.CUDNN_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION}
CASES = {"causal 196": ((4, 6, 196, 196, 64), "causal"), "mask 197": ((4, 6, 197, 197, 64), "mask")}

def _inputs(shape, seed=0):
    B, H, Nq, Nk, d = shape
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, Nq, d, device="cuda", dtype=torch.float64, generator=g)
    k, v = (torch.randn(B, H, Nk, d, device="cuda", dtype=torch.float64, generator=g) for _ in range(2))
    grad = torch.randn(B, H, Nq, d, device="cuda", dtype=torch.float32, generator=g)
    return (q, k, v), grad

def _attention(backend, dtype, qkv, grad, kind):
    """ output and (dq, dk, dv) in float32, for one backend at one dtype """
    q, k, v = (t.detach().to(dtype).requires_grad_() for t in qkv)
    mask = None
    if kind == "mask": # the original path's dense boolean mask (modules.py _make_attn_masks)
        mask = torch.tril(torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device="cuda"))[None, None]
    with sdpa_kernel([backend]):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=kind == "causal")
    grads = torch.autograd.grad((out.float() * grad).sum(), (q, k, v))
    return [out.float()] + [g.float() for g in grads]

def _run_or_skip(name, dtype, qkv, grad, kind):
    try:
        return _attention(BACKENDS[name], dtype, qkv, grad, kind)
    except RuntimeError as e: # "No available kernel": this backend doesn't take these inputs here
        pytest.skip(f"{name} SDPA unavailable for {dtype} {kind}: {str(e).splitlines()[0][:80]}")

NAMES = ("out", "dq", "dk", "dv")

@needs_cuda
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", BACKENDS)
def test_sdpa_backend_fp32(name, case):
    torch.backends.cuda.matmul.allow_tf32 = False
    shape, kind = CASES[case]
    qkv, grad = _inputs(shape)
    new = _run_or_skip(name, torch.float32, qkv, grad, kind)
    ref = _attention(SDPBackend.MATH, torch.float32, qkv, grad, kind)
    for n, a, b in zip(NAMES, new, ref):
        d, r = max_rel(a, b)
        DIFFS.append((f"sdpa {name} fp32 [{case}]", f"{n} vs math", d, "rtol 1e-5"))
        assert r <= 1e-5, f"{name} fp32 {case} {n}: relative {r:.3e} > 1e-5"

@needs_cuda
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", BACKENDS)
def test_sdpa_backend_bf16(name, case):
    shape, kind = CASES[case]
    qkv, grad = _inputs(shape)
    new = _run_or_skip(name, torch.bfloat16, qkv, grad, kind)
    math = _attention(SDPBackend.MATH, torch.bfloat16, qkv, grad, kind)
    exact = [t.float() for t in _attention(SDPBackend.MATH, torch.float64, qkv, grad, kind)]
    err_new = max(max_rel(a, b)[1] for a, b in zip(new, exact))
    err_math = max(max_rel(a, b)[1] for a, b in zip(math, exact))
    DIFFS.append((f"sdpa {name} bf16 [{case}]", "err vs fp64", err_new, f"<=1.5x {err_math:.1e}"))
    assert err_new <= 1.5 * err_math + 1e-6, f"{name} bf16 {case}: error {err_new:.3e} vs math's own {err_math:.3e}"

# ---------------------------------------------------------------- reduce-overhead + gradient accumulation

ACCUM, BATCHES, BATCH = 2, 4, 2 # 4 micro-batches = 2 optimizer steps of 2 accumulated micro-steps each

def _fit(mode, monkeypatch):
    """ LitREPA through the Lightning Trainer (as the runner), lr 0 so the weights stay fixed; returns the losses and
    the accumulated gradients just before each optimizer step """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    import kern_step
    from harness import StopAfter, _SyntheticDataModule
    from repa import LitREPA

    if mode != "eager":
        # exactly what bench/kern_step.py does for this mode; monkeypatch undoes the global changes afterwards
        monkeypatch.setattr(torch, "compile", torch.compile)
        monkeypatch.setattr(LitREPA, "on_train_start", LitREPA.on_train_start)
        monkeypatch.setattr(LitREPA, "optimizer_zero_grad", LitREPA.optimizer_zero_grad)
        # eager's RNG inside compiled code, so the RoPE rescale draw matches (see the module docstring)
        monkeypatch.setattr(torch._inductor.config, "fallback_random", True)
        kern_step.apply_compile_mode(torch, mode)
        assert mode in kern_step.CUDAGRAPH_MODES
        kern_step.stable_grad_buffers(torch)

    class Grads(Callback):
        def __init__(self):
            self.steps = []

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            self.steps.append({n.replace("_orig_mod.", ""): p.grad.detach().float().cpu().clone() # compiled names
                               for n, p in pl_module.named_parameters() if p.grad is not None})

    cfg = make_config("tiny", "all", batch_size=BATCH, global_batch_size=BATCH * ACCUM * torch.cuda.device_count(),
                      blr=0.0, grad_clip=0.0)
    cfg.compile = mode != "eager" # make_config builds uncompiled models; LitREPA compiles when this is set
    torch.manual_seed(0)
    lit = LitREPA(cfg)
    grads, stop = Grads(), StopAfter(BATCHES)
    trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1, precision="32-true", logger=False,
                         accumulate_grad_batches=cfg.grad_accum_steps, limit_val_batches=0, num_sanity_val_steps=0,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         use_distributed_sampler=False, callbacks=[grads, stop])
    trainer.fit(lit, datamodule=_SyntheticDataModule(cfg, BATCHES))
    return [l.item() for l in stop.losses], grads.steps

@needs_cuda
def test_reduce_overhead_grad_accumulation_matches_eager(monkeypatch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.reset()
    from torch._dynamo.utils import counters
    losses_ref, steps_ref = _fit("eager", monkeypatch)
    counters["inductor"]["cudagraph_skips"] = 0
    losses_new, steps_new = _fit("reduce-overhead", monkeypatch)
    skips = counters["inductor"]["cudagraph_skips"]
    torch._dynamo.reset()
    assert skips == 0, f"CUDA graphs were skipped {skips} time(s): the test would not exercise them"
    assert len(steps_ref) == len(steps_new) == BATCHES // ACCUM
    for a, b in zip(losses_new, losses_ref):
        assert abs(a - b) <= 1e-4 * abs(b), f"loss {a} vs eager {b}"
    for i, (new, ref) in enumerate(zip(steps_new, steps_ref)):
        worst = 0.0
        for n in ref:
            d, r = max_rel(new[n], ref[n])
            worst = max(worst, r)
            assert r <= 1e-4, f"optimizer step {i}, {n}: relative {r:.3e} > 1e-4"
        DIFFS.append((f"reduce-overhead accum {ACCUM}, step {i}", "grad vs eager (rel)", worst, "rtol 1e-4"))
