"""
LayerScale = 1 variants of the perf_* equivalence tests (tests/test_perf_equivalence.py).

The configs start LayerScale at 1e-5 (layerscale_value), so every attention and MLP output reaches the residual stream
scaled by 1e-5: a difference in the attention path shows up in the loss 1e5 times smaller, below fp32 resolution
(tests/test_kern_equivalence.py saw exactly 0.0 loss differences for a different attention computation). With
LayerScale 1 the attention path reaches the loss at full weight. Same tolerances as the originals: fp32 rtol 1e-5
(1e-4 compiled), bf16 within 1.5x of the original's own bf16 error, FP8 within 5% of bf16.

    pytest tests/test_perf_equivalence_ls1.py -v          # CPU tests always run; GPU tests skip without CUDA
"""
import math

import pytest
import torch

from conftest import DIFFS
from modules import REPA
from repa import LitREPA, fp8_supported, prediction_loss
from test_perf_equivalence import FORWARD_FLAGS, RTOL, loss_and_grads, make_batch, make_config, max_rel

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
needs_fp8 = pytest.mark.skipif(not fp8_supported(), reason="FP8 needs a GPU with compute capability >= 8.9")
LS = dict(layerscale_value=1.0)

def model_pair(size, perf, device, compile_models=False):
    torch.manual_seed(0)
    cfg_ref, cfg_new = make_config(size, **LS), make_config(size, perf, **LS)
    ref = REPA(cfg_ref).to(device).train()
    new = REPA(cfg_new).to(device).train()
    new.load_state_dict(ref.state_dict())
    if compile_models:
        ref, new = torch.compile(ref), torch.compile(new)
    return (cfg_ref, ref), (cfg_new, new)

def check_fp32(test_id, size, perf, device, batch=4, compile_models=False, rtol=RTOL):
    (cfg_ref, ref), (cfg_new, new) = model_pair(size, perf, device, compile_models)
    img, pos, s = make_batch(cfg_ref, batch, device)
    loss_ref, g_ref = loss_and_grads(ref, cfg_ref, img, pos, s)
    loss_new, g_new = loss_and_grads(new, cfg_new, img, pos, s)
    d_loss, r_loss = max_rel(loss_new, loss_ref)
    DIFFS.append((test_id, "loss", d_loss, f"rtol {rtol:g}"))
    failures, worst_diff, worst_rel = [], 0.0, 0.0
    for name in g_ref:
        d, r = max_rel(g_new[name], g_ref[name])
        worst_diff, worst_rel = max(worst_diff, d), max(worst_rel, r)
        if r > rtol:
            failures.append(f"{name}: abs {d:.3e}, rel {r:.3e}")
    DIFFS.append((test_id, f"grad (worst rel {worst_rel:.1e})", worst_diff, f"rtol {rtol:g}"))
    assert r_loss <= rtol, f"loss: {loss_new.item()} vs {loss_ref.item()} (rel {r_loss:.3e})"
    assert not failures, f"{len(failures)} gradient(s) over rtol {rtol}: " + "; ".join(failures[:5])

@pytest.mark.parametrize("perf", FORWARD_FLAGS)
def test_fp32_cpu_tiny_ls1(perf):
    check_fp32(f"fp32 cpu tiny LS1 [{perf}]", "tiny", perf, "cpu")

@needs_cuda
@pytest.mark.parametrize("perf", FORWARD_FLAGS)
def test_fp32_cuda_small_ls1(perf):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check_fp32(f"fp32 cuda small LS1 [{perf}]", "small", perf, "cuda", batch=4)

@needs_cuda
@pytest.mark.parametrize("perf", FORWARD_FLAGS)
def test_bf16_cuda_small_ls1(perf):
    torch.backends.cuda.matmul.allow_tf32 = False
    (cfg_ref, ref), (cfg_new, new) = model_pair("small", perf, "cuda")
    img, pos, s = make_batch(cfg_ref, 4, "cuda")
    loss_32, g_32 = loss_and_grads(ref, cfg_ref, img, pos, s)
    loss_ref, g_ref = loss_and_grads(ref, cfg_ref, img, pos, s, torch.bfloat16)
    loss_new, g_new = loss_and_grads(new, cfg_new, img, pos, s, torch.bfloat16)
    err_ref = max(max_rel(g_ref[n], g_32[n])[1] for n in g_32)
    err_new = max(max_rel(g_new[n], g_32[n])[1] for n in g_32)
    DIFFS.append((f"bf16 cuda small LS1 [{perf}]", "grad err vs fp32", err_new, f"<=1.5x {err_ref:.1e}"))
    assert err_new <= 1.5 * err_ref + 1e-6, f"bf16 grad error {err_new:.3e} vs original's {err_ref:.3e} (fp32 reference)"
    assert abs(loss_new.item() - loss_32.item()) <= 1.5 * abs(loss_ref.item() - loss_32.item()) + 1e-4

@needs_cuda
def test_fp32_cuda_compiled_all_ls1():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.reset()
    check_fp32("fp32 cuda tiny compiled LS1 [all]", "tiny", "all", "cuda", compile_models=True, rtol=1e-4)

@needs_fp8
def test_fp8_loss_tracks_bf16_ls1():
    """ as test_fp8_loss_tracks_bf16 with LayerScale 1: 20 AdamW steps, per-step losses within 5% of bf16 """
    losses = {}
    for perf in ("", "fp8"):
        torch.manual_seed(0)
        cfg = make_config("small", perf, **LS)
        lit = LitREPA(cfg).cuda().train()
        opt = torch.optim.AdamW(lit.parameters(), lr=1e-4)
        out = []
        for step in range(20):
            img, pos, s = make_batch(cfg, 16, "cuda", seed=step) # FP8: batch % 16 (config.py perf_fp8 check)
            torch.manual_seed(1000 + step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tgt, pred = lit.model(img, position_ids=pos)
                loss = prediction_loss(tgt, pred, s=s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            out.append(loss.item())
        losses[perf] = out
    rel = [abs(a - b) / abs(b) for a, b in zip(losses["fp8"], losses[""])]
    DIFFS.append(("fp8 vs bf16 LS1, 20 AdamW steps, S", "loss (max rel)", max(rel), "rtol 5e-2"))
    assert all(math.isfinite(x) for x in losses["fp8"]) and max(rel) <= 5e-2
