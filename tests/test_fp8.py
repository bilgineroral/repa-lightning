"""
perf_fp8 setup (repa.py convert_to_fp8 / fp8_module_filter, config.py batch check) and FP8 with all Tier 1 flags.

FP8's grad_weight matmul contracts over the rows of each Linear input (torchao float8_linear.py:
mm(grad_output.t(), input)), and torch._scaled_mm needs that dimension divisible by 16. Rows are batch x tokens
(196 or 197), so the per-GPU batch must be a multiple of 16; and perf_query_proj_once's layer-1 Q projection runs on
one row (the shared query token w), so that q_proj stays bf16. Before this, every runner FP8 step (--flags all,fp8)
failed in that backward.

The CPU tests check the setup without FP8 matmuls; the GPU test needs compute capability >= 8.9.
"""
import math

import pytest
import torch

from conftest import DIFFS
from config import PretrainConfig, PretrainConfigSmall, parse_perf_flags
from modules import REPA
from repa import LitREPA, convert_to_fp8, fp8_supported, prediction_loss
from test_perf_equivalence import make_batch, make_config

needs_fp8 = pytest.mark.skipif(not fp8_supported(), reason="FP8 needs a GPU with compute capability >= 8.9")

# ---------------------------------------------------------------- CPU: batch check

def test_fp8_batch_must_be_multiple_of_16():
    with pytest.raises(ValueError, match=r"perf_fp8 needs per-GPU batch size divisible by 16 \(torch._scaled_mm\)"):
        make_config("small", "all,fp8", batch_size=8)
    make_config("small", "all,fp8", batch_size=16) # fine
    make_config("small", "all", batch_size=8) # the check is only for perf_fp8

# ---------------------------------------------------------------- CPU: which Linear layers stay bf16

def _converted(perf, Config=PretrainConfigSmall, **kw):
    from torchao.float8.float8_linear import Float8Linear
    cfg = Config(compile=False, wandb=False, **parse_perf_flags(perf), **kw)
    model = REPA(cfg)
    linear = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    not16 = {n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and (m.in_features % 16 or m.out_features % 16)}
    model = convert_to_fp8(model, cfg)
    fp8 = {n for n, m in model.named_modules() if isinstance(m, Float8Linear)}
    return linear, fp8, not16

@pytest.mark.parametrize("Config", [PretrainConfigSmall, PretrainConfig], ids=["S", "B"])
def test_fp8_filter_all_flags(Config, capsys):
    """ all,fp8 (query_proj_once on): exactly layer 1's q_proj stays bf16, every other Linear becomes Float8Linear """
    linear, fp8, not16 = _converted("all,fp8", Config)
    assert not16 == set() # the %16 rule skips nothing in REPA-S / -B
    assert linear - fp8 == {"encoder.layers.0.attention.q_proj"}
    assert "perf_fp8: Linear layers left in bf16: encoder.layers.0.attention.q_proj" in capsys.readouterr().out

def test_fp8_filter_without_query_proj_once():
    linear, fp8, _ = _converted("fp8")
    assert fp8 == linear # nothing runs on one row: everything converts

def test_fp8_filter_follows_query_start():
    """ arch_query_top_k moves the query stream's first layer, and with it the one-row q_proj """
    linear, fp8, _ = _converted("all,fp8", arch_query_top_k=4)
    assert linear - fp8 == {"encoder.layers.8.attention.q_proj"} # 12 layers, the query stream starts at layer 8

# ---------------------------------------------------------------- GPU: all Tier 1 flags + FP8

@needs_fp8
def test_fp8_all_flags_loss_tracks_bf16():
    """ the runner's FP8 configuration (--flags all,fp8, query_proj_once on), forward and backward: 20 AdamW steps on
    the same batches (REPA-S shapes, batch 16), per-step losses within 5% of all flags in bf16 """
    from torchao.float8.float8_linear import Float8Linear
    losses = {}
    for perf in ("all", "all,fp8"):
        torch.manual_seed(0)
        cfg = make_config("small", perf)
        lit = LitREPA(cfg).cuda().train()
        if perf == "all,fp8":
            kinds = {n: type(m) for n, m in lit.model.named_modules()}
            assert kinds["encoder.layers.0.attention.q_proj"] is torch.nn.Linear
            assert kinds["encoder.layers.1.attention.q_proj"] is Float8Linear
        opt = torch.optim.AdamW(lit.parameters(), lr=1e-4)
        out = []
        for step in range(20):
            img, pos, s = make_batch(cfg, 16, "cuda", seed=step)
            torch.manual_seed(1000 + step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tgt, pred = lit.model(img, position_ids=pos)
                loss = prediction_loss(tgt, pred, s=s, sync_free=cfg.perf_no_cpu_sync)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            out.append(loss.item())
        losses[perf] = out
    rel = [abs(a - b) / abs(b) for a, b in zip(losses["all,fp8"], losses["all"])]
    DIFFS.append(("fp8 vs bf16, all flags, 20 AdamW steps, S", "loss (max rel)", max(rel), "rtol 5e-2"))
    assert all(math.isfinite(x) for x in losses["all,fp8"]) and max(rel) <= 5e-2
