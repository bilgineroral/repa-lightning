"""
Sanity tests for the arch_* flags (config.py). These flags CHANGE the model, so there is no equivalence test: each
setting must build, run forward / backward with finite loss, give every parameter a gradient (DDP), and lower the loss
over 50 steps on one fixed tiny batch. Plus option B's tests from ~/repa-notes/option_b_design.md (Implementation
plan, step 8): flags off unchanged, no target leakage, B with zero e_tgt in raster order == the NEPA branch, old
configs load.

    pytest tests/test_arch_sanity.py -v          # CPU only, about a minute
"""
import math

import pytest
import torch

from conftest import DIFFS
from config import ARCH_FLAGS, PretrainConfig, parse_arch_flags
from modules import REPA
from repa import LitREPA, prediction_loss

TINY = dict(image_size=64, patch_size=16, hidden_size=64, num_attention_heads=4, num_hidden_layers=3,
            intermediate_size=128, compile=False, wandb=False)
ARCH_SPECS = [
    "single_stream_ref", "query_top_k=2", "query_top_k=1", "single_stream_target_pos",
    "single_stream_target_pos,target_pos_type=sincos", "single_stream_target_pos,target_pos_layers=2",
    "single_stream_target_pos,target_pos_null_p=0.2", "query_mlp_width=2d", "query_mlp_width=d", "train_res=32",
    "predict_frac=0.5", "predict_frac=0.25",
]
PERF_ALL = dict(perf_skip_last_content=True, perf_causal_no_mask=True, perf_query_proj_once=True, perf_no_cpu_sync=True)

def tiny_config(arch: str = "", perf: bool = False, **kw) -> PretrainConfig:
    flags = parse_arch_flags(arch, TINY["hidden_size"])
    if flags.get("arch_single_stream_ref") or flags.get("arch_single_stream_target_pos"):
        flags["use_query_stream"] = False
    return PretrainConfig(**TINY, **flags, **(PERF_ALL if perf else {}), **kw)

def tiny_batch(cfg, batch=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = (cfg.image_size // cfg.patch_size) ** 2
    img = torch.randn(batch, cfg.num_channels, cfg.image_size, cfg.image_size, generator=g)
    pos = torch.stack([torch.randperm(N, generator=g) for _ in range(batch)])
    return {"img": img, "position_ids": pos, "s": torch.ones(batch, dtype=torch.long)}

@pytest.mark.parametrize("perf", [False, True], ids=["perf-none", "perf-all"])
@pytest.mark.parametrize("arch", ARCH_SPECS)
def test_arch_sanity(arch, perf):
    torch.manual_seed(0)
    cfg = tiny_config(arch, perf)
    lit = LitREPA(cfg).train()
    batch = tiny_batch(cfg)
    opt = torch.optim.AdamW(lit.parameters(), lr=1e-3)
    losses = []
    for step in range(50):
        torch.manual_seed(step) # RoPE rescale and null rows draw from the global RNG
        tgt, pred, s, weight = lit._predict(batch, train=True)
        loss = prediction_loss(tgt, pred, s=s, sync_free=cfg.perf_no_cpu_sync, weight=weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0: # every parameter is in the graph (zero gradient is fine), so DDP needs no find_unused_parameters
            missing = [n for n, p in lit.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing, f"parameters outside the graph: {missing}"
        opt.step()
        losses.append(loss.item())
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    DIFFS.append((f"arch [{arch}] {'perf all' if perf else 'perf none'}", "loss first5 -> last5", first - last, "must be > 0"))
    assert all(math.isfinite(x) for x in losses), losses
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"

def test_arch_shapes():
    """ pred rows: N for the single-stream modes, N - offset for predict_frac; train_res changes N """
    for arch, rows in (("single_stream_target_pos", 16), ("predict_frac=0.25", 4), ("predict_frac=0.5", 8), ("train_res=32", 4)):
        cfg = tiny_config(arch)
        b = tiny_batch(cfg, batch=2)
        tgt, pred = REPA(cfg)(b["img"], position_ids=b["position_ids"])
        assert tgt.shape == pred.shape == (2, rows, cfg.hidden_size), (arch, tgt.shape, pred.shape)

def test_arch_flag_parsing_and_validation():
    assert parse_arch_flags("query_top_k=4,query_mlp_width=2d", 384) == {"arch_query_top_k": 4, "arch_query_mlp_width": 768}
    assert parse_arch_flags("none", 384) == {}
    with pytest.raises(ValueError):
        parse_arch_flags("not_a_flag", 384)
    with pytest.raises(ValueError, match="use_query_stream=False"):
        PretrainConfig(**TINY, arch_single_stream_target_pos=True) # B without turning the query stream off
    with pytest.raises(ValueError):
        PretrainConfig(**TINY, arch_query_top_k=4) # more than num_hidden_layers (3)
    with pytest.raises(ValueError):
        PretrainConfig(**TINY, use_rope=False, arch_train_res=32)

# ---------------------------------------------------------------- option B (design doc, step 8)

def test_flags_off_unchanged():
    """ with the arch fields absent (an old pickled config) or off: same parameter names and outputs """
    cfg_new = tiny_config()
    cfg_old = tiny_config()
    for name in ARCH_FLAGS:
        delattr(cfg_old, f"arch_{name}")
    torch.manual_seed(0)
    a = REPA(cfg_new)
    torch.manual_seed(0)
    b = REPA(cfg_old)
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    batch = tiny_batch(cfg_new)
    torch.manual_seed(1)
    out_a = a(batch["img"], position_ids=batch["position_ids"])[1]
    torch.manual_seed(1)
    out_b = b(batch["img"], position_ids=batch["position_ids"])[1]
    assert torch.equal(out_a, out_b)
    LitREPA(cfg_old) # loads without the new fields

@pytest.mark.parametrize("arch", ["single_stream_target_pos", "single_stream_target_pos,target_pos_layers=2",
                                  "single_stream_ref"])
def test_option_b_no_leakage(arch):
    """ row j's prediction depends on exactly patches pi_0..pi_(j-1) of the image (never on pi_j or later) """
    torch.manual_seed(0)
    cfg = tiny_config(arch)
    model = REPA(cfg).eval()
    G, P = cfg.image_size // cfg.patch_size, cfg.patch_size
    img = torch.randn(1, 3, cfg.image_size, cfg.image_size, requires_grad=True)
    pos = torch.randperm(G * G)[None]
    _, pred = model(img, position_ids=pos)
    r = torch.randn(cfg.hidden_size) # pred is LayerNorm output: its plain sum is constant, so project it instead
    for j in range(G * G):
        (grad,) = torch.autograd.grad((pred[0, j] * r).sum(), img, retain_graph=True)
        per_patch = grad[0].abs().sum(0).reshape(G, P, G, P).sum((1, 3)).flatten() # [N]
        seen = set(pos[0, :j].tolist())
        for patch in range(G * G):
            assert (per_patch[patch] > 0) == (patch in seen), (j, patch, per_patch[patch].item())

def test_option_b_zero_target_pos_matches_nepa():
    """ e_tgt = 0 and raster order: option B (is_causal, last row dropped) == the NEPA branch (explicit mask) """
    torch.manual_seed(0)
    cfg_b = tiny_config("single_stream_target_pos")
    cfg_n = PretrainConfig(**TINY, use_permutation=False, use_query_stream=False)
    b, n = REPA(cfg_b).eval(), REPA(cfg_n).eval()
    missing, unexpected = b.load_state_dict(n.state_dict(), strict=False)
    assert missing == ["tgt_pos_embed"] and not unexpected
    with torch.no_grad():
        b.tgt_pos_embed.zero_()
    batch = tiny_batch(cfg_b)
    raster = torch.arange(16)[None].expand(4, -1)
    with torch.no_grad():
        tgt_b, pred_b = b(batch["img"], position_ids=raster)
        tgt_n, pred_n = n(batch["img"], position_ids=raster)
    diff = (pred_b - pred_n).abs().max().item()
    DIFFS.append(("option B (e_tgt=0, raster) vs NEPA branch", "pred", diff, "atol 1e-5"))
    assert torch.equal(tgt_b, tgt_n) and diff <= 1e-5

def test_option_b_target_pos_params_in_embed_group():
    """ tgt_pos_embed / tgt_null get embed_lr and weight decay, like pos_embed (design doc, step 4) """
    lit = LitREPA(tiny_config("single_stream_target_pos,target_pos_null_p=0.2"))
    opt = lit._create_optimizer()
    ids = {id(lit.model.tgt_pos_embed): "tgt_pos_embed", id(lit.model.tgt_null): "tgt_null"}
    for g in opt.param_groups:
        for p in g["params"]:
            if id(p) in ids:
                assert g["lr"] == lit.hparams.config.embed_lr and g["weight_decay"] == lit.hparams.config.weight_decay, ids[id(p)]
