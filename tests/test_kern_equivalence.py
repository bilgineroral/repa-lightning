"""
Equivalence tests for the kern_* flags (config.py): with the flag on, the same loss and parameter gradients as the
current path, from the same weights, batch and seed (fp32, rtol 1e-5 as tests/test_perf_equivalence.py).

    pytest tests/test_kern_equivalence.py -v          # CPU tests always run; GPU tests skip without CUDA

kern_attn_fused only changes the computation on the square causal path, so it is compared with perf_causal_no_mask on
(and with all Tier 1 forward flags on). FlexAttention has no CPU backward: on CPU the flex tests check the forward only;
forward and backward run on the GPU, in eager mode and compiled (the Triton kernel that training uses).
"""
import pytest
import torch
import torch.nn.functional as F

from conftest import DIFFS
from config import parse_kern_flags
from modules import REPA, PatchEmbed, SelfAttention
from test_perf_equivalence import RTOL, loss_and_grads, make_batch, make_config, max_rel

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])

def check_close(test_id, name, new, ref, rtol=RTOL):
    d, r = max_rel(new, ref)
    DIFFS.append((test_id, name, d, f"rtol {rtol:g}"))
    assert r <= rtol, f"{test_id} {name}: max abs diff {d:.3e}, relative {r:.3e} > {rtol}"

def model_pair(size, perf, kern, device, **kw):
    torch.manual_seed(0)
    cfg_ref, cfg_new = make_config(size, perf, **kw), make_config(size, perf, **parse_kern_flags(kern), **kw)
    ref = REPA(cfg_ref).to(device).train()
    new = REPA(cfg_new).to(device).train()
    new.load_state_dict(ref.state_dict()) # the kern flags add no parameters
    return (cfg_ref, ref), (cfg_new, new)

def check_model(test_id, size, perf, kern, device, backward=True, batch=4, compile_models=False, rtol=RTOL, **kw):
    (cfg_ref, ref), (cfg_new, new) = model_pair(size, perf, kern, device, **kw)
    if compile_models:
        ref, new = torch.compile(ref), torch.compile(new)
    img, pos, s = make_batch(cfg_ref, batch, device)
    if not backward:
        with torch.no_grad():
            torch.manual_seed(1234)
            t_ref, p_ref = ref(img, position_ids=pos)
            torch.manual_seed(1234)
            t_new, p_new = new(img, position_ids=pos)
        check_close(test_id, "pred (fwd only)", p_new, p_ref, rtol)
        check_close(test_id, "tgt (fwd only)", t_new, t_ref, rtol)
        return
    loss_ref, g_ref = loss_and_grads(ref, cfg_ref, img, pos, s)
    loss_new, g_new = loss_and_grads(new, cfg_new, img, pos, s)
    check_close(test_id, "loss", loss_new, loss_ref, rtol)
    worst_diff, worst_rel = 0.0, 0.0
    for n in g_ref:
        d, r = max_rel(g_new[n], g_ref[n])
        assert r <= rtol, f"{test_id} grad {n}: max abs diff {d:.3e}, relative {r:.3e} > {rtol}"
        worst_diff, worst_rel = max(worst_diff, d), max(worst_rel, r)
        if g_ref[n].abs().max() == 0: # skip_last_content's unused parameters stay exactly zero
            assert g_new[n].abs().max() == 0, f"{n}: gradient should stay exactly zero"
    DIFFS.append((test_id, f"grad (worst rel {worst_rel:.1e})", worst_diff, f"rtol {rtol:g}"))

# ---------------------------------------------------------------- parsing

def test_parse_kern_flags():
    assert parse_kern_flags("") == {}
    assert parse_kern_flags("patch_linear,attn_fused=flex") == {"kern_patch_linear": True, "kern_attn_fused": "flex"}
    with pytest.raises(ValueError):
        parse_kern_flags("attn_fused=other")
    with pytest.raises(ValueError):
        parse_kern_flags("nope")
    with pytest.raises(ValueError):
        make_config("tiny", kern_attn_fused="other")

# ---------------------------------------------------------------- kern_patch_linear

@pytest.mark.parametrize("device", DEVICES)
def test_patch_linear_module(device):
    torch.manual_seed(0)
    cfg = make_config("small")
    ref = PatchEmbed(cfg).to(device)
    new = PatchEmbed(make_config("small", kern_patch_linear=True)).to(device)
    new.load_state_dict(ref.state_dict())
    img = torch.randn(3, 3, cfg.image_size, cfg.image_size, device=device)
    out_ref, out_new = ref(img), new(img)
    assert out_new.shape == out_ref.shape
    check_close(f"patch_linear module [{device}]", "output", out_new, out_ref)
    g = torch.randn_like(out_ref)
    gr = torch.autograd.grad(out_ref, list(ref.parameters()), g)
    gn = torch.autograd.grad(out_new, list(new.parameters()), g)
    for name, a, b in zip(("weight", "bias"), gn, gr):
        check_close(f"patch_linear module [{device}]", f"grad {name}", a, b)

def test_patch_linear_falls_back_on_other_sizes():
    """ image sides that aren't multiples of the patch size: the conv path (it drops the remainder) """
    torch.manual_seed(0)
    cfg = make_config("tiny")
    ref, new = PatchEmbed(cfg), PatchEmbed(make_config("tiny", kern_patch_linear=True))
    new.load_state_dict(ref.state_dict())
    img = torch.randn(2, 3, 70, 66)
    assert torch.equal(new(img), ref(img))

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("perf", ["", "all"])
def test_patch_linear_model(device, perf):
    check_model(f"patch_linear model [{perf or 'baseline'}, {device}]", "tiny", perf, "patch_linear", device)

# ---------------------------------------------------------------- kern_attn_fused: attention level

def _attention_inputs(device, B=2, N=196, D=384, seed=0, expand=False):
    torch.manual_seed(seed)
    cfg = make_config("small")
    attn = SelfAttention(cfg).to(device)
    x_h = torch.randn(B, N, D, device=device, requires_grad=True)
    x_g = torch.randn(1, 1, D, device=device, requires_grad=True) if expand else torch.randn(B, N, D, device=device, requires_grad=True)
    d = cfg.hidden_size // cfg.num_attention_heads
    angles = torch.rand(B, N, d, device=device) * 6.28
    pe = (torch.cos(angles), torch.sin(angles))
    return attn, x_h, x_g, pe

def _two_calls(attn, x_h, x_g, pe, q_expand):
    h_out, k, v = attn(x_q=x_h, x_kv=x_h, position_embeddings=pe, return_kv=True, is_causal=True)
    g_out = attn(x_q=x_g, x_kv=x_h, position_embeddings=pe, kv=(k, v), is_causal=True, q_expand=q_expand)
    return h_out, g_out

class _Fused(torch.nn.Module):
    """ fused_two_stream as a module: torch.compile of a module keeps the parameters' gradients (a compiled closure
    over the module does not) """
    def __init__(self, attn, pe, mode, q_expand):
        super().__init__()
        self.attn, self.pe, self.mode, self.q_expand = attn, pe, mode, q_expand

    def forward(self, x_h, x_g):
        return self.attn.fused_two_stream(x_h, x_g, self.pe, self.pe, self.mode, q_expand=self.q_expand)

def check_attention(test_id, mode, device, backward=True, expand=False, compile_fused=False):
    attn, x_h, x_g, pe = _attention_inputs(device, expand=expand)
    q_expand = (x_h.shape[0], x_h.shape[1]) if expand else None
    h_ref, g_ref = _two_calls(attn, x_h, x_g, pe, q_expand)
    fn = _Fused(attn, pe, mode, q_expand)
    if compile_fused: # the Triton FlexAttention kernel, as training runs it
        fn = torch.compile(fn)
    if not backward:
        with torch.no_grad():
            h_new, g_new = fn(x_h, x_g)
        check_close(test_id, "h out (fwd only)", h_new, h_ref.detach())
        check_close(test_id, "g out (fwd only)", g_new, g_ref.detach())
        return
    h_new, g_new = fn(x_h, x_g)
    check_close(test_id, "h out", h_new, h_ref)
    check_close(test_id, "g out", g_new, g_ref)
    gh, gg = torch.randn_like(h_ref), torch.randn_like(g_ref)
    inputs = [x_h, x_g] + list(attn.parameters())
    ref_grads = torch.autograd.grad((h_ref * gh).sum() + (g_ref * gg).sum(), inputs)
    new_grads = torch.autograd.grad((h_new * gh).sum() + (g_new * gg).sum(), inputs)
    names = ["x_h", "x_g"] + [n for n, _ in attn.named_parameters()]
    for name, a, b in zip(names, new_grads, ref_grads):
        check_close(test_id, f"grad {name}", a, b)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("expand", [False, True])
def test_attn_fused_gqa_attention(device, expand):
    check_attention(f"attn_fused gqa attention [{device}{', q_expand' if expand else ''}]", "gqa", device, expand=expand)

@pytest.mark.parametrize("expand", [False, True])
def test_attn_fused_flex_attention_cpu_forward(expand):
    check_attention(f"attn_fused flex attention [cpu fwd{', q_expand' if expand else ''}]", "flex", "cpu",
                    backward=False, expand=expand)

@needs_cuda
@pytest.mark.parametrize("expand", [False, True])
def test_attn_fused_flex_attention_cuda_eager(expand):
    check_attention(f"attn_fused flex attention [cuda eager{', q_expand' if expand else ''}]", "flex", "cuda",
                    expand=expand)

@needs_cuda
def test_attn_fused_flex_attention_cuda_compiled():
    """ the Triton FlexAttention kernel (forward and backward), as training runs it under torch.compile """
    torch._dynamo.reset()
    check_attention("attn_fused flex attention [cuda compiled]", "flex", "cuda", compile_fused=True)

# ---------------------------------------------------------------- kern_attn_fused: whole model

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("perf", ["causal_no_mask", "all"])
def test_attn_fused_gqa_model(device, perf):
    check_model(f"attn_fused gqa model [{perf}, {device}]", "tiny", perf, "attn_fused=gqa", device)

@pytest.mark.parametrize("perf", ["causal_no_mask", "all"])
def test_attn_fused_flex_model_cpu_forward(perf):
    check_model(f"attn_fused flex model [{perf}, cpu fwd]", "tiny", perf, "attn_fused=flex", "cpu", backward=False)

@needs_cuda
@pytest.mark.parametrize("perf", ["causal_no_mask", "all"])
def test_attn_fused_flex_model_cuda(perf):
    check_model(f"attn_fused flex model [{perf}, cuda]", "tiny", perf, "attn_fused=flex", "cuda")

@pytest.mark.parametrize("kern,device,backward", [("attn_fused=gqa", "cpu", True), ("attn_fused=flex", "cpu", False)]
                         + ([("attn_fused=gqa", "cuda", True), ("attn_fused=flex", "cuda", True)] if CUDA else []))
def test_attn_fused_model_layerscale_1(kern, device, backward):
    """ LayerScale starts at 1e-5, which hides attention differences below fp32 resolution in the loss (the tests above
    show 0.0 there); with LayerScale 1 the attention output reaches the loss at full weight """
    check_model(f"{kern} model [all, {device}, layerscale 1{'' if backward else ', fwd'}]", "tiny", "all", kern, device,
                backward=backward, layerscale_value=1.0)

@needs_cuda
def test_kern_all_model_cuda_small():
    """ the real REPA-S shapes (N = 196), both flags, all Tier 1 forward flags """
    check_model("patch_linear+attn_fused=flex model [all, cuda, S]", "small", "all", "patch_linear,attn_fused=flex",
                "cuda", batch=2)

@needs_cuda
def test_kern_all_model_cuda_compiled():
    """ both models under torch.compile (Inductor reorders float ops, hence rtol 1e-4 as test_fp32_cuda_compiled_all) """
    torch._dynamo.reset()
    check_model("patch_linear+attn_fused=flex model [all, cuda, compiled]", "tiny", "all", "patch_linear,attn_fused=flex",
                "cuda", compile_models=True, rtol=1e-4)

@pytest.mark.parametrize("perf,calls", [("causal_no_mask", 3), ("all", 2)]) # all: skip_last_content leaves the last layer's queries only
@pytest.mark.parametrize("mode", ["gqa", "flex"])
def test_attn_fused_path_is_taken(monkeypatch, perf, calls, mode):
    """ the equivalence tests above only mean something if the fused call actually runs (tiny: 3 layers) """
    seen = []
    orig = SelfAttention.fused_two_stream

    def counting(self, *a, **kw):
        seen.append(a[4] if len(a) > 4 else kw.get("mode"))
        return orig(self, *a, **kw)
    monkeypatch.setattr(SelfAttention, "fused_two_stream", counting)
    (cfg, _), (_, new) = model_pair("tiny", perf, f"attn_fused={mode}", "cpu")
    img, pos, _ = make_batch(cfg, 2, "cpu")
    with torch.no_grad():
        new(img, position_ids=pos)
    assert seen == [mode] * calls

def test_attn_fused_off_without_causal_path():
    """ without perf_causal_no_mask (masked attention), the flag must not change anything: bit-identical """
    (cfg_ref, ref), (cfg_new, new) = model_pair("tiny", "", "attn_fused=gqa", "cpu")
    img, pos, s = make_batch(cfg_ref, 2, "cpu")
    l_ref, g_ref = loss_and_grads(ref, cfg_ref, img, pos, s)
    l_new, g_new = loss_and_grads(new, cfg_new, img, pos, s)
    assert torch.equal(l_ref, l_new) and all(torch.equal(g_ref[n], g_new[n]) for n in g_ref)
