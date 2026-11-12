"""
One Tier 3 benchmark step: bench/step_time.py or bench/profile_step.py with settings that need no change to the model
code, applied in this process before the model is built.
  --mode                 torch.compile mode for LitREPA's torch.compile call (repa.py:60): default, reduce-overhead,
                         max-autotune, max-autotune-no-cudagraphs
  --coord-descent        torch._inductor.config.coordinate_descent_tuning = True
  --fp8-recipe           torchao Float8LinearConfig.from_recipe_name(...) for perf_fp8 (repa.py:40-47 uses the default
                         recipe, tensorwise); needs "fp8" in --flags
  --sdpa                 restrict SDPA to one backend (global torch.backends.cuda toggles): flash, cudnn, efficient
Everything after the known options goes to the step script unchanged.

    python bench/kern_step.py --kind time --mode max-autotune --meta m.json -- --model S --flags all --out dir
"""
import argparse
import json
import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
sys.path.insert(1, os.path.dirname(BENCH)) # the repo, for `import repa` before harness.py adds it

MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
SDPA = ("default", "flash", "cudnn", "efficient")
RECIPES = ("tensorwise", "rowwise", "rowwise_with_gw_hp")

def apply_compile_mode(torch, mode):
    if mode == "default":
        return
    original = torch.compile

    def compile_with_mode(model=None, **kw):
        kw.setdefault("mode", mode)
        return original(model, **kw)
    torch.compile = compile_with_mode # repa.py looks torch.compile up at call time

CUDAGRAPH_MODES = ("reduce-overhead", "max-autotune")

def stable_grad_buffers(torch):
    """ CUDA graph modes + gradient accumulation: a .grad first allocated inside a CUDA graph replay is overwritten by
    the next replay ("accessing gradient tensor output of CUDAGraphs that has been overwritten", seen on the 3060 with
    reduce-overhead). Preallocate zeroed .grad buffers before training and zero them in place instead of setting them
    to None. Same numbers: accumulating into zeros equals starting from None. """
    from repa import LitREPA

    def on_train_start(self):
        for p in self.parameters():
            if p.requires_grad and p.grad is None:
                p.grad = torch.zeros_like(p)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=False)
    LitREPA.on_train_start = on_train_start
    LitREPA.optimizer_zero_grad = optimizer_zero_grad

def apply_fp8_recipe(torch, recipe):
    import repa
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    config = Float8LinearConfig.from_recipe_name(recipe)

    def convert_to_fp8(model): # repa.convert_to_fp8 with an explicit recipe; same module filter
        return convert_to_float8_training(
            model, config=config,
            module_filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and m.in_features % 16 == 0 and m.out_features % 16 == 0)
    repa.convert_to_fp8 = convert_to_fp8 # LitREPA.__init__ calls it by module-global name

def apply_sdpa(torch, backend):
    if backend == "default":
        return
    b = torch.backends.cuda
    b.enable_flash_sdp(backend == "flash")
    b.enable_mem_efficient_sdp(backend == "efficient")
    b.enable_cudnn_sdp(backend == "cudnn")
    b.enable_math_sdp(False) # math would otherwise come before cudnn in the default priority order

def sdpa_probe(torch):
    """ which kernel family an is_causal bf16 call (our S shape) dispatches to under the current toggles """
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile
    q, k, v = (torch.randn(8, 6, 196, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()
            torch.cuda.synchronize()
    except Exception as e:
        return {"family": "error", "error": str(e)[:300]}
    names = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    n = " ".join(names).lower()
    # cudnn first: cuDNN's kernel names contain "flash" too (cudnn_generated_..._sdpa_sm80_flash_fprop_...)
    family = next((f for f, keys in (("cudnn", ("cudnn",)), ("flash", ("flash",)), ("efficient", ("fmha", "mem_eff")),
                                     ("math", ("softmax", "bmm", "gemm"))) if any(x in n for x in keys)), "unknown")
    return {"family": family, "kernels": [x[:120] for x in names if any(k in x.lower() for k in ("flash", "fmha", "cudnn", "attention", "sdpa"))][:6]}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" in argv:
        i = argv.index("--")
        own, rest = argv[:i], argv[i + 1:]
    else:
        own, rest = argv, []
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["time", "profile"], required=True)
    p.add_argument("--mode", choices=MODES, default="default")
    p.add_argument("--coord-descent", action="store_true")
    p.add_argument("--fp8-recipe", choices=RECIPES, default=None)
    p.add_argument("--sdpa", choices=SDPA, default="default")
    p.add_argument("--meta", default=None, help="write the applied settings (and the SDPA probe) here")
    args = p.parse_args(own)

    import torch
    apply_compile_mode(torch, args.mode)
    if args.mode in CUDAGRAPH_MODES:
        stable_grad_buffers(torch)
    if args.coord_descent:
        import torch._inductor.config as inductor_config
        inductor_config.coordinate_descent_tuning = True
    if args.fp8_recipe:
        apply_fp8_recipe(torch, args.fp8_recipe)
    apply_sdpa(torch, args.sdpa)

    meta = {"kind": args.kind, "mode": args.mode, "coord_descent": args.coord_descent, "fp8_recipe": args.fp8_recipe,
            "sdpa": args.sdpa, "stable_grad_buffers": args.mode in CUDAGRAPH_MODES, "step_args": rest}
    if args.sdpa != "default":
        meta["sdpa_probe"] = sdpa_probe(torch)
    if args.meta: # before the run, so a failed run still says what was applied
        with open(args.meta, "w") as f:
            json.dump(meta, f, indent=2)

    script = {"time": "step_time", "profile": "profile_step"}[args.kind]
    module = __import__(script)
    sys.argv = [os.path.join(BENCH, script + ".py"), *rest]
    module.main()

if __name__ == "__main__":
    main()
