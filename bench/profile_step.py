"""
Profile REPA training steps (torch.profiler) and bucket GPU kernel time into matmul / attention / layernorm /
other_elementwise (plus optimizer and memcpy, reported separately). Saves JSON + a Chrome trace (open in Perfetto).

    python bench/profile_step.py --model S --flags none
    python bench/profile_step.py --model S --flags all
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from pytorch_lightning.callbacks import Callback
from torch.profiler import ProfilerActivity, profile

from harness import (REPO, LitREPA, add_common_args, build_config, build_datamodule, build_trainer, environment,
                     flag_names, run_tag)

def bucket(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("flash", "fmha", "attention", "attn", "sdpa")):
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "xmma", "cublas", "matmul", "triton_tem", "s16816", "s1688", "wgmma")):
        return "matmul"
    if "layer_norm" in n or "layernorm" in n:
        return "layernorm"
    if any(k in n for k in ("adam", "multi_tensor")):
        return "optimizer"
    if any(k in n for k in ("memcpy", "memset")):
        return "memcpy"
    return "other_elementwise"

class Profiler(Callback):
    """ profiles the steps after the warmup """
    def __init__(self, warmup, active):
        self.warmup, self.active, self.seen = warmup, active, 0
        self.prof = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.seen == self.warmup:
            torch.cuda.synchronize()
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
            self.prof.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.seen += 1
        if self.prof is not None and self.seen == self.warmup + self.active:
            torch.cuda.synchronize()
            self.prof.__exit__(None, None, None)

def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.set_defaults(steps=10)
    p.add_argument("--out", default=os.path.join(REPO, "bench", "results", "profiles"))
    args = p.parse_args()

    cfg = build_config(args)
    args.batch = cfg.batch_size
    flags = flag_names(args.flags)
    profiler = Profiler(args.warmup, args.steps)
    trainer, optimizer, accum, total = build_trainer(args, cfg, [profiler])
    dm = build_datamodule(args, cfg, total)
    torch.manual_seed(cfg.seed)
    trainer.fit(LitREPA(cfg), datamodule=dm)
    prof = profiler.prof

    per_bucket, per_kernel, intervals = defaultdict(float), defaultdict(lambda: [0.0, 0]), []
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA or e.is_user_annotation or e.name.startswith("ProfilerStep"):
            continue # kernels only: annotations (ProfilerStep, "## Call CompiledFxGraph ##") span many kernels
        dur = e.time_range.end - e.time_range.start # us
        intervals.append((e.time_range.start, e.time_range.end))
        per_bucket[bucket(e.name)] += dur
        per_kernel[e.name][0] += dur
        per_kernel[e.name][1] += 1
    intervals.sort()
    busy, cur_s, cur_e = 0.0, None, None # union of kernel intervals = GPU busy time
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = intervals[-1][1] - intervals[0][0] if intervals else 0.0
    total_kernel = sum(per_bucket.values())
    n = args.steps

    result = {
        "model": args.model, "flags": flags, "batch": cfg.batch_size, "data": args.data, "optimizer": optimizer,
        "accum": accum if optimizer else None, "precision": args.precision, "compile": cfg.compile, "steps_profiled": n,
        "gpu_span_ms_per_step": span / 1e3 / n, "gpu_busy_ms_per_step": busy / 1e3 / n,
        "gpu_idle_pct": 100 * (1 - busy / span) if span else None,
        "buckets_ms_per_step": {k: v / 1e3 / n for k, v in sorted(per_bucket.items(), key=lambda kv: -kv[1])},
        "buckets_pct_of_kernel_time": {k: 100 * v / total_kernel for k, v in sorted(per_bucket.items(), key=lambda kv: -kv[1])},
        "top_kernels": [{"name": k, "bucket": bucket(k), "ms_per_step": v / 1e3 / n, "calls_per_step": c / n}
                        for k, (v, c) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:40]],
        **environment(),
    }
    os.makedirs(args.out, exist_ok=True)
    tag = run_tag(args, flags)
    prof.export_chrome_trace(os.path.join(args.out, tag + ".trace.json"))
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"{args.model} [{'+'.join(flags) or 'baseline'}] b{cfg.batch_size}: GPU {result['gpu_span_ms_per_step']:.1f} ms/step, "
          f"idle {result['gpu_idle_pct']:.1f}%")
    for k, v in result["buckets_ms_per_step"].items():
        print(f"  {k:18s} {v:8.2f} ms/step {result['buckets_pct_of_kernel_time'][k]:6.1f}%")
    print(f"-> {os.path.join(args.out, tag)}.json / .trace.json")

if __name__ == "__main__":
    main()
