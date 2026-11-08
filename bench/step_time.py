"""
Time REPA pretraining steps with and without the perf_* flags. One GPU, the real LitREPA step through Lightning.

    python bench/step_time.py --model S --flags none --data synthetic
    python bench/step_time.py --model S --flags all --data real --data-root /path/to/imagenet --optimizer

ms/step is the interval between the starts of consecutive training batches (CUDA events: data wait, H2D copy,
forward, backward, optimizer); compute_ms is the start-to-end interval of a batch (no data wait). imgs/sec comes from
wall-clock time over the timed steps. Writes one JSON file to bench/results/.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from harness import (REPO, LitREPA, add_common_args, build_config, build_datamodule, build_trainer, environment,
                     flag_names, run_tag, stop_callback)

class StepTimer(Callback):
    def __init__(self, warmup):
        self.warmup, self.starts, self.ends = warmup, [], []
        self.t0 = self.t1 = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if len(self.starts) == self.warmup: # first timed step
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.t0 = time.perf_counter()
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.starts.append(e)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.ends.append(e)

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        self.t1 = time.perf_counter()

def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--out", default=os.path.join(REPO, "bench", "results"))
    args = p.parse_args()

    cfg = build_config(args)
    args.batch = cfg.batch_size
    flags = flag_names(args.flags)
    timer = StepTimer(args.warmup)
    trainer, optimizer, accum, total = build_trainer(args, cfg, [timer])
    dm = build_datamodule(args, cfg, total)
    torch.manual_seed(cfg.seed)
    trainer.fit(LitREPA(cfg), datamodule=dm)

    w, n = args.warmup, args.steps
    period = np.array([timer.starts[i].elapsed_time(timer.starts[i + 1]) for i in range(w, w + n)])
    compute = np.array([timer.starts[i].elapsed_time(timer.ends[i]) for i in range(w, w + n)])
    wall = timer.t1 - timer.t0 # covers the n timed steps plus the one stopping step
    result = {
        "model": args.model, "flags": flags, "batch": cfg.batch_size, "data": args.data,
        "data_root": args.data_root, "num_workers": args.num_workers if args.data == "real" else 0,
        "optimizer": optimizer, "accum": accum if optimizer else None, "precision": args.precision,
        "compile": cfg.compile, "warmup": w, "steps": n,
        "ms_per_step_median": float(np.median(period)), "ms_per_step_p90": float(np.percentile(period, 90)),
        "compute_ms_median": float(np.median(compute)), "compute_ms_p90": float(np.percentile(compute, 90)),
        "imgs_per_sec": cfg.batch_size * (n + 1) / wall,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        **stop_callback(trainer).loss_summary(),
        **environment(),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, run_tag(args, flags) + ".json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{args.model} {args.data} [{'+'.join(flags) or 'baseline'}] b{cfg.batch_size}: "
          f"{result['ms_per_step_median']:.1f} ms/step (p90 {result['ms_per_step_p90']:.1f}), "
          f"{result['imgs_per_sec']:.0f} img/s, peak {result['peak_mem_gib']:.2f} GiB, "
          f"loss {result['loss_first']:.4f} -> {result['loss_last']:.4f} (finite: {result['loss_all_finite']}) -> {path}")

if __name__ == "__main__":
    main()
