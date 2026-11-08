"""
bench/results/*.json (from step_time.py) -> markdown tables, one per (GPU, model, data, batch, optimizer) group,
with the speedup of each flag set over that group's baseline. Uses the newest result per flag set. Architecture-flag
runs get a second table with the FLOP-model estimate next to the measured speedup.

    python bench/summarize.py                     # prints and writes bench/results/summary.md
"""
import argparse
import glob
import json
import os

ORDER = ["", "skip_last_content", "causal_no_mask", "query_proj_once", "no_cpu_sync", "uint8_images", "fused_adamw",
         "ddp_static_graph"]

# FLOP-model speedups over the two-stream baseline, from the report (repa_benchmark_addendum.md). They are FLOP ratios,
# not step-time predictions.
ESTIMATES = {
    "single_stream_ref": ("1.83", "flop_model.py: single stream, S 1.835x / B 1.832x"),
    "query_top_k=6": ("1.36", ""),
    "query_top_k=4": ("1.52", ""),
    "single_stream_target_pos": ("1.83", ""),
    "query_mlp_width=2d": ("1.26", ""),
    "query_mlp_width=d": ("1.40", ""),
    "train_res=160": ("~1.6", "for a 160 -> 224 px schedule (75/25); the 160 px step alone is 2.12x (S) / 2.08x (B) in flop_model.py"),
    "predict_frac=0.5": ("1.36", ""),
    "predict_frac=0.25": ("1.61", ""),
}

def perf_name(flags):
    perf = set(ORDER[1:])
    if not flags:
        return "baseline"
    rest = [f for f in flags if f not in perf]
    return "+".join(["all"] + rest) if perf <= set(flags) else "+".join(flags)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(here, "results"))
    p.add_argument("--out", default=None, help="default: <results>/summary.md")
    args = p.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if "ms_per_step_median" in r:
            r.setdefault("arch", "")
            runs.append(r)

    groups = {}
    for r in sorted(runs, key=lambda r: r["time"]): # newest last, so it wins
        key = (r["gpu"], r["model"], r["data"], r["batch"], r["optimizer"], r.get("accum"), r["compile"], r["precision"])
        groups.setdefault(key, {})[(r["arch"], "+".join(r["flags"]))] = r

    lines = []
    for (gpu, model, data, batch, opt, accum, comp, prec), runs_ in sorted(groups.items(), key=lambda kv: str(kv[0])):
        head = (f"### {model}, {data} data, batch {batch}, "
                f"{'optimizer every ' + str(accum) + ' step(s)' if opt else 'no optimizer step'}, {gpu}")
        env = f"torch {next(iter(runs_.values()))['torch']}, compile={comp}, {prec}\n"
        base = runs_.get(("", ""))
        perf_rows = {k[1]: r for k, r in runs_.items() if k[0] == ""}
        arch_rows = {k: r for k, r in runs_.items() if k[0] != ""}

        if perf_rows:
            lines += [head, env,
                      "| flags | ms/step (median) | ms/step (p90) | compute ms | img/s | peak GiB | speedup |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for k in sorted(perf_rows, key=lambda k: ("+" in k, ORDER.index(k) if k in ORDER else 99, k)):
                r = perf_rows[k]
                speedup = f"{base['ms_per_step_median'] / r['ms_per_step_median']:.3f}x" if base else "n/a"
                lines.append(f"| {perf_name(r['flags'])} | {r['ms_per_step_median']:.1f} | {r['ms_per_step_p90']:.1f} | "
                             f"{r['compute_ms_median']:.1f} | {r['imgs_per_sec']:.0f} | {r['peak_mem_gib']:.2f} | {speedup} |")
            lines.append("")

        if arch_rows:
            lines += [head + " (architecture flags: these change the model)", env,
                      "| arch | perf flags | ms/step | img/s | peak GiB | vs baseline | vs same perf flags | FLOP estimate | note |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---|"]
            for (arch, pk) in sorted(arch_rows, key=lambda k: (list(ESTIMATES).index(k[0]) if k[0] in ESTIMATES else 99, k)):
                r = arch_rows[(arch, pk)]
                same_perf = perf_rows.get(pk)
                vs_base = f"{base['ms_per_step_median'] / r['ms_per_step_median']:.3f}x" if base else "n/a"
                vs_perf = f"{same_perf['ms_per_step_median'] / r['ms_per_step_median']:.3f}x" if same_perf else "n/a"
                est, note = ESTIMATES.get(arch, ("n/a", ""))
                lines.append(f"| {arch} | {perf_name(r['flags'])} | {r['ms_per_step_median']:.1f} | {r['imgs_per_sec']:.0f} | "
                             f"{r['peak_mem_gib']:.2f} | {vs_base} | {vs_perf} | {est} | {note} |")
            lines.append("")

    text = "\n".join(lines) if lines else "no results found"
    print(text)
    out = args.out or os.path.join(args.results, "summary.md")
    with open(out, "w") as f:
        f.write(text + "\n")

if __name__ == "__main__":
    main()
