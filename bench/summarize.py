"""
bench/results/*.json (from step_time.py) -> markdown tables, one per (GPU, model, data, batch, optimizer) group,
with the speedup of each flag set over that group's baseline. Uses the newest result per flag set.

    python bench/summarize.py                     # prints and writes bench/results/summary.md
"""
import argparse
import glob
import json
import os

ORDER = ["", "skip_last_content", "causal_no_mask", "query_proj_once", "no_cpu_sync", "uint8_images", "fused_adamw",
         "ddp_static_graph"]

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
            runs.append(r)

    groups = {}
    for r in sorted(runs, key=lambda r: r["time"]): # newest last, so it wins
        key = (r["gpu"], r["model"], r["data"], r["batch"], r["optimizer"], r.get("accum"), r["compile"], r["precision"])
        groups.setdefault(key, {})["+".join(r["flags"])] = r

    lines = []
    for (gpu, model, data, batch, opt, accum, comp, prec), by_flags in sorted(groups.items(), key=lambda kv: str(kv[0])):
        lines.append(f"### {model}, {data} data, batch {batch}, "
                     f"{'optimizer every ' + str(accum) + ' step(s)' if opt else 'no optimizer step'}, {gpu}")
        lines.append(f"torch {next(iter(by_flags.values()))['torch']}, compile={comp}, {prec}\n")
        lines.append("| flags | ms/step (median) | ms/step (p90) | compute ms | img/s | peak GiB | speedup |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        base = by_flags.get("")
        keys = sorted(by_flags, key=lambda k: ("+" in k, ORDER.index(k) if k in ORDER else 99, k))
        for k in keys:
            r = by_flags[k]
            speedup = f"{base['ms_per_step_median'] / r['ms_per_step_median']:.3f}x" if base else "n/a"
            perf = set(ORDER[1:])
            rest = [f for f in r["flags"] if f not in perf]
            name = "baseline" if k == "" else ("+".join(["all"] + rest) if perf <= set(r["flags"]) else k)
            lines.append(f"| {name} | {r['ms_per_step_median']:.1f} | {r['ms_per_step_p90']:.1f} | "
                         f"{r['compute_ms_median']:.1f} | {r['imgs_per_sec']:.0f} | {r['peak_mem_gib']:.2f} | {speedup} |")
        lines.append("")
    text = "\n".join(lines) if lines else "no results found"
    print(text)
    out = args.out or os.path.join(args.results, "summary.md")
    with open(out, "w") as f:
        f.write(text + "\n")

if __name__ == "__main__":
    main()
