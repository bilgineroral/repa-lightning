"""
Tier 3 benchmark sequence (bench/KERNEL_PLAN.md section 5) on one GPU. Started by bench/run_all.sh, which
runs bench/kern_preflight.py first (GPU, driver -> conda env, idle check) and then this script with the env's python.

Steps, in the doc's order (S and B unless --models):
  00 vanilla                base REPA, no Tier 1 flags, default compile (step_time.py --flags none, the vanilla arm of
                            bench/run_tier1.sh): the reference for loss_vs_vanilla and speedup_vs_vanilla
  00 torch preflight        bench/preflight.py: versions, SDPA dispatch, DATA_ROOT layout
  01 profile, default       all Tier 1 flags, default torch.compile
  02 compile modes          default / reduce-overhead / max-autotune / max-autotune-no-cudagraphs, + coordinate descent
  03 FP8 recipes            tensorwise / rowwise / rowwise_with_gw_hp, with default compile and with the best 02 mode
  04 profile, best          best of 02-03, then the re-rank rule (memory-bound groups > 5% of kernel time move up)
  05 ranks 3-5              SDPA backend (cudnn, efficient), kern_patch_linear, kern_attn_fused (gqa, flex)
  06 data                   real vs synthetic data with the best settings (needs --data-root)
Each step writes <results>/<run_id>/<step>.json and is skipped when that file exists (resumable). A failed step writes
<step>.failed.json instead and is retried on the next run. Every result records nvidia-smi and system load snapshots.
"""
import argparse
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

BENCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH)

MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
TEMPLATE_MODES = ("max-autotune", "max-autotune-no-cudagraphs") # Inductor GEMM templates: need >= 68 SMs
FP8_RECIPES = ("tensorwise", "rowwise", "rowwise_with_gw_hp")
MIN_SMS_TEMPLATES = 68 # torch/_inductor/utils.py is_big_gpu
FP8_CAPABILITY = (8, 9) # repa.py fp8_supported
RERANK_THRESHOLD_PCT = 5.0
# memory-bound kernel groups of the re-rank rule (kernel_research.md section 5) and the build-plan flag each promotes
RERANK_GROUPS = {
    "gelu": "kern_gelu_epilogue",
    "rope_qknorm": "kern_gelu_epilogue / kern_stacked_streams (QK-norm + RoPE epilogue)",
    "column_reductions": "Triton multi-output reduction (section 4.8c)",
    "transpose_copy": "kern_stacked_streams",
    "layernorm_fused": "kern_ln_prologue",
}

# ---------------------------------------------------------------- plan

@dataclass
class Step:
    id: str
    kind: str # "preflight", "time", "profile"
    model: str = ""
    flags: str = "all"
    mode: str = "default" # or "@best"
    coord_descent: bool = False # or resolved with "@best"
    fp8_recipe: str = "" # "", a recipe, or "@best"
    sdpa: str = "default"
    data: str = "synthetic"
    kern: str = "" # --kern flags (config.py KERN_FLAGS)
    requires: list = field(default_factory=list) # "fp8", "templates", "data_root", "config:<field>"
    note: str = ""

LOSS_TOL_FP8, LOSS_TOL = 0.05, 0.01 # |loss_vs_vanilla| above these is flagged in the summary

def build_plan(models, dry_run):
    # vanilla first: the reference every other step's loss and speed is compared with
    plan = [Step(f"00_vanilla_{m}", "time", m, flags="none") for m in models]
    plan.append(Step("00_preflight", "preflight"))
    for m in models:
        plan.append(Step(f"01_profile_{m}_default", "profile", m))
    for m in models:
        for mode in MODES:
            plan.append(Step(f"02_compile_{m}_{mode}", "time", m, mode=mode,
                             requires=["templates"] if mode in TEMPLATE_MODES else []))
        plan.append(Step(f"02_compile_{m}_default+cd", "time", m, coord_descent=True))
        plan.append(Step(f"02_compile_{m}_max-autotune+cd", "time", m, mode="max-autotune", coord_descent=True,
                         requires=["templates"]))
    for m in models:
        for r in FP8_RECIPES:
            plan.append(Step(f"03_fp8_{m}_{r}_default", "time", m, flags="all,fp8", fp8_recipe=r, requires=["fp8"]))
            plan.append(Step(f"03_fp8_{m}_{r}_bestmode", "time", m, flags="all,fp8", fp8_recipe=r, mode="@best",
                             requires=["fp8", "best_differs"]))
    for m in models:
        plan.append(Step(f"04_profile_{m}_best", "profile", m, mode="@best", fp8_recipe="@best"))
    for m in models:
        for b in ("cudnn", "efficient"): # flash is the default backend (step 02 default)
            plan.append(Step(f"05_sdpa_{m}_{b}", "time", m, mode="@best", fp8_recipe="@best", sdpa=b))
        plan.append(Step(f"05_patch_linear_{m}", "time", m, mode="@best", fp8_recipe="@best", kern="patch_linear",
                         requires=["config:kern_patch_linear"]))
        for v in ("gqa", "flex"):
            plan.append(Step(f"05_attn_fused_{v}_{m}", "time", m, mode="@best", fp8_recipe="@best",
                             kern=f"attn_fused={v}", requires=["config:kern_attn_fused"]))
    for m in models:
        for d in ("synthetic", "real"):
            plan.append(Step(f"06_data_{m}_{d}", "time", m, mode="@best", fp8_recipe="@best", data=d,
                             requires=["data_root"] if d == "real" else []))
    if dry_run: # the local 3060: sm_89-only steps never run
        for s in plan:
            if "fp8" in s.requires or "templates" in s.requires:
                s.requires.append("not_dry_run")
    return plan

def unmet(step, hw, args, config_fields):
    """ -> reason the step can't run here, or None """
    for pattern in getattr(args, "skip", None) or []:
        if re.search(pattern, step.id):
            return f"--skip {pattern}"
    for r in step.requires:
        if r == "not_dry_run" and args.dry_run:
            return "dry run: sm_89-only step"
        if r == "fp8" and tuple(hw["capability"]) < FP8_CAPABILITY:
            return f"FP8 needs capability >= 8.9, GPU has {hw['capability'][0]}.{hw['capability'][1]}"
        if r == "templates" and hw["sms"] < MIN_SMS_TEMPLATES:
            return f"max-autotune GEMM templates need >= {MIN_SMS_TEMPLATES} SMs, GPU has {hw['sms']}"
        if r == "data_root" and not args.data_root:
            return "no --data-root"
        if r.startswith("config:") and r[len("config:"):] not in config_fields:
            return f"{r[len('config:'):]} not implemented in config.py yet (Tier 3 build plan)"
    return None

# ---------------------------------------------------------------- best settings from finished steps

def load_results(run_dir):
    out = {}
    for path in glob.glob(os.path.join(run_dir, "*.json")):
        name = os.path.basename(path)[:-len(".json")]
        if name.endswith(".failed") or name in ("summary", "rerank") or name.startswith("preflight_gate"):
            continue
        with open(path) as f:
            out[name] = json.load(f)
    return out

def ms(rec):
    return (rec.get("result") or {}).get("ms_per_step_median")

def best_mode(results, model):
    """ fastest 02 step for the model -> (mode, coord_descent, step id); default if none finished """
    cands = [(ms(r), r["settings"]["mode"], r["settings"]["coord_descent"], k) for k, r in results.items()
             if k.startswith(f"02_compile_{model}_") and r.get("status") == "ok" and ms(r)]
    if not cands:
        return "default", False, None
    _, mode, cd, k = min(cands)
    return mode, cd, k

def best_overall(results, model):
    """ fastest of 02 and 03 -> settings dict; FP8 only if it beats the best non-FP8 step """
    mode, cd, k = best_mode(results, model)
    best = {"mode": mode, "coord_descent": cd, "fp8_recipe": "", "from": k,
            "ms": ms(results[k]) if k else None}
    for name, r in results.items():
        if name.startswith(f"03_fp8_{model}_") and r.get("status") == "ok" and ms(r) and (best["ms"] is None or ms(r) < best["ms"]):
            s = r["settings"]
            best = {"mode": s["mode"], "coord_descent": s["coord_descent"], "fp8_recipe": s["fp8_recipe"], "from": name,
                    "ms": ms(r)}
    return best

def resolve(step, results):
    """ concrete settings for a step whose mode / FP8 recipe is '@best' """
    s = {"mode": step.mode, "coord_descent": step.coord_descent, "fp8_recipe": step.fp8_recipe, "flags": step.flags,
         "kern": step.kern}
    if step.fp8_recipe == "@best":
        b = best_overall(results, step.model)
        s.update(mode=b["mode"], coord_descent=b["coord_descent"], fp8_recipe=b["fp8_recipe"], best_from=b["from"])
    elif step.mode == "@best":
        mode, cd, k = best_mode(results, step.model)
        s.update(mode=mode, coord_descent=cd, best_from=k)
    if s["fp8_recipe"] and "fp8" not in s["flags"].split(","):
        s["flags"] += ",fp8"
    return s

# ---------------------------------------------------------------- snapshots

def _run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"failed: {e}"

def snapshot(gpu):
    meminfo = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in ("MemTotal", "MemAvailable"):
                    meminfo[k] = v.strip()
    except OSError:
        pass
    try:
        load = os.getloadavg()
    except OSError:
        load = None
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nvidia_smi": _run(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,name,driver_version,temperature.gpu,"
                            "power.draw,clocks.sm,clocks.mem,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader"]),
        "gpu_processes": _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                               "--format=csv,noheader"]),
        "loadavg": load, "cpus": os.cpu_count(), "meminfo": meminfo,
        "top_cpu": _run(["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"]).splitlines()[:6],
    }

# ---------------------------------------------------------------- running one step

class Runner:
    def __init__(self, args, hw, py, run_dir, log):
        self.args, self.hw, self.py, self.run_dir, self.log = args, hw, py, run_dir, log
        self.child = None

    def batch_accum(self, model):
        batch = {"S": self.args.batch_s, "B": self.args.batch_b}[model]
        # dry run: accumulation 2, so the few steps include optimizer steps (fused AdamW, grad zeroing under CUDA graphs)
        global_batch = self.args.global_batch or (2 * batch if self.args.dry_run else 4096)
        return batch, max(1, global_batch // batch)

    def command(self, step, s, raw):
        if step.kind == "preflight":
            cmd = [self.py, os.path.join(BENCH, "preflight.py"), "--json", os.path.join(raw, "preflight.json")]
            return cmd + (["--data-root", self.args.data_root] if self.args.data_root else [])
        batch, accum = self.batch_accum(step.model)
        warmup = self.args.warmup
        steps = self.args.profile_steps if step.kind == "profile" else self.args.steps
        inner = ["--model", step.model, "--flags", s["flags"], "--batch", str(batch), "--warmup", str(warmup),
                 "--steps", str(steps), "--out", raw]
        if s.get("kern"):
            inner += ["--kern", s["kern"]]
        if step.kind == "time":
            inner += ["--optimizer", "--accum", str(accum), "--data", step.data]
            if step.data == "real":
                inner += ["--data-root", self.args.data_root, "--num-workers", str(self.args.num_workers)]
        own = [self.py, os.path.join(BENCH, "kern_step.py"), "--kind", step.kind, "--mode", s["mode"],
               "--sdpa", step.sdpa, "--meta", os.path.join(raw, "meta.json")]
        if s["coord_descent"]:
            own.append("--coord-descent")
        if s["fp8_recipe"]:
            own += ["--fp8-recipe", s["fp8_recipe"]]
        return own + ["--"] + inner

    def execute(self, step, s):
        """ run one step -> record dict (status 'ok' or 'failed') """
        raw = os.path.join(self.run_dir, "raw", step.id)
        if os.path.isdir(raw): # leftovers of an interrupted or failed attempt
            shutil.rmtree(raw)
        os.makedirs(raw)
        cmd = self.command(step, s, raw)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.args.gpu_visible, **cache_env(self.run_dir, s, step.sdpa))
        log_path = os.path.join(self.run_dir, "logs", f"{step.id}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        before = snapshot(self.args.gpu)
        t0 = time.time()
        status, rc, err = "failed", None, None
        with open(log_path, "w") as logf:
            logf.write(" ".join(cmd) + "\n\n")
            logf.flush()
            self.child = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            try:
                rc = self.child.wait(timeout=self.args.step_timeout)
            except subprocess.TimeoutExpired:
                kill_group(self.child)
                err = f"timeout after {self.args.step_timeout} s"
            finally:
                self.child = None
        duration = time.time() - t0
        after = snapshot(self.args.gpu)
        result, meta = None, None
        if rc == 0:
            result = self.find_result(step, raw)
            if result is None:
                err = "exit 0 but no result JSON"
            else:
                status = "ok"
        elif err is None:
            err = f"exit code {rc}"
        if os.path.exists(os.path.join(raw, "meta.json")):
            with open(os.path.join(raw, "meta.json")) as f:
                meta = json.load(f)
        rec = {"step": step.id, "status": status, "error": err, "kind": step.kind, "model": step.model,
               "data": step.data, "sdpa": step.sdpa, "note": step.note, "settings": s, "command": cmd,
               "returncode": rc, "duration_s": round(duration, 1), "timeout_s": self.args.step_timeout,
               "log": os.path.relpath(log_path, self.run_dir), "raw_dir": os.path.relpath(raw, self.run_dir),
               "snapshot_before": before, "snapshot_after": after, "meta": meta, "result": result,
               "dry_run": self.args.dry_run, "hostname": socket.gethostname()}
        if status != "ok":
            with open(log_path) as f:
                rec["log_tail"] = f.read()[-3000:]
        return rec

    def find_result(self, step, raw):
        if step.kind == "preflight":
            p = os.path.join(raw, "preflight.json")
        else:
            files = [f for f in glob.glob(os.path.join(raw, "*.json"))
                     if not f.endswith((".trace.json", "meta.json"))]
            p = max(files, key=os.path.getmtime) if files else None
        if not p or not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

def cache_family(settings):
    """ Inductor cache directory name for a step's settings. Modes that differ only in CUDA graphs share one: CUDA
    graphs are applied after compilation, the compiled kernels and GEMM autotune results are the same, and the FX graph
    cache key includes the cudagraphs option anyway. Coordinate descent gets its own directory, so its tuned kernel
    configs never leak into the steps without it. """
    family = {"default": "default", "reduce-overhead": "default", "max-autotune": "max-autotune",
              "max-autotune-no-cudagraphs": "max-autotune"}[settings["mode"]]
    return family + ("+cd" if settings.get("coord_descent") else "")

def cache_env(run_dir, settings, sdpa):
    """ environment for a step's Inductor caches, shared by every step with the same cache_family, so max-autotune
    benchmarks each GEMM shape once per run instead of once per step. The FX graph and AOTAutograd cache keys ignore the
    SDPA backend toggles (seen on the 3060: an "efficient" step replayed the cudnn step's compiled graph), so steps with
    a non-default SDPA backend turn those two caches off: they still reuse the GEMM autotune results (keyed by kernel
    and shape) but compile their own graphs. """
    env = {"TORCHINDUCTOR_CACHE_DIR": os.path.join(run_dir, "inductor_cache", cache_family(settings))}
    if sdpa != "default":
        env.update(TORCHINDUCTOR_FX_GRAPH_CACHE="0", TORCHINDUCTOR_AUTOGRAD_CACHE="0")
    return env

def kill_group(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            continue

def write_json(path, obj):
    """ atomic: a run killed mid-write never leaves a partial <step>.json that would count as done """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def run_plan(plan, run_dir, execute, can_run, resolve_fn, log):
    """ the resumable loop: skip steps whose <id>.json exists, record unmet requirements, run the rest.
    execute(step, settings) -> record; can_run(step) -> None or reason; resolve_fn(step, results) -> settings.
    Returns {step id: (state, detail)} for this invocation. """
    states = {}
    for step in plan:
        done = os.path.join(run_dir, f"{step.id}.json")
        if os.path.exists(done):
            log(f"[skip] {step.id}: done ({os.path.basename(done)} exists)")
            states[step.id] = ("done", "")
            continue
        reason = can_run(step)
        if reason is None and "best_differs" in step.requires:
            s = resolve_fn(step, load_results(run_dir))
            if s["mode"] == "default" and not s["coord_descent"]:
                reason = "best compile mode is default: same as the _default FP8 step"
        if reason:
            log(f"[n/a ] {step.id}: {reason}")
            states[step.id] = ("not_run", reason)
            continue
        s = resolve_fn(step, load_results(run_dir))
        log(f"[run ] {step.id}: {describe(step, s)}")
        rec = execute(step, s)
        failed = os.path.join(run_dir, f"{step.id}.failed.json")
        if rec["status"] == "ok":
            write_json(done, rec)
            if os.path.exists(failed):
                os.remove(failed)
            r = rec.get("result") or {}
            extra = f", {r['ms_per_step_median']:.1f} ms/step" if r.get("ms_per_step_median") else ""
            log(f"[ ok ] {step.id} in {rec['duration_s']:.0f} s{extra}")
            states[step.id] = ("ok", "")
        else:
            write_json(failed, rec)
            log(f"[FAIL] {step.id}: {rec['error']} (see {rec.get('log')})")
            states[step.id] = ("failed", rec["error"])
    return states

def describe(step, s):
    parts = [f"model {step.model}"] if step.model else []
    if step.kind != "preflight":
        parts += [f"flags {s['flags']}", f"compile {s['mode']}" + (" +cd" if s["coord_descent"] else "")]
        if s.get("fp8_recipe"):
            parts.append(f"fp8 {s['fp8_recipe']}")
        if step.sdpa != "default":
            parts.append(f"sdpa {step.sdpa}")
        if step.data != "synthetic":
            parts.append(f"data {step.data}")
        if s.get("kern"):
            parts.append(f"kern {s['kern']}")
        if step.note:
            parts.append(step.note)
    return ", ".join(parts) or step.kind

# ---------------------------------------------------------------- re-rank rule (profile groups)

def kernel_group(name):
    n = name.lower()
    if any(k in n for k in ("flash", "fmha", "attention", "attn", "sdpa")):
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "xmma", "cublas", "matmul", "triton_tem", "s16816", "s1688", "wgmma",
                            "convolve")):
        return "matmul"
    if "gelu" in n:
        return "gelu"
    if re.search(r"\bcos\b|_cos_|_sin_|\bsin\b", n):
        return "rope_qknorm"
    if re.search(r"triton_red_fused.*sum", n):
        return "column_reductions"
    if re.search(r"clone_transpose|direct_copy|_unsafe_view.*clone|nchwtonhwc", n):
        return "transpose_copy"
    if "layer_norm" in n or "layernorm" in n:
        return "layernorm_fused"
    if any(k in n for k in ("memcpy", "memset")):
        return "memcpy"
    return "other"

def group_shares(kernels):
    """ kernels: iterable of (name, duration) -> {group: % of total kernel time} """
    tot, per = 0.0, {}
    for name, dur in kernels:
        g = kernel_group(name)
        per[g] = per.get(g, 0.0) + dur
        tot += dur
    return {g: 100 * v / tot for g, v in sorted(per.items(), key=lambda kv: -kv[1])} if tot else {}

def rerank_from_shares(shares, threshold=RERANK_THRESHOLD_PCT):
    promoted = [{"group": g, "pct": round(p, 2), "promote": RERANK_GROUPS[g]}
                for g, p in shares.items() if g in RERANK_GROUPS and p > threshold]
    return sorted(promoted, key=lambda x: -x["pct"])

def profile_kernels(run_dir, step_id):
    """ (name, dur) of every GPU kernel in a profile step's Chrome trace; falls back to the JSON's top_kernels """
    raw = os.path.join(run_dir, "raw", step_id)
    traces = glob.glob(os.path.join(raw, "*.trace.json"))
    if traces:
        with open(traces[0]) as f:
            ev = json.load(f).get("traceEvents", [])
        return [(e["name"], e.get("dur", 0)) for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rec_path = os.path.join(run_dir, f"{step_id}.json")
    if os.path.exists(rec_path):
        with open(rec_path) as f:
            r = json.load(f).get("result") or {}
        return [(k["name"], k["ms_per_step"]) for k in r.get("top_kernels", [])]
    return []

def rerank(run_dir, models):
    out = {}
    for m in models:
        sid = f"04_profile_{m}_best"
        kernels = profile_kernels(run_dir, sid)
        if not kernels:
            continue
        shares = group_shares(kernels)
        out[m] = {"profile": sid, "group_pct": {g: round(p, 2) for g, p in shares.items()},
                  "promoted": rerank_from_shares(shares), "threshold_pct": RERANK_THRESHOLD_PCT}
    return out

# ---------------------------------------------------------------- summary

def vs_vanilla(step, rec, results):
    """ (final loss, loss_vs_vanilla, speedup_vs_vanilla, flagged) of a time step vs 00_vanilla_<model>; None where it
    doesn't apply or the vanilla step is missing or failed. Flagged: |loss_vs_vanilla| above LOSS_TOL_FP8 for FP8
    steps, LOSS_TOL otherwise """
    r = (rec or {}).get("result") or {}
    loss = r.get("loss_last") if step.kind == "time" else None
    van = results.get(f"00_vanilla_{step.model}")
    vr = ((van or {}).get("result") or {}) if (van or {}).get("status") == "ok" else {}
    v_loss, v_ms = vr.get("loss_last"), vr.get("ms_per_step_median")
    rel = (loss - v_loss) / abs(v_loss) if (loss is not None and v_loss) else None
    speed = v_ms / r["ms_per_step_median"] if (step.kind == "time" and v_ms and r.get("ms_per_step_median")) else None
    fp8 = "fp8" in str(((rec or {}).get("settings") or {}).get("flags", "")).split(",")
    flagged = rel is not None and abs(rel) > (LOSS_TOL_FP8 if fp8 else LOSS_TOL)
    return loss, rel, speed, flagged

def summary_rows(plan, results, states):
    """ one row per step: [id, status, model, settings, ms/step, img/s, peak GiB, speedup (vs 02 default), run s,
    note, loss, loss_vs_vanilla, speedup_vs_vanilla, loss_flagged] """
    rows = []
    for step in plan:
        rec = results.get(step.id)
        state, detail = states.get(step.id, ("", ""))
        if rec:
            r, s = rec.get("result") or {}, rec.get("settings") or {}
            base = results.get(f"02_compile_{step.model}_default")
            speed = (ms(base) / r["ms_per_step_median"]) if (base and ms(base) and r.get("ms_per_step_median")) else None
            setting = "" if step.kind == "preflight" else (
                ("no Tier 1 flags, " if s.get("flags") in ("", "none") else "")
                + f"{s.get('mode')}{' +cd' if s.get('coord_descent') else ''}"
                + (f", fp8 {s['fp8_recipe']}" if s.get("fp8_recipe") else "")
                + (f", sdpa {step.sdpa}" if step.sdpa != "default" else "")
                + (f", kern {s['kern']}" if s.get("kern") else "")
                + (f", {step.data}" if step.data != "synthetic" else ""))
            if step.kind == "profile":
                b = r.get("buckets_pct_of_kernel_time", {})
                note = ", ".join(f"{k} {v:.0f}%" for k, v in list(b.items())[:4]) + f"; idle {r.get('gpu_idle_pct', 0):.1f}%"
            elif step.kind == "preflight":
                g = (r.get("gpus") or [{}])[0]
                note = f"{g.get('name', '?')} cc {g.get('capability', '?')}, torch {r.get('torch')}"
            else:
                note = ("loss finite" if r.get("loss_all_finite") else "LOSS NOT FINITE") if r else ""
                if s.get("mode") in ("reduce-overhead", "max-autotune"): # CUDA graph pools are allocated before the
                    note += "; peak GiB excludes CUDA-graph memory" # timed steps reset the peak counter
            rows.append([step.id, "ok" if state != "done" else "ok (earlier run)", step.model, setting,
                         r.get("ms_per_step_median") if step.kind == "time" else (r.get("gpu_span_ms_per_step") if step.kind == "profile" else None),
                         r.get("imgs_per_sec"), r.get("peak_mem_gib"), speed, rec.get("duration_s"), note,
                         *vs_vanilla(step, rec, results)])
        else:
            rows.append([step.id, state or "not run", step.model, "", None, None, None, None, None, detail,
                         None, None, None, False])
    return rows

def fmt(v, nd=1):
    if v is None:
        return ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

def summary_markdown(rows, hw, args, rr, elapsed):
    lines = [f"# Tier 3 benchmark run `{args.run_id}`", "",
             f"{hw.get('name')} (capability {hw['capability'][0]}.{hw['capability'][1]}, {hw.get('sms')} SMs), "
             f"physical GPU {args.gpu}, {'DRY RUN' if args.dry_run else 'full run'}, models {' '.join(args.models)}, "
             f"warmup {args.warmup}, steps {args.steps}. This invocation: {elapsed / 60:.1f} min.", "",
             "ms/step: median step time (time steps) or GPU span per step (profile steps). "
             "Speedup: vs `02_compile_<model>_default` (all Tier 1 flags, default compile). "
             "loss: final step's loss. vs vanilla: against `00_vanilla_<model>` (no Tier 1 flags, default compile; "
             f"same batch, steps and seed); a loss difference above {LOSS_TOL:.0%} ({LOSS_TOL_FP8:.0%} for FP8) is "
             "marked FLAG. n/a: not a timed step, or no successful vanilla step.", "",
             "| step | status | model | settings | ms/step | img/s | peak GiB | speedup | speedup vs vanilla | loss "
             "| loss vs vanilla | run s | note |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        rel = "n/a" if r[11] is None else f"{r[11]:+.2%}" + (" FLAG" if r[13] else "")
        svv = "n/a" if r[12] is None else f"{r[12]:.3f}x"
        loss = "n/a" if r[10] is None else f"{r[10]:.4f}"
        lines.append("| " + " | ".join([r[0], r[1], r[2], r[3], fmt(r[4]), fmt(r[5], 0), fmt(r[6], 2),
                                         fmt(r[7], 3) + ("x" if r[7] else ""), svv, loss, rel, fmt(r[8], 0),
                                         r[9]]) + " |")
    flagged = [r[0] for r in rows if r[13]]
    lines += ["", f"Loss flags: {', '.join(flagged) if flagged else 'none'}."]
    lines += ["", "## Re-rank rule (memory-bound groups above 5% of kernel time in the best profile)", ""]
    if not rr:
        lines.append("No `04_profile_*_best` profile yet.")
    for m, v in rr.items():
        shares = ", ".join(f"{g} {p:.1f}%" for g, p in v["group_pct"].items())
        lines.append(f"- **{m}** ({v['profile']}): {shares}")
        if v["promoted"]:
            for p in v["promoted"]:
                lines.append(f"  - promote: {p['group']} {p['pct']:.1f}% -> {p['promote']}")
        else:
            lines.append("  - nothing above the threshold: keep the build order")
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------- main

def hardware(py, gpu_visible):
    code = ("import json, torch; p = torch.cuda.get_device_properties(0); "
            "print(json.dumps({'name': p.name, 'capability': [p.major, p.minor], 'sms': p.multi_processor_count, "
            "'memory_gib': p.total_memory / 2**30}))")
    out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=300,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_visible))
    if out.returncode != 0:
        raise SystemExit(f"cannot query the GPU with torch: {out.stderr.strip()[-500:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])

def config_fields(py):
    code = ("import sys, json, dataclasses; sys.path.insert(0, '.'); from config import PretrainConfig; "
            "print(json.dumps([f.name for f in dataclasses.fields(PretrainConfig)]))")
    out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO)
    return set(json.loads(out.stdout.strip().splitlines()[-1])) if out.returncode == 0 else set()

def parse_args(argv=None):
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", required=True, help="physical GPU index or UUID (nvidia-smi numbering)")
    p.add_argument("--gpu-visible", default=None, help="CUDA_VISIBLE_DEVICES for the steps (default: --gpu)")
    p.add_argument("--dry-run", action="store_true", help="local check: S only, small batch, few steps, no sm_89-only steps")
    p.add_argument("--run-id", default=None, help="results/<run_id>/; the same id resumes (default: kern-<host>-gpu<N>[-dry])")
    p.add_argument("--results", default=os.path.join(BENCH, "results"))
    p.add_argument("--models", nargs="+", choices=["S", "B"], default=None, help="default S B (dry run: S)")
    p.add_argument("--batch-s", type=int, default=None, help="per-GPU batch for S (default 256; dry run 32)")
    p.add_argument("--batch-b", type=int, default=None, help="per-GPU batch for B (default 128; dry run 16)")
    p.add_argument("--global-batch", type=int, default=None,
                   help="gradient accumulation = global / batch (default 4096 as config.py; dry run 2 x batch)")
    p.add_argument("--warmup", type=int, default=None, help="untimed steps (default 20; dry run 3)")
    p.add_argument("--steps", type=int, default=None, help="timed steps (default 100; dry run 5)")
    p.add_argument("--profile-steps", type=int, default=None, help="profiled steps (default 10; dry run 3)")
    p.add_argument("--step-timeout", type=int, default=None, help="seconds per step (default 5400; dry run 1200)")
    p.add_argument("--data-root", default=env("DATA_ROOT") or None, help="ImageFolder root for the real-data step")
    p.add_argument("--num-workers", type=int, default=int(env("NUM_WORKERS") or 8))
    p.add_argument("--skip", nargs="*", default=[], metavar="REGEX",
                   help="don't run steps whose id matches, e.g. 'max-autotune\\+cd' to fit a time budget")
    p.add_argument("--preflight-json", default=None, help="kern_preflight.py report (copied into the run dir)")
    p.add_argument("--python", default=sys.executable, help="python of the benchmark env (default: this one)")
    a = p.parse_args(argv)
    dry = a.dry_run
    # bench/launch.sh passes MODELS / WARMUP / STEPS as environment variables
    a.models = a.models or (env("MODELS").split() if env("MODELS") else (["S"] if dry else ["S", "B"]))
    a.batch_s = a.batch_s or (32 if dry else 256)
    a.batch_b = a.batch_b or (16 if dry else 128)
    a.warmup = a.warmup if a.warmup is not None else int(env("WARMUP") or (3 if dry else 20))
    a.steps = a.steps if a.steps is not None else int(env("STEPS") or (5 if dry else 100))
    a.profile_steps = a.profile_steps if a.profile_steps is not None else (3 if dry else 10)
    a.step_timeout = a.step_timeout or (1200 if dry else 5400)
    a.gpu_visible = a.gpu_visible or a.gpu
    a.run_id = a.run_id or f"kern-{socket.gethostname()}-gpu{a.gpu}{'-dry' if dry else ''}"
    if a.data_root:
        a.data_root = os.path.expanduser(a.data_root)
    return a

def main(argv=None):
    args = parse_args(argv)
    run_dir = os.path.join(args.results, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "runner.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    t0 = time.time()
    log(f"===== kern_runner start: run {args.run_id} in {run_dir} (dry run: {args.dry_run}), python {args.python}")
    if args.preflight_json and os.path.exists(args.preflight_json):
        shutil.copy(args.preflight_json, os.path.join(run_dir, f"preflight_gate_{time.strftime('%Y%m%d-%H%M%S')}.json"))
    hw = hardware(args.python, args.gpu_visible)
    fields = config_fields(args.python)
    log(f"GPU {hw['name']}, capability {hw['capability'][0]}.{hw['capability'][1]}, {hw['sms']} SMs, "
        f"{hw['memory_gib']:.0f} GiB; models {args.models}; batch S {args.batch_s} / B {args.batch_b}; "
        f"warmup {args.warmup}, steps {args.steps}; step timeout {args.step_timeout} s")

    plan = build_plan(args.models, args.dry_run)
    runner = Runner(args, hw, args.python, run_dir, log)

    def on_signal(signum, frame): # stop the current step's process group too, then exit; its JSON is never written
        if runner.child is not None:
            kill_group(runner.child)
        log(f"===== interrupted by signal {signum}: rerun with the same --run-id to resume")
        sys.exit(130)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    states = run_plan(plan, run_dir, runner.execute, lambda s: unmet(s, hw, args, fields), resolve, log)

    results = load_results(run_dir)
    rr = rerank(run_dir, args.models)
    write_json(os.path.join(run_dir, "rerank.json"), rr)
    rows = summary_rows(plan, results, states)
    md = summary_markdown(rows, hw, args, rr, time.time() - t0)
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(md)
    columns = ["step", "status", "model", "settings", "ms_per_step", "imgs_per_sec", "peak_gib", "speedup",
               "run_s", "note", "loss", "loss_vs_vanilla", "speedup_vs_vanilla", "loss_flagged"]
    write_json(os.path.join(run_dir, "summary.json"), {"columns": columns, "rows": [dict(zip(columns, r)) for r in rows],
                                                       "rerank": rr, "hardware": hw,
                                                       "best": {m: best_overall(results, m) for m in args.models}})
    print("\n" + md)
    failed = [k for k, (st, _) in states.items() if st == "failed"]
    log(f"===== kern_runner end: {len(failed)} failed step(s) {failed}; summary {os.path.join(run_dir, 'summary.md')}")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
