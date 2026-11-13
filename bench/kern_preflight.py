"""
Tier 3 runner preflight (called by bench/run_all.sh before any conda env is chosen). Standard library only, so it runs
with any python3. Queries nvidia-smi for the target GPU and decides:
  - compute capability: 8.9 (RTX 6000 Ada) expected; anything else is a warning, not an error
  - driver version -> conda env: >= 580 runs CUDA 13 builds (repa-kern), 525-579 needs CUDA 12 builds (repa-kern-cu126),
    < 525 is unsupported (CUDA 13.0 release notes, see bench/ENV.md)
  - idle / busy: other compute processes on the GPU abort the run. Where nvidia-smi can't list processes (WSL), memory
    in use above --busy-mem-mib or utilization above --busy-util also counts as busy.

    python3 bench/kern_preflight.py --gpu 0 [--json out.json] [--force]
Exit codes: 0 ok, 2 GPU busy (unless --force), 3 no usable GPU / driver / env.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ENV_CUDA13, ENV_CUDA12 = "repa-kern", "repa-kern-cu126"
TARGET_CAPABILITY = "8.9"

def select_env(driver_version: str):
    """ driver version string (e.g. '580.65.06') -> (env name or None, reason) """
    try:
        major = int(driver_version.strip().split(".")[0])
    except (ValueError, AttributeError):
        return None, f"cannot parse driver version {driver_version!r}"
    if major >= 580:
        return ENV_CUDA13, f"driver {driver_version} >= 580: CUDA 13 builds (torch +cu130)"
    if major >= 525:
        return ENV_CUDA12, f"driver {driver_version} is 525-579: CUDA 12 builds (torch +cu126)"
    return None, f"driver {driver_version} < 525: neither CUDA 13 nor CUDA 12.x builds run on it"

def capability_warning(cap: str):
    if cap == TARGET_CAPABILITY:
        return None
    return (f"compute capability {cap}, expected {TARGET_CAPABILITY} (RTX 6000 Ada): sm_89-only steps (FP8, "
            f"max-autotune GEMM templates on GPUs with < 68 SMs) will be skipped and timings are not the target's")

def is_busy(processes, mem_used_mib, util_pct, busy_mem_mib, busy_util):
    """ -> list of reasons the GPU looks busy (empty = idle) """
    reasons = []
    if processes:
        reasons.append(f"{len(processes)} compute process(es): {processes}")
    if mem_used_mib is not None and mem_used_mib > busy_mem_mib:
        reasons.append(f"{mem_used_mib} MiB memory in use > {busy_mem_mib} MiB")
    if util_pct is not None and util_pct > busy_util:
        reasons.append(f"utilization {util_pct}% > {busy_util}%")
    return reasons

def _smi(args, timeout=30):
    out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi {' '.join(args)} failed: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout

def _num(x):
    try:
        return float(x)
    except ValueError:
        return None

def query_gpu(gpu: str, samples: int = 3):
    """ one physical GPU (nvidia-smi index or UUID) -> dict; utilization and memory are the max over a few samples """
    fields = "index,uuid,name,compute_cap,driver_version,memory.used,memory.total,utilization.gpu"
    rows = []
    for i in range(samples):
        line = _smi(["--id=" + gpu, "--query-gpu=" + fields, "--format=csv,noheader,nounits"]).strip().splitlines()[0]
        rows.append([x.strip() for x in line.split(",")])
        if i + 1 < samples:
            time.sleep(0.5)
    r = rows[0]
    info = {"index": r[0], "uuid": r[1], "name": r[2], "compute_cap": r[3], "driver_version": r[4],
            "memory_total_mib": _num(r[6]),
            "memory_used_mib": max((_num(x[5]) for x in rows if _num(x[5]) is not None), default=None),
            "utilization_pct": max((_num(x[7]) for x in rows if _num(x[7]) is not None), default=None)}
    apps = _smi(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"])
    info["processes"] = [l.strip() for l in apps.splitlines() if l.strip() and l.split(",")[0].strip() == info["uuid"]]
    return info

def conda_env_python(name: str):
    """ python of conda env `name`, or None. Uses `conda env list` if conda is on PATH, else common install paths """
    candidates = []
    if shutil.which("conda"):
        try:
            out = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, timeout=60).stdout
            for l in out.splitlines():
                parts = l.split()
                if parts and parts[0] == name:
                    candidates.append(os.path.join(parts[-1], "bin", "python"))
        except Exception:
            pass
    for base in ("~/miniconda3", "~/anaconda3", "~/miniforge3", "/opt/conda"):
        candidates.append(os.path.join(os.path.expanduser(base), "envs", name, "bin", "python"))
    return next((c for c in candidates if os.path.exists(c)), None)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", required=True, help="physical GPU index (as nvidia-smi numbers it) or UUID")
    p.add_argument("--json", default=None, help="write the report here")
    p.add_argument("--force", action="store_true", help="continue on a busy GPU (timings will be affected)")
    p.add_argument("--busy-mem-mib", type=float, default=2048,
                   help="memory in use above this counts as busy (catches WSL, where processes aren't listed)")
    p.add_argument("--busy-util", type=float, default=20, help="utilization %% above this counts as busy")
    p.add_argument("--env-name", default=os.environ.get("ENV_NAME") or None,
                   help="use this conda env instead of the driver-based choice")
    args = p.parse_args(argv)

    report = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpu_arg": args.gpu, "warnings": [], "problems": []}
    try:
        info = query_gpu(args.gpu)
    except Exception as e:
        report["problems"].append(f"cannot query GPU {args.gpu}: {e}")
        return _finish(report, args, 3)
    report["gpu"] = info

    w = capability_warning(info["compute_cap"])
    if w:
        report["warnings"].append(w)

    env, reason = select_env(info["driver_version"])
    report["env_reason"] = reason
    if args.env_name:
        report["env_reason"] += f"; overridden by ENV_NAME / --env-name to {args.env_name}"
        env = args.env_name
    report["env"] = env
    if env is None:
        report["problems"].append(reason)
        return _finish(report, args, 3)
    py = conda_env_python(env)
    report["python"] = py
    if py is None:
        report["problems"].append(f"conda env '{env}' not found: create it as in bench/ENV.md")
        return _finish(report, args, 3)

    busy = is_busy(info["processes"], info["memory_used_mib"], info["utilization_pct"], args.busy_mem_mib, args.busy_util)
    report["busy"] = busy
    if busy:
        msg = f"GPU {args.gpu} looks busy: " + "; ".join(busy)
        if args.force:
            report["warnings"].append(msg + " (continuing: --force)")
        else:
            report["problems"].append(msg + " (pick another GPU or pass --force)")
            return _finish(report, args, 2)
    return _finish(report, args, 0)

def _finish(report, args, code):
    report["exit_code"] = code
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    g = report.get("gpu", {})
    if g:
        print(f"kern_preflight: GPU {g['index']} {g['name']}, capability {g['compute_cap']}, driver {g['driver_version']}, "
              f"{g['memory_used_mib']:.0f}/{g['memory_total_mib']:.0f} MiB used, util {g['utilization_pct']:.0f}%, "
              f"{len(g['processes'])} process(es) listed")
    if report.get("env"):
        print(f"kern_preflight: env {report['env']} ({report['env_reason']}) -> {report.get('python')}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    for x in report["problems"]:
        print(f"FAIL: {x}")
    print("kern_preflight:", {0: "OK", 2: "BUSY", 3: "FAILED"}[code])
    return code

if __name__ == "__main__":
    sys.exit(main())
