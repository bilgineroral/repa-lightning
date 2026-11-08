"""
Environment checks before a benchmark run (called by bench/preflight.sh). Reports what the machine has; assumes nothing.

    python bench/preflight.py [--data-root /path/to/imagenet] [--json out.json]

Exit code 1 if something the run needs is missing (no CUDA GPU, or --data-root given but not laid out as dataset.py
expects); 0 otherwise. Warnings (other processes on the GPU, a subset with fewer images) don't fail.
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# what prepare_imagenet.py writes (its lines 5-6 and NUM_IMAGES) and dataset.py reads: <root>/<split>/<wnid>/*.JPEG
EXPECTED = {"train": 1_281_167, "val": 50_000}
EXPECTED_CLASSES = 1000
IMAGE_EXT = (".jpeg", ".jpg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp") # torchvision IMG_EXTENSIONS

def gpu_info(torch):
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        out.append({"index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
                    "memory_gib": round(total / 2**30, 1), "free_gib": round(free / 2**30, 1), "sms": p.multi_processor_count})
    return out

def gpu_processes():
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        me = str(os.getpid()) # this script itself holds a CUDA context by now
        return [l.strip() for l in q.stdout.splitlines() if l.strip() and l.split(",")[1].strip() != me]
    except Exception as e:
        return [f"nvidia-smi failed: {e}"]

def sdpa_backends(torch):
    """ which SDPA backends accept our attention calls (fwd + bwd, bf16), and which kernel runs by default """
    import warnings
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    warnings.filterwarnings("ignore", message=".*(kernel not used|not support|runtime disabled).*")
    from torch.profiler import ProfilerActivity, profile

    def call(shape, kind):
        B, H, Nq, Nk, d = shape
        q = torch.randn(B, H, Nq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, Nk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, Nk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        if kind == "causal":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else: # the repo's dense boolean masks (modules.py _make_attn_masks)
            mask = torch.tril(torch.ones(Nq, Nk, dtype=torch.bool, device="cuda"))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
        o.sum().backward()

    def kernel_family(names):
        n = " ".join(names).lower()
        for fam, keys in (("flash", ("flash",)), ("cudnn", ("cudnn",)), ("mem_efficient", ("fmha", "efficient", "mem_eff")),
                          ("math", ("softmax", "bmm", "gemm"))):
            if any(k in n for k in keys):
                return fam
        return "unknown"

    cases = {}
    for model, H in (("S", 6), ("B", 12)):
        cases[f"{model} baseline content (mask 197x197)"] = ((8, H, 197, 197, 64), "mask")
        cases[f"{model} baseline query (mask 196x197)"] = ((8, H, 196, 197, 64), "mask")
        cases[f"{model} causal_no_mask (is_causal 196)"] = ((8, H, 196, 196, 64), "causal")
    backends = {"flash": SDPBackend.FLASH_ATTENTION, "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
                "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}
    report = {}
    for name, (shape, kind) in cases.items():
        accepted = []
        for bname, b in backends.items():
            try:
                with sdpa_kernel([b]):
                    call(shape, kind)
                accepted.append(bname)
            except Exception:
                pass
        with profile(activities=[ProfilerActivity.CUDA]) as prof: # default dispatch, as in training
            call(shape, kind)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
                 and not getattr(e, "is_user_annotation", False)]
        attn = [n for n in names if any(k in n.lower() for k in ("flash", "fmha", "attention", "attn", "cudnn"))]
        report[name] = {"accepted_backends": accepted, "default_kernel_family": kernel_family(attn or names),
                        "default_attention_kernels": sorted(set(attn))[:4]}
    return report

def check_data(root):
    """ dataset.py: ImageFolder at <root>/<split>; pretrain.py reads train and val """
    problems, warnings, found = [], [], {"root": root}
    if not os.path.isdir(root):
        return [f"DATA_ROOT {root!r} is not a directory"], warnings, found
    found["top_level"] = sorted(os.listdir(root))[:20]
    for split in ("train", "val"):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            problems.append(f"missing {d}/ (expected <root>/{split}/<wnid>/*.JPEG; found at top level: {found['top_level']})")
            continue
        classes = sorted(e.name for e in os.scandir(d) if e.is_dir())
        loose = [e.name for e in os.scandir(d) if e.is_file()][:5]
        n_images, bad_ext, sample = 0, 0, None
        for c in classes:
            for e in os.scandir(os.path.join(d, c)):
                if e.is_file():
                    if e.name.lower().endswith(IMAGE_EXT):
                        n_images += 1
                        sample = sample or e.path
                    else:
                        bad_ext += 1
        found[split] = {"classes": len(classes), "images": n_images, "non_image_files": bad_ext,
                        "loose_files_in_split_dir": loose, "first_classes": classes[:3]}
        if not classes:
            problems.append(f"{d}/ has no class folders (expected {EXPECTED_CLASSES} <wnid>/ folders); loose files: {loose}")
            continue
        if n_images == 0:
            problems.append(f"{d}/ has {len(classes)} class folders but no image files")
            continue
        if len(classes) != EXPECTED_CLASSES:
            warnings.append(f"{split}: {len(classes)} class folders, ImageNet has {EXPECTED_CLASSES}")
        if n_images != EXPECTED[split]:
            warnings.append(f"{split}: {n_images:,} images, full ImageNet has {EXPECTED[split]:,} (fine for a subset)")
        try:
            from PIL import Image
            with Image.open(sample) as im: # read-only
                im.verify()
            found[split]["sample_ok"] = sample
        except Exception as e:
            problems.append(f"{split}: cannot read sample image {sample}: {e}")
    if "train" in found and "val" in found and isinstance(found["train"], dict) and isinstance(found["val"], dict):
        if found["train"]["classes"] != found["val"]["classes"]:
            warnings.append(f"train has {found['train']['classes']} classes but val has {found['val']['classes']}")
    return problems, warnings, found

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--skip-sdpa", action="store_true")
    args = p.parse_args()

    result, problems, warnings = {}, [], []
    try:
        import torch
    except Exception as e:
        print(f"FAIL: cannot import torch: {e}")
        return 1
    result["python"] = sys.version.split()[0]
    result["torch"] = torch.__version__
    result["cuda_build"] = torch.version.cuda
    result["cudnn"] = torch.backends.cudnn.version()
    for m in ("pytorch_lightning", "transformers", "timm", "torchao", "pytest"):
        try:
            result[m] = __import__(m).__version__
        except Exception as e:
            result[m] = f"MISSING ({type(e).__name__})"
            (problems if m == "pytorch_lightning" else warnings).append(f"{m} not importable")

    if not torch.cuda.is_available():
        problems.append("no CUDA GPU visible to torch (check CUDA_VISIBLE_DEVICES / that you are on a GPU node)")
    else:
        result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
        result["gpus"] = gpu_info(torch)
        result["gpu_processes"] = gpu_processes()
        cap = torch.cuda.get_device_capability(0)
        result["fp8_possible"] = cap >= (8, 9)
        result["fp8_note"] = (f"GPU 0 capability {cap[0]}.{cap[1]}: FP8 (torchao float8) "
                              + ("possible" if cap >= (8, 9) else "needs >= 8.9 (Ada / Hopper or newer); FP8 runs will be skipped"))
        if result["gpu_processes"]:
            warnings.append(f"other processes are using GPUs: {result['gpu_processes']}")
        if not args.skip_sdpa:
            try:
                result["sdpa"] = sdpa_backends(torch)
            except Exception as e:
                warnings.append(f"SDPA backend check failed: {e}")

    if args.data_root:
        dp, dw, found = check_data(os.path.expanduser(args.data_root))
        result["data"] = found
        problems += [f"DATA_ROOT: {x}" for x in dp]
        warnings += [f"DATA_ROOT: {x}" for x in dw]
    else:
        warnings.append("DATA_ROOT not set: real-data runs will be skipped")

    result["problems"], result["warnings"] = problems, warnings
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    for w in warnings:
        print(f"WARNING: {w}")
    for x in problems:
        print(f"FAIL: {x}")
    print("preflight:", "FAILED" if problems else "OK")
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
