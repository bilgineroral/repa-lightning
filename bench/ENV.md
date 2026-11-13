# Benchmark envs: `repa-kern` and `repa-kern-cu126`

`bench/run_all.sh` (the Tier 3 runner) runs in one of two conda envs. Its preflight (`bench/kern_preflight.py`) reads
the NVIDIA driver version and picks the env automatically:

| driver | CUDA builds | env |
|---|---|---|
| ≥ 580 | CUDA 13 (torch `+cu130`) | `repa-kern` |
| 525–579 | CUDA 12 (torch `+cu126`) | `repa-kern-cu126` |
| < 525 | none | preflight fails |

CUDA 13.0 GA needs driver ≥ 580.65.06 on Linux. CUDA 12.x minor-version compatibility covers 525 up to, but not
including, 580 (CUDA 13.0 release notes, https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).
`ENV_NAME=<env>` overrides the choice.

## What's in them

| package | version | source |
|---|---|---|
| python | 3.14.8 | `environment.yml` |
| torch / torchvision | 2.14.1 / 0.29.1 (`+cu130`, or `+cu126` in the fallback) | `environment.yml` (fallback: the cu126 index) |
| triton | 3.8.0 | installed as a torch dependency (FlexAttention, Inductor max-autotune) |
| pytorch-lightning, transformers, timm, … | as pinned | `environment.yml` |
| torchao | 0.18.0 | `bench/setup_env.sh` (FP8: `perf_fp8`, recipes) |
| pytest | 9.2.0 | `bench/setup_env.sh` |
| nvidia-dali | 2.4.0 (`cuda130`, or `cuda120` in the fallback) | optional, for the planned `kern_gpu_decode`; the runner doesn't need it yet |

Not included:
- Transformer Engine: builds from source against a CUDA toolkit; only needed if torchao FP8 is below 1.05x on B.
- flash-attn: FA2 is already inside SDPA, and FA3/FA4 don't support sm_89.
- Liger, Apex, xFormers: rejected in `bench/KERNEL_PLAN.md` §6.

## 1. Driver check first

On a GPU node:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
```

Record the output with the benchmark results. `bench/run_all.sh` checks it again, and so does `bench/preflight.sh`,
which prints the `nvidia-smi` header.

## 2a. Driver ≥ 580: `repa-kern` (from the repo root)

```bash
ENV_NAME=repa-kern bash bench/setup_env.sh
```

Optional, for GPU JPEG decoding experiments (the DALI wheel must match torch's CUDA major version):

```bash
conda run -n repa-kern python -m pip install nvidia-dali-cuda130==2.4.0
```

## 2b. Driver 525–579: `repa-kern-cu126`

Same packages as `environment.yml`, except that torch and torchvision come from the cu126 index. torch 2.14.1 and
torchvision 0.29.1 have `+cu126` wheels for cp314; there are no cu128/cu129 builds of these versions.

```bash
conda create -y -n repa-kern-cu126 python=3.14.8 pip
```

```bash
conda env config vars set -n repa-kern-cu126 PYTHONNOUSERSITE=1
```

```bash
conda run -n repa-kern-cu126 python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

```bash
conda run -n repa-kern-cu126 python -m pip install pytorch-lightning==2.6.6 lightning==2.6.6 transformers==5.19.0 timm==1.0.30 numpy==2.5.3 pillow==12.3.0 opencv-python-headless==5.0.0.93 wandb==0.30.0 huggingface-hub==1.33.0 pytest torchao==0.18.0
```

Optional:

```bash
conda run -n repa-kern-cu126 python -m pip install nvidia-dali-cuda120==2.4.0
```

Check that the second install did not replace torch. It must still report `2.14.1+cu126`:

```bash
conda run -n repa-kern-cu126 python -c "import torch, torchao; print(torch.__version__, torch.version.cuda, torchao.__version__)"
```

torchao's wheel is CUDA-agnostic, but its FP8 path depends on torch's CUDA build. Untested with cu126: confirm with the
tests below on the node.

## 3. Verify

```bash
conda run -n repa-kern python -m pytest -q tests/
```

(`-n repa-kern-cu126` for the fallback.) On a GPU with compute capability ≥ 8.9 the FP8 tests run too; on older GPUs
they are skipped.

## 4. Use

`bench/run_all.sh` picks the env itself (above). The older scripts read `ENV_NAME` (`bench/env.sh`):

```bash
ENV_NAME=repa-kern DATA_ROOT=$HOME/imagenet bash bench/preflight.sh
```

In its output, check that the GPUs report capability 8.9 and 48 GB (RTX 6000 Ada), and that `fp8_possible` is true.
