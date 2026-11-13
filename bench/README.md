# REPA speed benchmarks

Benchmarks for REPA pretraining speed on one GPU. `bench/run_all.sh` times a vanilla baseline (no speed flags) and
then the optional speed flags and settings below, and checks that the loss still matches. Every flag is off by default.

## Setup

Driver 580 or newer (CUDA 13):

```bash
ENV_NAME=repa-kern bash bench/setup_env.sh
```

Driver 525 to 579 (CUDA 12 fallback):

```bash
conda create -y -n repa-kern-cu126 python=3.14.8 pip
conda env config vars set -n repa-kern-cu126 PYTHONNOUSERSITE=1
conda run -n repa-kern-cu126 python -m pip install torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126
conda run -n repa-kern-cu126 python -m pip install pytorch-lightning==2.6.6 lightning==2.6.6 \
    transformers==5.19.0 timm==1.0.30 numpy==2.5.3 pillow==12.3.0 opencv-python-headless==5.0.0.93 \
    wandb==0.30.0 huggingface-hub==1.33.0 pytest torchao==0.18.0
```

You don't pick the env yourself: `run_all.sh` starts with a preflight that reads the driver version, picks
`repa-kern` or `repa-kern-cu126`, and stops with a message if the env is missing, the driver is too old, or the GPU is
busy. It warns if the GPU isn't compute capability 8.9 (RTX 6000 Ada). `ENV_NAME=<env>` overrides the choice.

## Run it

```bash
bash bench/run_all.sh --gpu 0 --data-root /path/to/imagenet --skip gqa --dry-run   # ~10 min check
```

```bash
bash bench/run_all.sh --gpu 0 --data-root /path/to/imagenet --skip gqa             # full, ~2-3 h
```

To smoke-test every step, including FP8 and max-autotune, before the full run:
`bash bench/run_all.sh --gpu 0 --data-root /path/to/imagenet --skip gqa --models S --warmup 3 --steps 5 --global-batch 512 --run-id smoke`

- Resume: run the same command again. Finished steps are skipped, failed ones are retried.
- Results: `bench/results/<run_id>/summary.md` (also `summary.json`, one JSON per step, and logs in `logs/`).
- Data layout: `<root>/train/<wnid>/*.JPEG`. Without `--data-root` the real-data step is skipped.

## Runner flags

| Flag | What it does |
|---|---|
| `--gpu N` | Physical GPU to use, as `nvidia-smi` numbers it. Without it, set `CUDA_VISIBLE_DEVICES` to one GPU yourself. |
| `--force` | Run even if the GPU looks busy (timings will be off). |
| `--data-root DIR` | ImageNet-style folder for the real-data step. Defaults to `$DATA_ROOT`. |
| `--num-workers N` | Dataloader workers for the real-data step (default 8, or `$NUM_WORKERS`). |
| `--dry-run` | Short check: model S only, batch 32, 3 + 5 steps, no FP8 or max-autotune steps. |
| `--skip REGEX ...` | Don't run steps whose name matches, e.g. `gqa` or `max-autotune\+cd`. |
| `--run-id ID` | Name of the results folder. The same ID resumes. Default `kern-<host>-gpu<N>`, plus `-dry`. |
| `--step-timeout S` | Seconds before a step is killed (default 5400, dry run 1200). |
| `--models S B` | Which models to run (default both, dry run S). |
| `--batch-s N`, `--batch-b N` | Per-GPU batch for S and B (default 256 and 128, dry run 32 and 16). |
| `--global-batch N` | Sets gradient accumulation to N / batch (default 4096, dry run 2 x batch). |
| `--warmup N`, `--steps N` | Untimed and timed steps per run (default 20 and 100). |
| `--profile-steps N` | Steps captured in each profile (default 10). |
| `--results DIR` | Parent folder for results (default `bench/results`). |
| `--gpu-visible ID` | `CUDA_VISIBLE_DEVICES` for the steps, if it differs from `--gpu` (default: same). |

`--python` and `--preflight-json` are set by `run_all.sh`; don't pass them.

## Speed flags

Config flags live in `config.py`. `pretrain.py` takes the `perf_*` ones as `--perf name,name` (`all` turns on the
seven Tier 1 flags). The `kern_*` ones are only on the bench scripts' `--kern` for now. Bench-only settings are
applied by the runner and have no config field.

| Flag | What it does | Notes |
|---|---|---|
| `perf_skip_last_content` | In the last layer the content stream only computes K/V, since the rest of its output never reaches the loss. | Config. Same results. |
| `perf_causal_no_mask` | Drops the last permuted patch so both streams use causal SDPA without masks, which lets FlashAttention run. | Config. Needed by `kern_attn_fused`. |
| `perf_query_proj_once` | In layer 1 runs LayerNorm and the Q projection on the one shared query row, then broadcasts. | Config. RoPE models only. |
| `perf_no_cpu_sync` | Masked-mean loss instead of boolean indexing, and `train/lr` logged once per epoch. | Config. Removes GPU waits. |
| `perf_uint8_images` | Dataloader returns uint8 images; scaling and Normalize run on the GPU. | Config. Bit-identical inputs. |
| `perf_fused_adamw` | `AdamW(fused=True)`. | Config. bf16 and fp32 only. |
| `perf_ddp_static_graph` | DDP with `static_graph=True`. | Config. Multi-GPU only. |
| `perf_fp8` | torchao float8 training for the Linear layers. | Config, `--perf fp8`, not in `all`. Needs compute capability 8.9 or newer (Ada/Hopper). Not bit-exact. |
| FP8 recipe | `tensorwise` (default), `rowwise` or `rowwise_with_gw_hp` for `perf_fp8`. | Bench-only. |
| `kern_patch_linear` | Patch embedding as a reshape plus one matmul instead of the Conv2d, same weights. | Config. 0-2% faster in testing. |
| `kern_attn_fused=flex` | Both streams' attention in one FlexAttention call over the shared K/V. | Config. Needs `perf_causal_no_mask` and `torch.compile`. 0-2% slower in testing. |
| `kern_attn_fused=gqa` | Same, as one SDPA call with stacked query heads. | Config. 1-3% slower in testing, skipped with `--skip gqa`. |
| Compile mode | `torch.compile` mode: default, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`, with or without coordinate descent tuning. | Bench-only. max-autotune needs 68+ SMs. |
| SDPA backend | Forces cuDNN or memory-efficient attention instead of the default (FlashAttention). | Bench-only. |

## Reading the summary

- `ms/step`: median time per training step (GPU time per step for profile rows).
- `img/s`: images per second over the timed steps.
- `peak GiB`: peak GPU memory. It undercounts CUDA-graph modes (`reduce-overhead`, `max-autotune`), which the note says.
- `speedup_vs_vanilla`: vanilla's ms/step divided by this row's. (`speedup` is the same against `02_compile_<model>_default`, all Tier 1 flags.)
- `loss_vs_vanilla`: relative difference of the final loss against the vanilla step, same batch, steps and seed. Marked FLAG above 1% (5% for FP8). It checks that the numbers match, not training over time: the learning rate is close to zero in these short runs.

## Tests

```bash
conda run -n repa-kern python -m pytest tests/
```

CPU tests always run. GPU tests run when a GPU is visible; the FP8 ones also need compute capability 8.9.
