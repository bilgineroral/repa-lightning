# REPA pretraining speed benchmarks

Speed flags for pretraining, plus tests and benchmarks to check them. Every flag is off by default, and with all
flags off the code behaves exactly as before.

| flag (`--perf` name) | what it does | config field |
|---|---|---|
| `skip_last_content` | In the last layer, the content stream computes only K/V. Its attention output and MLP never reached the loss. | `perf_skip_last_content` |
| `causal_no_mask` | Drops the last permuted patch from the content stream (no query attends to it), so both streams use SDPA `is_causal=True` with no masks. This lets FlashAttention run. | `perf_causal_no_mask` |
| `query_proj_once` | In layer 1 every query row is `w`, so LayerNorm, the Q projection and QK-norm run on one row and are broadcast. RoPE models only. | `perf_query_proj_once` |
| `no_cpu_sync` | Masked-mean loss instead of boolean indexing; `train/lr` is logged once per epoch (same epoch mean). Both removed a GPU-to-CPU wait per batch. | `perf_no_cpu_sync` |
| `uint8_images` | The dataloader returns uint8 images from pinned memory; `x / 255` and `Normalize` run on the GPU (bit-identical inputs). | `perf_uint8_images` |
| `fused_adamw` | `AdamW(fused=True)`. Lightning refuses to clip gradients for fused optimizers under AMP, so `LitREPA.configure_gradient_clipping` runs the same `clip_grad_norm_` itself. bf16 and fp32 only. | `perf_fused_adamw` |
| `ddp_static_graph` | `DDPStrategy(static_graph=True)`. Multi-GPU only; does nothing on one GPU. | `perf_ddp_static_graph` |
| `fp8` (Tier 2, not in `all`) | torchao float8 training for every `nn.Linear` (default dynamic tensorwise recipe). Needs compute capability >= 8.9 (Ada / Hopper) and refuses otherwise. **Not bit-exact**: the test checks the loss stays within 5% of bf16 over 20 steps. Use `--perf all,fp8`. | `perf_fp8` |

Training with flags: `python pretrain.py --model small --data-root <root> --perf all`
(or a comma list, e.g. `--perf skip_last_content,causal_no_mask`).

Except for `fp8`, the model and loss are unchanged: same parameters, same checkpoints, and old checkpoints still load. The tests
check equal loss and gradients. Equality is exact up to float rounding only with drop path 0 (the default). With
drop path, `skip_last_content` changes the order of random draws.

## On Arcturus, from a fresh login

Run these from the repo root, in order.

**1. Get the code**

```bash
cd ~/repa-lightning && git fetch origin && git checkout repa-speedup && git pull
```

**2. Environment (once).** If a conda env named `repa` already exists, this only reports its versions and changes
nothing. If it's missing packages, or isn't yours, the script prints what to do (for example
`ENV_NAME=repa-bench bash bench/setup_env.sh`, then pass the same `ENV_NAME` to the next steps).

```bash
bash bench/setup_env.sh
```

**3. Preflight.** Reports GPUs (name, count, capability, memory, other processes), torch and CUDA versions, which
attention kernel actually runs for our shapes, whether FP8 is possible, `sbatch`, CPUs and free disk. It also checks
that `DATA_ROOT` has the layout `dataset.py` expects (`<root>/train/<wnid>/*.JPEG` and `<root>/val/<wnid>/*.JPEG`)
and fails with what it found if not. Read the output before launching.

```bash
DATA_ROOT=$HOME/imagenet bash bench/preflight.sh
```

**4. Launch.** Submits `bench/run_all.sh` to Slurm if `sbatch` exists. Otherwise it starts a tmux session (or
nohup) on `GPU=0` and refuses if that GPU is busy. Choose another with `GPU=<id>`. GPU count, time limit and
partition are variables at the top of `bench/launch.sh`. The script prints how to check progress.

```bash
DATA_ROOT=$HOME/imagenet bash bench/launch.sh
```

`run_all.sh` runs, in order:
1. preflight
2. the full pytest suite
3. a batch-size probe for S and B: the config batch (256 / 128), halved until it fits, checked with all flags too.
   Gradient accumulation is set to keep the config's global batch of 4096.
4. the Tier 1 matrix: baseline, each flag and all flags, for synthetic and real data
5. profiles of baseline vs all flags
6. FP8 runs (`--perf fp8` and `all,fp8`), only if preflight found compute capability >= 8.9; otherwise skipped with
   a message

Everything goes to `bench/results/<timestamp>/` (also `bench/results/latest/`), with a log (`run_all.log`), the
preflight JSON, the chosen batch sizes (`batch_sizes.txt`), one JSON per run, profiles and Chrome traces. A failed
step doesn't stop the rest; failures are listed at the end of the log.

**5. Summary**

```bash
cat bench/results/latest/summary.md
```

To regenerate it: `python bench/summarize.py --results bench/results/latest/tier1`.

### Do not

- Never write inside the dataset directories. Nothing in `bench/` writes there: preflight only lists and opens
  files. Don't run `make_fake_imagefolder.py` with a real dataset path.
- Check `nvidia-smi` before launching. Without Slurm, `launch.sh` refuses a busy GPU unless `FORCE=1`, but other
  users' jobs can still start later and slow the timings.
- Don't modify a shared or someone else's `repa` env. Use `ENV_NAME=repa-bench`.
- Don't commit results. `bench/results/` is git-ignored.

## Running parts by hand

```bash
python -m pytest tests/ -v
```

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT=$HOME/imagenet BATCH_S=256 ACCUM_S=16 MODELS=S bash bench/run_tier1.sh
```

```bash
python bench/step_time.py --model S --flags all --data real --data-root $HOME/imagenet --optimizer --accum 16
```

```bash
python bench/profile_step.py --model S --flags all
```

`run_tier1.sh` variables: `MODELS`, `DATAS`, `DATA_ROOT`, `BATCH_S`/`BATCH_B`, `ACCUM` (or `ACCUM_S`/`ACCUM_B`),
`WARMUP` (20, covers `torch.compile`), `STEPS` (100), `NUM_WORKERS` (8), `OUT`. Synthetic runs have no data decode
and no optimizer step, to isolate the model step. Real-data runs use the real dataloader and include the optimizer
step every `ACCUM` micro-batches.

## What the numbers mean

- `ms/step`: time between the starts of consecutive training batches, measured with CUDA events. Includes waiting for
  data, the host-to-device copy, forward, backward and (if enabled) the optimizer step.
- `compute ms`: start to end of one training batch. Excludes the wait for the next batch.
- `img/s`: wall-clock throughput over the timed steps.
- `speedup`: baseline ms/step divided by this row's ms/step, within the same table.
- Profile buckets: GPU kernel time per step in matmul / attention / layernorm / other_elementwise (plus optimizer and
  memcpy). Annotations are excluded.

## Running locally (no ImageNet)

```bash
python bench/make_fake_imagefolder.py --root ~/repa_fake --train 8000 --val 200
```

```bash
MODELS=S STEPS=10 WARMUP=5 DATA_ROOT=~/repa_fake bash bench/launch.sh
```

The fake JPEGs have realistic sizes, so decode cost is similar, but the content is random noise.

## Tier 3

Custom-kernel plans (not implemented): `bench/TIER3.md`.
