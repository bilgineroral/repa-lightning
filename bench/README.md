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

## Architecture flags (these change the model)

Speed-only benchmarks of the report's alternatives. They **change the model**, so there are no equivalence tests. The
sanity tests check that each setting builds, runs, gives every parameter a gradient, and lowers the loss over 50 steps
on a tiny batch (`tests/test_arch_sanity.py`). Accuracy needs pilot runs. Set them with `--arch` in `pretrain.py` or
the bench scripts, comma separated `name[=value]`. Config fields are `arch_<name>`, all off by default.

| `--arch` | what changes | FLOP estimate |
|---|---|---:|
| `single_stream_ref` | no query stream: the content stream alone, square causal over the permuted order (reference for the two-stream cost) | 1.83 |
| `query_top_k=K` | query stream only in the top K layers; it starts as `w` at layer L-K (K = 6, 4) | 1.36, 1.52 |
| `single_stream_target_pos` | option B, as in `~/repa-notes/option_b_design.md`: single stream, row j gets the target position `e_tgt[pi_j]` added. Options: `target_pos_type=learned\|sincos`, `target_pos_layers=k` (add before each of the last k layers instead of at the input), `target_pos_null_p=p` (null token on a fraction of rows, excluded from the loss) | 1.83 |
| `query_mlp_width=W` | separate query-stream MLP of width W (`2d`, `d`, or a number) | 1.26, 1.40 |
| `train_res=160` | pretrain at 160 px (100 patches). RoPE coordinates are normalized, so no other change. | ~1.6 (schedule) |
| `predict_frac=F` | predict only the last F of each order; query rows for the other targets aren't computed (F = 0.5, 0.25) | 1.36, 1.61 |

FLOP estimates are the report's ratios to the two-stream baseline, not step-time predictions. `train_res`'s 1.6 is
for a 75% / 25% 160 to 224 px schedule; one 160 px step is 2.12x (S) in `flop_model.py`.

The single-stream modes set `use_query_stream=False` themselves (in `pretrain.py` and the bench). The two-stream perf
flags (`skip_last_content`, `causal_no_mask`, `query_proj_once`) do nothing in single-stream mode. Option B always
uses `is_causal`. Not implemented from the option B design: injecting the null token in fine-tuning
(`classifier.py` / `segmentation.py`, its step 6).

```bash
MODELS=S BATCH_S=128 ACCUM_S=32 bash bench/run_arch.sh
```

This runs each setting alone and with all Tier 1 perf flags, plus both baselines, on synthetic data with the
optimizer on. `run_tier12.sh` runs it after the Tier 1 matrix; results go to `bench/results/<timestamp>/arch/` and
`summary_arch.md`.

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

`launch.sh` starts `run_all.sh`, the Tier 3 runner (next section). The Tier 1/2 run that used to be `run_all.sh` is
now `bench/run_tier12.sh`; start it by hand (`DATA_ROOT=$HOME/imagenet bash bench/run_tier12.sh`). It runs, in order:
1. preflight
2. the full pytest suite
3. a batch-size probe for S and B: the config batch (256 / 128), halved until it fits, checked with all flags too.
   Gradient accumulation is set to keep the config's global batch of 4096.
4. the Tier 1 matrix: baseline, each flag and all flags, for synthetic and real data
5. profiles of baseline vs all flags
6. FP8 runs (`--perf fp8` and `all,fp8`), only if preflight found compute capability >= 8.9; otherwise skipped with
   a message
7. the architecture-flag runs (`bench/run_arch.sh`)

Everything goes to `bench/results/<timestamp>/` (also `bench/results/latest/`), with a log (`run_tier12.log`), the
preflight JSON, the chosen batch sizes (`batch_sizes.txt`), one JSON per run, profiles and Chrome traces. A failed
step doesn't stop the rest; failures are listed at the end of the log.

**5. Summary**

```bash
cat bench/results/latest/summary.md
```

```bash
cat bench/results/latest/summary_arch.md
```

To regenerate it: `python bench/summarize.py --results bench/results/latest/tier1`.

### Do not

- Never write inside the dataset directories. Nothing in `bench/` writes there: preflight only lists and opens
  files. Don't run `make_fake_imagefolder.py` with a real dataset path.
- Check `nvidia-smi` before launching. Without Slurm, `launch.sh` refuses a busy GPU unless `FORCE=1`, but other
  users' jobs can still start later and slow the timings.
- Don't modify a shared or someone else's `repa` env. Use `ENV_NAME=repa-bench`.
- Don't commit results. `bench/results/` is git-ignored.

## Tier 3 runner (`run_all.sh`)

The benchmark sequence of `~/repa-notes/kernel_research.md` section 5, on one GPU:

```bash
bash bench/run_all.sh --gpu 0 --data-root $HOME/imagenet
```

```bash
bash bench/run_all.sh --gpu 0 --dry-run --data-root ~/repa_fake --num-workers 4
```

1. `bench/kern_preflight.py` (no conda env needed) reads the GPU through `nvidia-smi`.
   - Compute capability 8.9 (RTX 6000 Ada) is expected; anything else is a warning.
   - The driver version picks the conda env: >= 580 uses `repa-kern` (CUDA 13), 525-579 uses `repa-kern-cu126`. See `~/repa-notes/env/README.md`. `ENV_NAME=...` overrides the choice.
   - It refuses a busy GPU (other processes, memory in use > 2 GiB, or utilization > 20%) unless `--force`.
2. `bench/kern_runner.py` runs the steps with that env's python and `CUDA_VISIBLE_DEVICES=<gpu>`:
   - `00` torch preflight
   - `01` profile, default compile
   - `02` compile modes: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs, and coordinate descent
   - `03` FP8 recipes, with default compile and with the best mode
   - `04` profile with the best settings, plus the re-rank rule
   - `05` SDPA cudnn / efficient, plus `kern_patch_linear` / `kern_attn_fused` once `config.py` has them
   - `06` real vs synthetic data

   Compile mode, FP8 recipe and SDPA backend are applied by `bench/kern_step.py` without changing the model code. Each step gets its own Inductor cache, because the FX graph cache ignores the SDPA toggles.
3. Results go to `bench/results/<run_id>/<step>.json`, together with nvidia-smi and system-load snapshots from before and after the step. A rerun with the same `--run-id` skips finished steps; the default id is fixed per host, GPU and dry/full. Failed steps write `<step>.failed.json` and are retried next time.
4. Each step has a timeout: `--step-timeout`, default 5400 s (dry run 1200 s).
5. At the end, `summary.md` / `summary.json` and `rerank.json` are written to the run directory.

`--dry-run` is for the local 3060: S only, batch 32, 3 + 5 steps. It always skips the sm_89-only steps (FP8, max-autotune GEMM templates). Full-run steps that the hardware can't do (capability < 8.9, fewer than 68 SMs, no `--data-root`) are listed as "not run" and no JSON is written for them.

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
