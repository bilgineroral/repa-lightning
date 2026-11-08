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

Training with flags: `python pretrain.py --model small --data-root <root> --perf all`
(or a comma list, e.g. `--perf skip_last_content,causal_no_mask`).

The model and loss are unchanged: same parameters, same checkpoints, and old checkpoints still load. The tests
check equal loss and gradients. Equality is exact up to float rounding only with drop path 0 (the default). With
drop path, `skip_last_content` changes the order of random draws.

## Running on Arcturus

### 1. Get the branch and set up the environment

```bash
cd ~/repa-lightning
git fetch origin
git checkout repa-speedup
git log -1 --oneline
```

```bash
conda env create -f environment.yml   # once; or reuse an existing env made from this file
conda activate repa
pip install pytest torchao            # pytest for the tests, torchao for the FP8 flag
```

Record what the machine actually has:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.device_count())"
```

```bash
nproc && nvidia-smi
```

### 2. Equivalence tests (a few minutes)

```bash
python -m pytest tests/test_perf_equivalence.py -v
```

The run ends with a "max differences vs. baseline" table. GPU tests are skipped when no GPU is visible.

### 3. Tier 1 run matrix

Real-data runs need an ImageFolder root with `train/`. Use one GPU so results are comparable:

```bash
CUDA_VISIBLE_DEVICES=0 DATA_ROOT=$HOME/imagenet bash bench/run_tier1.sh
```

This runs baseline, each flag alone, and all flags, for models S and B, on synthetic and real data. Each run adds
a JSON file to `bench/results/`, and the script ends by writing `bench/results/summary.md`.

- Synthetic data: no decode cost and no optimizer step (`--no-optimizer`), to isolate the model step.
- Real data: the real dataloader (8 workers by default) plus an optimizer step every micro-batch (`ACCUM=1`).
  Pretraining steps the optimizer every 8 to 16 micro-batches, so `ACCUM=1` overstates optimizer cost. Use
  `ACCUM=8` for a closer match.
- Defaults are batch 256 (S) and 128 (B), the config values. Lower them if memory runs out: `BATCH_S=128 BATCH_B=64`.
- Subsets: `MODELS=S DATAS=real ...`. Steps and warmup: `STEPS=100 WARMUP=20` (the warmup covers `torch.compile`).
- `ddp_static_graph` does nothing on one GPU, so its row is expected to match the baseline.

Single runs:

```bash
python bench/step_time.py --model S --flags all --data real --data-root $HOME/imagenet --optimizer
```

```bash
python bench/summarize.py
```

### 4. Profiling (baseline vs. all flags)

```bash
python bench/profile_step.py --model S --flags none
```

```bash
python bench/profile_step.py --model S --flags all
```

Each run prints GPU time per step in buckets (matmul / attention / layernorm / other elementwise, plus optimizer and
memcpy) and saves a JSON file and a Chrome trace to `bench/results/profiles/`. Open the trace in https://ui.perfetto.dev.

### 5. FP8 (Tier 2)

Not in this commit yet. It comes after the Tier 1 run.

## What the numbers mean

- `ms/step`: time between the starts of consecutive training batches, measured with CUDA events. Includes waiting for
  data, the host-to-device copy, forward, backward and (if enabled) the optimizer step.
- `compute ms`: start to end of one training batch. Excludes the wait for the next batch.
- `img/s`: wall-clock throughput over the timed steps.
- `speedup`: baseline ms/step divided by this row's ms/step, within the same table.

## Running locally (no ImageNet)

```bash
python bench/make_fake_imagefolder.py --root ~/repa_fake --train 8000 --val 200
```

```bash
MODELS=S BATCH_S=64 DATA_ROOT=~/repa_fake bash bench/run_tier1.sh
```

The fake JPEGs have realistic sizes, so decode cost is similar, but the content is random noise.
