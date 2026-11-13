# Kernel research for REPA pretraining (Tier 3 follow-up)

Research only: no repo edits, no GPU runs. Date: 2026-11-11. Repo: `~/repa-lightning`, branch `repa-speedup` (HEAD `6d36fd0`).

**Citation key.**
- **P** = `bench/results/latest/profiles/S_synthetic_all_b128_GeForce-RTX-3060_20261108-012048.json`, plus its `.trace.json`. REPA-S, batch 128, all Tier 1 flags, bf16, `torch.compile` default mode, synthetic data, 10 steps, RTX 3060.
- **T1** = `bench/results/latest/tier1/summary.md`.
- **CPU-check** = a CPU-only script I ran in this session. Each one is described where it's used.
- "assumption" = not verified by a source or a measurement.

**Target GPU on Arcturus: 4x NVIDIA RTX 6000, assumed Ada (sm_89), 48 GB. Verify with `bench/preflight.sh`.**
- The name is ambiguous: an older Turing "Quadro RTX 6000" (sm_75, 24 GB) also exists (assumption). 48 GB is the Ada card. Preflight's `capability` field settles it: it must read 8.9.
- RTX 6000 Ada specs (from the search results; the NVIDIA datasheet text could not be read directly):
  - 142 SMs (one aggregator; matches NVIDIA's "142 … RT Cores" core breakdown)
  - 18,176 CUDA cores, 568 Tensor Cores
  - 48 GB GDDR6, 384-bit, **960 GB/s**
  - Sources: NVIDIA datasheet (https://www.nvidia.com/content/dam/en-zz/pt_br/data-sheet/proviz-print-rtx6000-datasheet-web-ptbr.pdf; HP mirror https://valuehub.hp.com/static/nvidia/docs/NVIDIA%20RTX%206000%20Ada%20Generation%20Datasheet.pdf), TechPowerUp (https://www.techpowerup.com/gpu-specs/rtx-6000-ada-generation.c3933)
- Tensor throughput: dense BF16 about 360 TFLOPS and dense FP8 about 730 TFLOPS. Both are assumptions: derived from a remembered datasheet figure of 1,457 TFLOPS FP8 with sparsity, not read from a source this session.
- NVLink: assumption that the Ada card has none, so DDP all-reduce goes over PCIe.

**Driver check (preflight).**
- Record the driver version: `nvidia-smi --query-gpu=driver_version --format=csv,noheader`. `bench/preflight.sh:19` already prints the `nvidia-smi` header, which includes it.
- Our env uses torch 2.14.1+cu130 (`environment.yml`; preflight.json `cuda_build` 13.0). CUDA 13.x needs a Linux driver ≥ 580 (CUDA 13.0 GA: ≥ 580.65.06); CUDA 12.x minor-version compatibility covers 525 up to, but not including, 580 (CUDA 13.0 release notes, https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html; also the 13.0 Update 2 notes, https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html).
- If Arcturus reports a driver below 580, use the cu126 fallback env (`bench/ENV.md`).

**What the target changes, from source.**
- **max-autotune works.** Inductor's GEMM templates need `min_sms = … 68` (`torch/_inductor/utils.py:1803`, installed torch 2.14.1). 142 ≥ 68. This is the check that disabled it on the 3060 (28 SMs; TIER3.md:36-37).
- **FP8 works.** `fp8_supported()` requires capability ≥ (8, 9) (`repa.py:36-38`). TE: "FP8 training on NVIDIA Hopper, Ada, Blackwell, and Rubin GPUs" (https://github.com/NVIDIA/TransformerEngine).
- **FlashAttention-3 is out.** It requires "H100 / H800 GPU" (https://github.com/Dao-AILab/flash-attention). FA2 supports "Ampere, Ada, or Hopper GPUs" (same README). SDPA's built-in flash backend is FA2-derived (`pytorch_flash::` kernels, preflight.json).

---

## 1. Summary

1. On the 3060 the step is GEMM-bound and the GEMMs already run near peak. The GEMMs do 5.65 TFLOP per step (CPU-check) in 218 ms (P), about 26 TFLOP/s against a derived peak of about 25–27 (assumption). Memory-bound kernels are 31% of GPU time and run at DRAM bandwidth.
2. On the RTX 6000 Ada the step becomes memory-bound (projection, assumption).
   - Tensor compute is about 14x the 3060's, but bandwidth only about 2.7x (960 vs ~360 GB/s).
   - Projected per GPU at b128 for S: GEMMs about 16–25 ms, memory-bound kernels about 45 ms, attention about 10 ms.
   - Fusing memory-bound ops matters most. FP8 speeds only the smaller GEMM part.
3. **max-autotune (rank 1) is now testable and is the cheapest fusion.** Inductor fuses pointwise ops (GELU, bias, LayerScale, casts) into its Triton GEMM epilogues (`torch/_inductor/config.py:278`). Measure it before writing any custom kernel.
4. **FP8 (rank 2) is now testable** with the existing `perf_fp8` flag. Its gain is capped by the GEMM share of the step (about 25% on the target): even a perfect 2x on GEMMs gives at most 1/(0.75 + 0.125) ≈ 1.14x, minus the added memory-bound cast/amax kernels. **Expect ≤ 1.0x for REPA-S**; a modest gain is possible only for B (assumption, consistent with the torchao docs).
5. **Loader, per GPU:** about 1,500–1,800 img/s per GPU on the target (projection). The supervisor confirmed CPU cores are not a constraint. My 381 img/s per core came from about 67 KB fake JPEGs, so real ImageNet throughput per core is lower. Measure real-vs-synthetic on one GPU first. Options: `kern_gpu_decode`, or a pre-resized ImageNet copy (supervisor decision, §4.6).
6. Two-stream attention in one call is exact via SDPA `enable_gqa` or FlexAttention (CPU-check). Expected gain is small (≤3%).
7. **Build order:**
   1. `kern_compile_mode` (max-autotune)
   2. FP8 recipes (existing flag)
   3. `kern_patch_linear`
   4. `kern_attn_fused`
   5. `kern_sdpa_backend`
   6. `kern_gpu_decode`
   7. `kern_gelu_epilogue` (only if rank 1 doesn't fuse GELU)
   8. `kern_stacked_streams`
   9. `kern_ln_prologue`
8. Combined: about 1.05–1.12x on the 3060. On the RTX 6000 Ada, 1.2–1.4x per GPU is plausible, mostly from fusing memory-bound work (max-autotune, then ranks 7–9). FP8 adds at most about 1.14x before cast overhead, and likely nothing for S (assumption; the first target profile decides).

## 2. Current step breakdown (after all Tier 1 flags)

Source: P. GPU span 388.6 ms/step, busy 385.5 ms, **idle 0.81%**. 1688 kernels per step, mean 228 µs each (from the trace).

| bucket (P) | ms/step | % of kernel time |
|---|---:|---:|
| matmul (cuBLAS/CUTLASS GEMMs + patch-embed conv) | 218.0 | 56.6 |
| other elementwise | 60.8 | 15.8 |
| layernorm (Inductor kernels that fuse LN with residual, LayerScale, casts, QK-norm, RoPE) | 59.0 | 15.3 |
| attention (FA2 inside SDPA) | 45.7 | 11.9 |
| memcpy | 1.9 | 0.5 |

Finer groups across buckets, from my aggregation of the trace:

| group | ms/step | % | main kernels (P `top_kernels`) |
|---|---:|---:|---|
| GELU fwd + bwd | 30.3 | 7.9 | `triton_poi_fused_gelu_view_24` 10.2 ms; `triton_poi_fused_gelu_gelu_backward_view_8` 15.1 ms; `triton_red_fused_gelu_gelu_backward_sum_view_9` 5.0 ms (23 calls each) |
| column reductions (LayerScale / bias / LN-weight grads) | 18.9 | 4.9 | `triton_red_fused_mul_sum_view_3` 8.1 ms (45 calls); `..._to_copy_mul_sum_view_5` 3.8 ms (68 calls) |
| attention, split | 45.7 | 11.9 | bwd `flash_bwd_dq_dk_dv_loop_seqk_parallel` 24.6; fwd `flash_fwd_kernel` 10.5; `flash_bwd_dot_do_o` 6.0; `flash_bwd_convert_dq` 4.6 (23 calls = 12 query + 11 content) |
| patch embedding (Conv2d k=16 s=16) | ≈8.3 | 2.1 | `implicit_convolve_sgemm` 3.86 ms (1 call; not a tensor-op kernel); wgrad 2.24 + 0.45; `nchwToNhwc` 1.27 |
| transposes / copies | 8.6 | 2.2 | `direct_copy_kernel` 3.0 (23 calls); `triton_poi_fused_clone_transpose_18` 2.5 |
| RoPE cos/sin/index (fused) | 6.0 | 1.5 | `...cos_index_mul_native_layer_norm_backward...` |
| permutation gather/scatter, loss | <1 | <0.3 | `_scatter_gather_elementwise_kernel` 0.33 ms; the loss is not visible as its own kernel |

**Are the GEMMs at peak?**
- GEMM work is 5.645 TFLOP per b128 step for S (`aten.mm` 3.744 + `aten.addmm` 1.872 + conv 0.03), i.e. 25.9 TFLOP/s over the 218 ms matmul bucket.
- The CPU-check ran `FlopCounterMode` on CPU at batch 2 with the S config and the three model perf flags (`skip_last_content`, `causal_no_mask`, `query_proj_once`), then scaled to batch 128. B: 22.5 TFLOP per b128 step.
- 3060 peak: NVIDIA lists 3584 CUDA cores and a 1.78 GHz boost clock (https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/). The dense bf16 rate with fp32 accumulate is about 25.6 TFLOP/s at that clock; GeForce Ampere runs fp32 accumulate at half rate. Both are assumptions.

**Are the elementwise kernels at bandwidth?**
- GELU fwd: 23 × (read + write of [25088, 1536] bf16 = 77 MB each) = 3.5 GB in 10.2 ms, about 347 GB/s.
- GELU bwd: 5.3 GB in 15.1 ms, about 352 GB/s.
- Both are near the 3060's ~360 GB/s (assumption).
- The LayerScale grad reduction `mul_sum_view_3` reaches about 213 GB/s, about 60% of bandwidth.

**Data loader (T1).**
- Real data, all flags, optimizer every 32: 431.3 ms/step median vs 427.9 ms compute, i.e. 3.4 ms waiting on data with 8 workers.
- Synthetic with the same settings: 416.0 ms.

**Projection to one RTX 6000 Ada (assumption; scales P's buckets by spec ratios).**
- GEMMs: 218 ms ÷ ~14 (≈360 / 25.6 TFLOPS) ≈ 16 ms at the same efficiency. Realistically 18–25 ms at K = 384.
- Memory-bound (LN + elementwise, 120 ms): ÷ 2.7 (960 / 360 GB/s) ≈ 45 ms.
- Attention: 45.7 ms ÷ (between 2.7 and 14) ≈ 5–15 ms.
- Step ≈ 70–85 ms per 128 images, about 1,500–1,800 img/s per GPU. **Memory-bound work would be about 55–60% of the step.** That makes fusion (max-autotune, GELU / LN / reduction fusion, stacked streams) the main lever. FP8 only shrinks the ~25% that is GEMM.

---

## 3. Candidate table (target = RTX 6000 Ada, sm_89)

| # | candidate | targets (P) | expected gain | testable where | effort | correctness risk | library / custom |
|---|---|---|---|---|---|---|---|
| 1 | Inductor `max-autotune` / `max-autotune-no-cudagraphs` + `coordinate_descent_tuning` (`kern_compile_mode`) | GELU 30 ms, bias/LS epilogues, reductions, GEMMs | target: 5–15% (assumption) | target only (3060 has too few SMs) | trivial | low (rtol 1e-4 compiled) | library |
| 2 | FP8: torchao `perf_fp8` with `tensorwise` / `rowwise` / `rowwise_with_gw_hp` | GEMMs (~25% on target) | ceiling ≈ 1.14x (Amdahl on the 25% GEMM share) minus cast overhead; S: ≤ 1.0x likely; B: 0–10% (assumption) | target only (sm_89) | none–low (recipe option) | medium (not bit-exact) | library |
| 2b | Transformer Engine FP8 (`kern_te_fp8`) | GEMMs + LN-to-FP8 cast | only if torchao is <1.05x on B | target only | medium | medium | library |
| 3 | Patch embed as an unfold + GEMM (`kern_patch_linear`) | ≈8.3 ms conv path (2.1%) | ≈1.5–2% | 3060 + target | low | low | custom (PyTorch ops) |
| 4 | Fused two-stream attention (`kern_attn_fused`: `gqa` or `flex`) | attention 11.9% + the K/V-grad add | −2% to +3% | 3060 + target | low | low (exact, CPU-check) | library (SDPA / FlexAttention) |
| 5 | SDPA backend choice (`kern_sdpa_backend`) | attention | 0–3% (assumption) | 3060 + target | trivial | low | library |
| 6 | GPU JPEG decode + GPU augmentation (`kern_gpu_decode`) | loader | 0 on 3060; target: decides whether 4 GPUs are fed (§4.6) | gain on target only | medium | medium (augmentation must match) | library (DALI / nvJPEG) |
| 7 | GELU epilogue / dGELU dgrad kernel (`kern_gelu_epilogue`) | GELU 7.9% | 3–6%, only if rank 1 leaves GELU unfused | correctness 3060, speed target | high | medium | custom Triton |
| 8 | Stacked-stream layout (`kern_stacked_streams`) | launches, transposes, GEMM size | target: 3–10% (assumption) | correctness 3060, speed target | medium–high | medium | custom (model refactor + #4) |
| 9 | LayerNorm fused into the next GEMM's prologue (`kern_ln_prologue`) | part of LN 59 ms | 1–3% on 3060; more on target (memory-bound) | correctness 3060, speed target | high | medium–high | custom Triton |
| 10 | CUDA graphs (part of `kern_compile_mode`) | launch gaps | 0 on 3060 (idle 0.8%); target 0–5% | correctness 3060, speed target | trivial | low–medium (memory, RNG) | library |
| 11 | Better LayerScale / bias-grad reductions | 18.9 ms (4.9%) at 60% of bandwidth | 1–2% (may come free with rank 1) | 3060 + target | medium | low | Inductor config or custom Triton |

---

## 4. Candidates in detail

### 4.1 Two-stream attention in one kernel

**What the code does now.**
- With `perf_causal_no_mask`, the content stream drops its last permuted token (`modules.py:436`), so both streams have N = 196 rows.
- Each layer makes two SDPA calls: content (`modules.py:121`, `is_causal=True`) and query, which reuses the content K/V via `kv=(k, v)` (`modules.py:251-255`), also `is_causal=True`.
- Both use the same square causal mask over the same K/V. Only Q differs.
- Both calls dispatch to `pytorch_flash::flash_fwd_kernel` / `flash_bwd_*` (preflight.json, "S causal_no_mask").

**Two exact single-call forms (CPU-check: B=2, H=6, N=196, d=64, fp64).**
- **(a) GQA, interleaved heads.** Stack the queries to `[B, 2H, N, d]` in head order (content h0, query h0, …), so query head `i` maps to KV head `i // 2`. Then call `SDPA(is_causal=True, enable_gqa=True)`.
  - Result: forward max |diff| = 0.0; gradients max |diff| = 3.6e-15.
  - Docs: GQA "works with FlashAttention, cuDNN attention, and the math kernel on CUDA tensors" and is "an experimental feature" (https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).
- **(b) FlexAttention, interleaved rows.** Row 2i = content i, row 2i+1 = query i. Use `mask_mod = q//2 >= kv`, `Q_LEN = 2N`, `KV_LEN = N`.
  - Result: forward max |diff| = 7.5e-7 in fp32.
  - Backward can't be checked on CPU: "FlexAttention does not support backward on CPU" (installed torch). It must be tested on the GPU.

**K/V reuse.**
- In form (b), each 128-row Q tile holds 64 positions of both streams, so every K/V tile serves both.
- In form (a), reuse across the two query heads isn't guaranteed. FA2 handles GQA "by passing in KV with fewer heads than Q" (flash-attention README). That each query head reloads K/V is an assumption.

**Why the gain is small.**
- Attention is 11.9% of the step (P).
- K/V per (batch, head) are 50 KB and stay in L2 (assumption). The RTX 6000 Ada's L2 is larger than the 3060's (assumption).
- What fusion removes: 11 extra launches of each attention kernel, the dK/dV add, and some causal tile waste.
- TIER3.md:66-69 measured GQA stacking on torch 2.6 at 232 vs 227 ms/step, i.e. slower.
- Flex's Triton backward is "0.86x–1.05x" of FA2 on causal masks (https://arxiv.org/html/2412.05496v1; read via a summary of the paper).
- FlexAttention's FA4 backend (`BACKEND="FLASH"`) targets Hopper/Blackwell (flash-attention README: FA4 "optimized for Hopper and Blackwell"), so it doesn't apply to sm_89. Use the Triton backend.
- Expected: −2% to +3%.

**xFormers.** Not installed. Its kernels are already the SDPA memory-efficient backend (assumption). Rejected.

**Test.** Flag on vs off: forward output and all four gradients, fp32 rtol 1e-5 (`tests/test_perf_equivalence.py:7`); bf16 tolerance; whole-model loss and grads via `check_fp32` (`tests/test_perf_equivalence.py:79`). 3060 and target.

### 4.2 Attention backends

- Default SDPA priority in the installed torch 2.14.1 is `[1, 2, 0, 3]` = flash, efficient, math, cudnn (`torch._C._get_sdp_priority_order()`, read on CPU; assumption that it doesn't vary by architecture).
- cuDNN is accepted for our causal shapes but never chosen by default (preflight.json).
- Force it with `torch.nn.attention.sdpa_kernel([...], set_priority=True)` (`torch/nn/attention/__init__.py:99`). The SDPA docs warn about nondeterministic algorithms (URL above).
- On sm_89 the candidates are FA2 (default), cuDNN and memory-efficient. Only measurement decides between them; there is no Ada-specific evidence either way (assumption).
- **Removed:** FA3 (Hopper only, see the header).

### 4.3 Prologue / epilogue fusion (now testable on target)

**Default mode now.** Inductor sends GEMMs to cuBLAS/CUTLASS (P kernel names) and fuses LN, residual, LayerScale and casts into Triton kernels (TIER3.md:24-28).

**With max-autotune on the RTX 6000 Ada (142 SMs ≥ 68, `utils.py:1803`):**
- "fuse pointwise into templates epilogues" `epilogue_fusion = True` (`config.py:278`)
- "fuse pointwise into template prologues" `prologue_fusion`, on by default (`config.py:285`, `71-81`)
- `benchmark_epilogue_fusion` (default on) picks fused vs separate per GEMM (`config.py:999`)
- `max_autotune_gemm_backends` default `"ATEN,TRITON,CPP"` (`config.py:647`), so cuBLAS stays in the race
- `coordinate_descent_tuning` (`config.py:794`) also tunes the reduction and pointwise kernels
- Docs: max-autotune "leverages Triton or template based matrix multiplications … It enables CUDA graphs by default"; `max-autotune-no-cudagraphs` is the same without them (https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
- Benchmark both modes, so the template effect is separated from the CUDA-graph effect.

**What it should fuse (assumption until the target profile):**
- GELU into the up-proj epilogue (exact erf, `modules.py:163`)
- bias and LayerScale multiplies into the proj/down-proj epilogues
- casts
- It probably won't pull LayerNorm (a reduction) into a prologue, since prologue fusion is "pointwise" only per the config comment.

**Custom fallbacks, only if the target profile still shows the pattern:**
- `kern_gelu_epilogue`: Triton up-proj + bias + GELU that also stores the pre-activation; down-proj dgrad + dGELU + bias-grad. Removes `gelu_view_24` and most of `gelu_backward_view_8` + `_sum_view_9`.
- `kern_ln_prologue`: for S, K = D = 384, so a 128-row A tile (96 KB bf16) fits in one CTA's shared memory. Row stats are computed inside the GEMM; the wgrad recomputes the normalized input from saved mean/rstd. This matters more on the target, where LN is memory-bound at 960 GB/s.

**Rejected here:**
- cuBLASLt GELU via `torch._addmm_activation`: its non-Lt path applies `gelu_(…, "tanh")` (pytorch `aten/src/ATen/native/cuda/Blas.cpp`, via summary), but our GELU is exact.
- Liger / Apex: see §6.

**QK-norm + RoPE** are already fused by Inductor (2.98 + 2.39 ms, P). No separate kernel.

**Test.** max-autotune: the `test_fp32_cuda_compiled_all` pattern, rtol 1e-4 (`tests/test_perf_equivalence.py:154-157`), run on the target. Custom kernels: kernel vs PyTorch reference, fwd/bwd, fp32 rtol 1e-5 and bf16, S and B shapes, M = 25088 and odd M (TIER3.md:47-52 lists the LN cases).

### 4.4 Launch overhead / CUDA graphs

- 3060: idle 0.81% (P). Not launch-bound.
- Target: about 1688 kernels in a projected ~75 ms step is about 45 µs per kernel. Probably still not launch-bound per GPU; the 4-process DDP host load may change this (assumption).
- `reduce-overhead` "reduces the overhead of python with CUDA graphs", "can come at the cost of more memory usage", and works only for "CUDA only graphs which do not mutate inputs" (torch.compile docs). 48 GB makes the memory cost a non-issue.
- Risk: in-model RNG. The RoPE rescale `uniform_` (`rope.py:76`), dropout and drop path must stay correct under the per-batch seeding (`repa.py:194-196`); assumption that Inductor handles it.
- Test: the compiled rtol 1e-4 test plus a 3-step loss-trajectory match with RNG on.

### 4.5 Low precision (now testable on target)

- **Existing:** `perf_fp8` = torchao `convert_to_float8_training`, default tensorwise recipe, Linear layers with dims divisible by 16 (`repa.py:40-47`). The guard requires ≥ sm_89 (`repa.py:36-38`), so it runs on the RTX 6000 Ada.
- **torchao docs** (https://docs.pytorch.org/ao/stable/workflows/training.html):
  - recipes "`tensorwise` (fastest), `rowwise`, `rowwise_with_gw_hp` (most accurate)"
  - "For small shapes … speedup < 1"; "E2E speedups increase as M,K,N … increase"
  - benchmarked on H100; I couldn't read the shape table (it is an image)
- **Our GEMM shapes:** M = 25,088.
  - S: (K,N) ∈ {(384,384), (384,768), (384,1536), (1536,384)}
  - B: {(768,768), (768,1536), (768,3072), (3072,768)}
- **Expected on Ada** (assumption):
  - **Upper bound:** FP8 at most doubles the throughput of the GEMM share, about 25% of the projected step (§2). The ceiling is 1/(0.75 + 0.25/2) ≈ 1.14x.
  - **Subtract:** the amax/scale/cast kernels FP8 adds. They are memory-bound, which is the share that already dominates on this card.
  - **REPA-S:** K and N are 384–1536, and the torchao docs say small shapes see "speedup < 1". **Expect ≤ 1.0x.**
  - **REPA-B:** larger K/N and a larger GEMM share give 0–10% at best.
  - What makes it pay off at all is Inductor fusing the casts into the neighbouring LN/elementwise kernels (default compile or max-autotune).
- **Plan:** a `kern_fp8_recipe={tensorwise,rowwise,rowwise_with_gw_hp}` option on the existing flag, through `Float8LinearConfig.from_recipe_name(...)` (torchao docs). Run it for S and B, each with default and max-autotune compile.
- **Accuracy check:** loss curve vs bf16 over the first 1k steps.
- **TE FP8** (Delayed / Float8CurrentScaling / Float8BlockScaling recipes, fp8_primer) is a fallback if torchao gives <1.05x on B. It isn't in the env by default: it builds from source; open question 3.
- **Removed:** MXFP8 / NVFP4 are Blackwell only ("MXFP8 and NVFP4 training on NVIDIA Blackwell GPUs", TE README).

### 4.6 Data pipeline (per GPU; single-GPU benchmarks first)

**Now.** CPU PIL decode + `RandomResizedCrop` + flip + `PILToTensor` (`dataset.py:59-62`), `num_workers = 8` per process (`config.py:120`). On the 3060 the loader keeps up (3.4 ms wait, T1).

**CPU-check.**
- Single-threaded, i5-10400F, 400 fake JPEGs (mean 67 KB, 300–500 px; `bench/make_fake_imagefolder.py:27`).
- **381 img/s per core**. Decode is 60% of that time.
- **Caveat:** that figure comes from about 67 KB fake JPEGs. Real ImageNet JPEGs are larger (assumption on mean size), so real per-core throughput will be lower. Re-measure on the real dataset.

**Demand, per GPU.**
- About 1,500–1,800 img/s per GPU (§2 projection, assumption).
- At 381 img/s per core that is about 4–5 fully used cores per GPU, more with real files and collate/IPC overhead.
- The default `num_workers = 8` (`config.py:120`) may need raising.
- The supervisor confirmed CPU cores are not a constraint. So the open questions are worker count and per-worker speed, not core supply.
- **Measure first on one GPU:** `bench/step_time.py --data real` vs `--data synthetic` at S and B, with the best rank 1–2 settings. Only then confirm on 4 GPUs.
- The loader is a bottleneck if real is more than about 3% slower than synthetic, or if ms/step minus compute ms grows beyond the 3.4 ms seen on the 3060 (T1).

**Options.**

*Pre-resized ImageNet copy (short side ~256).* **Supervisor decision.**
- **What:** a one-time offline pass that resizes each JPEG so the short side is about 256 px, re-encoded as JPEG. The pipeline is otherwise unchanged (`RandomResizedCrop(224)` etc., `dataset.py:59-62`).
- **Gain:** smaller files decode faster. Assumption: decode time scales roughly with pixel count, and decode is 60% of per-image CPU time in the CPU-check. It also cuts disk and page-cache I/O. No new code path and no GPU time spent on decoding.
- **Tradeoffs:**
  - (1) It changes the augmentation input. RRC crops are drawn from a downsampled image, so small-scale crops (RRC's default scale range starts at 0.08, assumption from torchvision defaults) are upsampled from fewer source pixels. That is a slight, method-adjacent change in image statistics.
  - (2) Double JPEG compression adds artifacts (quality setting matters).
  - (3) It needs disk space for a second dataset copy and a one-time preprocessing job.
  - (4) Results are no longer directly comparable with NEPA-style full-resolution pipelines unless the reference also uses it.
- **Test:** decode + RRC img/s per core on the copy vs the original (same CPU-check script); a short pretraining run comparing loss curves.

*`kern_gpu_decode` (DALI or nvJPEG).*
- **torchvision `decode_jpeg(device="cuda")`:** "decoded with nvjpeg"; batched lists are "more efficient"; but the API is "[DEPRECATED] Use TorchCodec instead" (https://docs.pytorch.org/vision/stable/generated/torchvision.io.decode_jpeg.html).
- **DALI:** `fn.decoders.image(device="mixed")` + `fn.random_resized_crop`, through `DALIGenericIterator`. "DALI addresses the problem of the CPU bottleneck by offloading data preprocessing to the GPU" (https://docs.nvidia.com/deeplearning/dali/user-guide/docs/). The hardware JPEG decoder option applies to A100-class parts. On Ada, nvJPEG decodes on the SMs (assumption), which takes some GPU time back from training.
- **Correctness risk.** DALI's crop sampling and resize filter differ from torchvision's. That is method-adjacent: open question 4.

**Test.** Decode PSNR vs PIL (> 40 dB, assumed threshold); RRC scale/aspect statistics over 10k draws; real vs synthetic `step_time.py` on the target at the production worker count.

### 4.7 Permutation gather/scatter and loss

Gathers are fused into Inductor kernels: `…gather…` 0.29 ms; `_scatter_gather_elementwise_kernel` 0.33 ms (P). The loss (`repa.py:17-20`) doesn't appear among kernels above 0.3 ms. Total under 0.3%. **Reject.**

### 4.8 Other items ≥ ~2%

**(a) Patch embedding (2.1%).**
- `Conv2d(k=16, s=16)` (`modules.py:14`) runs as `implicit_convolve_sgemm` (3.86 ms, 1 call), plus wgrad 2.69 ms and layout conversion 1.27 ms.
- The same math as a GEMM: reshape to [B·196, 768], then `@ W.view(D, 768).T + b`. That is 14.8 GFLOP, about 0.6 ms at 3060 peak (assumption).
- Parameters keep the Conv2d shape, so checkpoints don't change.
- Test: fp32 rtol 1e-5 on the embedding, its grads, and whole-model loss. 3060 and target.

**(b) Stacked-stream layout (`kern_stacked_streams`).**
- The streams share `q_proj`, `proj` and the MLP (`modules.py:251-269`). Only the LNs and LayerScales are per stream.
- Keep h and g in one [B, N, 2, D] buffer. Run each shared GEMM once with M = 2 × 25,088; apply per-stream LN/LS with stacked [2, D] parameters.
- Q viewed as [B, 2N, H, d] is then already in §4.1(b)'s interleaved order, so attention needs no copy.
- Effects:
  - halves GEMM and elementwise launches
  - removes the dK/dV add and the per-call transposes (2.2%)
  - gives max-autotune bigger M
- Must keep three paths working: `skip_last_content`, `query_proj_once`, and the `arch_query_mlp_width` fallback.
- Test: whole-model fp32 rtol 1e-5 for every perf-flag combination (`tests/test_perf_equivalence.py:99`).

**(c) Column reductions (4.9%).**
- LayerScale grad `sum(grad * x)` runs at about 60% of bandwidth (§2).
- Try `coordinate_descent_tuning` first (rank 1). Write a multi-output Triton reduction only if the target profile still shows it above about 3%.

### 4.9 Multi-GPU note (4 GPUs)

- DDP all-reduce happens once per optimizer step: `global_batch_size` 4096 / (128 × 4) = gradient accumulation 8 (`config.py:65-68`; batch 128 per device).
- S gradients are about 22M params × 4 bytes ≈ 90 MB per all-reduce (assumption from the parameter count), over PCIe, with no NVLink (assumption). This is small next to 8 micro-steps of ~75 ms.
- Not a kernel target. `perf_ddp_static_graph` already exists.

---

## 5. Ranked build plan (target RTX 6000 Ada; one GPU for kernel benchmarks, then 4-GPU confirmation)

Each item is a `kern_*` config flag, off by default, wired like the `perf_*` flags (`config.py:12-23`, `145-152`), timed with `bench/step_time.py` and profiled with `bench/profile_step.py`. Equivalence tests extend `tests/test_perf_equivalence.py` (fp32 rtol 1e-5; compiled rtol 1e-4).

| rank | flag | what to build | 3060 verifies | target verifies | equivalence test |
|---|---|---|---|---|---|
| 1 | `kern_compile_mode={default,max-autotune,max-autotune-no-cudagraphs,reduce-overhead}` + `kern_coord_descent` | pass the mode to `torch.compile` (`repa.py:60`); set `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1` | reduce-overhead correctness only | **speed + correctness for all modes** | compiled fp32 rtol 1e-4; 3-step loss match with RNG on |
| 2 | `kern_fp8_recipe={tensorwise,rowwise,rowwise_with_gw_hp}` on existing `perf_fp8` | pass `Float8LinearConfig.from_recipe_name` in `convert_to_fp8` (`repa.py:40-47`) | no (sm_86) | **yes**; run with ranks 1 default and max-autotune | per-layer output vs bf16 (relative error bound, e.g. < 5e-2, assumption); 1k-step loss curve vs bf16 |
| 3 | `kern_patch_linear` | reshape + `F.linear` with the conv weight as [D, 768] | yes | yes | fp32 rtol 1e-5, embedding + whole model |
| 4 | `kern_attn_fused={gqa,flex}` | SDPA `enable_gqa` with interleaved heads; FlexAttention with the `q//2 >= kv` block mask | yes | yes | per-call fwd + 4 grads vs two calls; whole model rtol 1e-5 |
| 5 | `kern_sdpa_backend={flash,cudnn,efficient}` | `sdpa_kernel(..., set_priority=True)` | yes | yes | fp32 rtol 1e-5 (cuDNN: 1e-4) |
| 6 | `kern_gpu_decode={dali,nvjpeg}`, or a pre-resized copy (supervisor decision) | GPU decode + per-sample RRC/flip; normalize already on GPU (`repa.py:188-192`). Alternative: an offline short-side-256 copy, with no code change | functional only | yes (single-GPU real-vs-synthetic decides it) | PSNR vs PIL; RRC parameter statistics; for the pre-resized copy, a loss-curve comparison |
| 7 | `kern_gelu_epilogue` | only if rank 1's target profile still shows separate GELU kernels | correctness | speed | fwd/bwd vs `F.gelu(F.linear)`, fp32 rtol 1e-5 and bf16, S/B, odd M |
| 8 | `kern_stacked_streams` | [B, N, 2, D] buffer, shared GEMMs once, attention via rank 4 `flex` | correctness | speed | whole-model fp32 rtol 1e-5 across perf-flag combos |
| 9 | `kern_ln_prologue` | Triton GEMM with LN in the A-load prologue + wgrad recompute | correctness | speed | TIER3.md:47-52 cases, fwd/bwd fp32 / bf16, S and B |

**Re-rank rule.** After ranks 1–2 are profiled on the target (step 4 below), recompute the §2 group table from the new profile with the best rank 1–2 settings on. **Any memory-bound group still above 5% of the step moves up** ahead of ranks 3–5. The groups are: GELU, column reductions, LayerNorm-fused kernels, transposes/copies, RoPE/QK-norm. Their custom kernels are ranks 7–9 and §4.8c:
- GELU → `kern_gelu_epilogue`
- LN → `kern_ln_prologue`
- reductions → a Triton reduction kernel
- transposes and several of these together → `kern_stacked_streams`

Groups that max-autotune fused away drop out. Ties go to the larger group. Attention and GEMM groups keep their ranks.

**Benchmark sequence on Arcturus (single GPU unless stated).**
1. preflight: capability 8.9, 48 GB, **driver version (≥ 580 for cu130, else the cu126 env)**, torch/CUDA versions
2. profile `all` at S and B (default compile)
3. rank 1 modes × S/B
4. rank 2 recipes × {default, best rank-1 mode} × S/B, then profile the best and **apply the re-rank rule**
5. ranks 3–5 (and any groups promoted by the rule)
6. real vs synthetic on one GPU at the production worker count (decides rank 6 / pre-resized copy); then confirm on 4 GPUs
7. re-profile the best combination; build the remaining custom kernels only for groups still above 5%

## 6. Rejected candidates

- **FlashAttention-3:** Hopper only ("H100 / H800 GPU", flash-attention README). Not available on sm_89.
- **FA4 / FlexAttention `BACKEND="FLASH"`:** targets "Hopper and Blackwell" (same README); needs `flash_attn.cute` (`torch/_inductor/kernel/flex/flex_flash_attention.py:160-170`).
- **MXFP8 / NVFP4:** Blackwell only (TE README); torchao MXFP8 is "prototype" on B200 (torchao docs).
- **Liger-Kernel:** standalone LayerNorm / RoPE / GeGLU kernels and FusedLinearCrossEntropy (https://github.com/linkedin/Liger-Kernel). Inductor already fuses our LN with residual/LayerScale (P). Opaque kernels would break those fusions (assumption). No cross-entropy here.
- **Apex** (FusedLayerNorm / FusedDense): same reasoning as Liger; FusedDense's GELU likely depends on tanh-GELU epilogues (assumption).
- **cuBLASLt GELU via `torch._addmm_activation`:** tanh-approximate GELU (Blas.cpp, via summary); ours is exact.
- **xFormers:** nothing beyond SDPA/Flex for this mask.
- **Custom two-stream FlashAttention from scratch:** FlexAttention gives the same structure (§4.1b) with a verified mask; attention is ≤12%.
- **Fused permutation/loss kernel:** under 0.3%.

## 7. Open questions for you

1. Preflight on Arcturus: does it report capability 8.9 and 48 GB, and what driver version (≥ 580 means the cu130 env works; below that, use cu126)?
2. Can I run a 10-second pure-matmul microbenchmark on the 3060 to confirm "GEMMs at peak" (§2)?
3. Is it OK for the Arcturus env to include DALI and (optionally) Transformer Engine? TE builds from source and needs a CUDA toolkit on the node.
4. Is GPU-side augmentation that is statistically equal but not bit-equal to torchvision's RandomResizedCrop acceptable? And **(supervisor decision)** is a pre-resized ImageNet copy (short side ~256) acceptable, given its small change to RRC input statistics (§4.6)?
5. Production `num_workers` and dataset location on Arcturus (local NVMe vs network FS)?
6. Should `kern_stacked_streams` cover the `arch_*` flags too, or only the production two-stream config?

## Appendix: benchmark env

A fresh env, `repa-kern`, made with the repo's own `ENV_NAME=repa-kern bash bench/setup_env.sh`: `environment.yml` + pytest + torchao. The existing `repa` env is untouched. Extras and their status are in `bench/ENV.md`.
