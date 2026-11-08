# Tier 3: custom kernels (plan only, nothing implemented)

Numbers are from local profiles on an RTX 3060, REPA-S, batch 64, bf16, `torch.compile` (default mode), synthetic
data, 10 profiled steps (`bench/profile_step.py`, 2026-11-08). Rerun on Arcturus before deciding anything: the bucket
shares depend on the GPU.

| bucket (share of GPU kernel time) | baseline | all Tier 1 flags |
|---|---:|---:|
| matmul | 52.3% | 56.2% |
| layernorm | 15.4% | 16.7% |
| other elementwise (GELU, residuals, RoPE, loss, ...) | 13.2% | 15.3% |
| attention | 17.4% | 11.4% |
| GPU time per step | 234.7 ms | 204.4 ms |

## (a) Fused LayerNorm + QK-norm + RoPE kernel

**Code path.** Per layer and stream (`modules.py`):
- `ln1_h` / `ln1_g` (LayerNorm over D) → `q_proj` / `kv_proj` (GEMM) → `q_norm` / `k_norm` (LayerNorm over head_dim,
  no affine by default) → `apply_rotary_pos_emb` (`rope.py:99-110`: `x * cos + rotate_half(x) * sin`, with cos and sin
  in fp32 gathered per sample, `[B, N, head_dim]`).
- After attention: `proj` (GEMM) → LayerScale → residual add → `ln2_*` → MLP (GEMM, GELU, GEMM) → LayerScale →
  residual add.

**What `torch.compile` already does.** The largest "layernorm" kernels in the profile are Inductor-generated Triton
kernels that already fuse LayerNorm with the neighbouring residual add, LayerScale multiply and casts. Example:
`triton_per_fused__to_copy_add_mul_native_layer_norm_backward_view_12`, 7.3 ms/step with all flags. GELU is fused the
same way (`triton_poi_fused_gelu_gelu_backward_view_9`, 7.3 ms/step). So most of the roughly 32% in these two buckets
is already-fused elementwise work. A hand-written kernel that only re-fuses the same ops would gain little.

**Where a gain could still come from.** Moving this memory-bound work into the GEMMs:
- the LayerNorm into the GEMM's input (prologue)
- QK-norm + RoPE into the `q_proj`/`kv_proj` output (epilogue)
- GELU into the up-projection's output (epilogue)

That removes one read and one write of the activations per fused op. Inductor does this only with its Triton GEMM
templates (`mode="max-autotune"`), not with cuBLAS. The local GPU can't test that: Inductor logged
"Not enough SMs to use max_autotune_gemm mode" on the 3060 (28 SMs).

**Plan, cheapest first:**
1. On Arcturus: `torch.compile(mode="max-autotune")` (and `max-autotune-no-cudagraphs`), all Tier 1 flags, S and B.
   Measure step time and rerun the profile. If the layernorm + elementwise share drops, the compiler already gets the
   gain and no custom kernel is needed.
2. If not: dump the generated code (`TORCH_LOGS=output_code`) and list which ops around the QK path stay unfused.
3. Only then write a Triton kernel for the remaining pattern, most likely QK-norm + RoPE as a GEMM epilogue for
   `q_proj` / `kv_proj`.

**Correctness tests** (same pattern as `tests/test_perf_equivalence.py`):
- Kernel vs the PyTorch reference, forward and backward, fp32 and bf16, shapes S (6 heads x 64) and B (12 x 64),
  N = 196 / 197.
- Inputs with large and small row norms (LayerNorm eps), and both identity and random RoPE rotations.
- The per-sample gathered cos/sin, including CLS's identity rotation (`rope.py:91-97`).
- Whole-model loss and gradient equivalence with the kernel on, rtol 1e-5 in fp32.

**Expected gain** (assumption, not measured): at most the share of the two buckets that is memory traffic moved
into GEMMs. If half of the roughly 32% went away, about 1.15x; realistically less, because GEMM epilogues add
register pressure. Step 1 above gives the first real number.

## (b) Custom two-stream attention kernel

**Why the expected gain is small:**
- With `causal_no_mask`, both streams already run FlashAttention. Preflight on the 3060 shows default dispatch
  `pytorch_flash::flash_fwd_kernel` / `flash_bwd_*` for the `is_causal` calls, versus the memory-efficient kernel
  for the original masked calls.
- Attention is then 11.4% of GPU time (23.0 ms of 204.4 ms per step).
- A fused two-stream kernel could save the second read of K/V and one launch per layer. It cannot remove the query
  stream's attention FLOPs. Even halving attention time is about 5-6% of the step (assumption).
- One fused call is already possible without a custom kernel: stacking the query heads GQA-style over shared K/V. In
  an earlier local measurement (RTX 3060, torch 2.6) it was slightly slower than two separate causal calls (232 vs
  227 ms/step). On-chip K/V reuse isn't guaranteed by the dispatch.

**Recommendation:** don't write it unless attention is above about 20% of GPU time on the Arcturus profile with all
flags on.
