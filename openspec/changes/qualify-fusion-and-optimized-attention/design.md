## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

## Goals / Non-Goals

Deliver the explicitly enumerated B06, B07 outcomes through Provider kernels; Runtime graph eligibility, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

## Decisions

### Implementation boundary and data flow

Qualify each fused pattern and attention profile independently, then expose portable operator semantics through existing dispatch. Cover numerical order, GQA, masks, position conventions, tails, long contexts and both prefill/decode workspace budgets.

### Long-context CUDA kernels (first increment)

The long-context campaign (`qualify-real-checkpoint-performance`) showed that Qwen2.5-0.5B on an RTX 3060 spent its 512- and 4096-token time in Device work, not Runtime host costs. `long_context_kernel_times` (CUDA Provider, ignored, release only, run 41850538460) timed each layer's work. Each prefill matmul reached 0.8-1.0 TFLOP/s. The prefill attention took 2.6 ms per layer at 512 rows and 156 ms at 4096. A decode step's attention took 463 us per layer over 4096 keys, which is 11 ms of a 16 ms step.

- **LM head on the last prompt row only (Runtime).** A prefill materializes only the last row of its logits. When a `matmul` or `matmul-bias` node's single output is such an edge, the Runtime hands it the last row of its `a` input alone. That row is read back: 896 floats for Qwen2.5-0.5B. Each output row depends only on the same input row, so the logits are unchanged. `prefill_lm_head_runs_on_the_last_row_only` checks that the logits invocation sees one row and that the logits match the oracle.
- **Split decode attention (CUDA).** One-row attention gives each chunk of 128 admitted keys its own block. One warp reads each K row, coalesced, and adds its products with a fixed butterfly. The weights are normalized before they weight V. `attention_combine_kernel` merges the chunks in chunk order, each weighted by its share of the softmax mass. The output is deterministic, and a weighted mean of V rows. The grid covers every K/V row, so a captured step replays at any later key count, and empty chunks are skipped. The chunk partials live in a scratch buffer that is grown only outside a capture and never freed while a capture may read it.
- **Tiled prefill attention (CUDA).** Each block serves 32 queries of one head, with 4 threads per query. K and V are staged through shared memory one tile of keys at a time. Each query keeps a running max, a running sum and the weighted mean of the V rows seen so far (online softmax), so its output stays within the V rows' range. The kernel serves uncaptured multi-row attention whose head dimension is a multiple of 4, up to 128. Other cases keep the existing kernels.
- **Tiled prefill matmuls (CUDA).** From 64 rows, each thread computes 8 rows x 4 columns of one split-K slice. The block's `a` rows are staged through shared memory 256 inners at a time. Each output still adds its slice sums inner-ascending, and its slices in slice order, so it is bit-identical to the one-row kernel. The conformance test checks this at 64 and 130 rows.

Conformance:
- The CUDA attention test adds decode over 300 and 1000 keys (the latter windowed), and prefill cases: multi-tile at 130 and 300 rows (the latter windowed), offset queries (40 of 200 rows) and a 128-dimension head.
- Large finite V values stay finite on both new kernels. The first split-kernel version summed unnormalized weights times V and overflowed; `block_attention_keeps_large_finite_values_finite` caught it (run 41856997166).

Evidence:
- **Setup.** RTX 3060, Qwen2.5-0.5B, greedy, 128 new tokens, 5 measured runs. The baseline is run 41845390146; the LM head alone is run 41855543895; the four changes together are run 41859746720.

| | Before | LM head only | All four |
|---|---|---|---|
| Warm TTFT median, 5 / 512 / 4096 prompt tokens | 15.0 / 743 / 9381 ms | 14.1 / 573 / 7940 ms | 15.0 / 220 / 3372 ms |
| TPOT median, 5 / 512 / 4096 prompt tokens | 4.4 / 5.8 / 16.4 ms | 4.4 / 5.7 / 16.1 ms | 4.5 / 4.8 / 6.7 ms |
| Decode throughput at 4096 prompt tokens | 60.8 tok/s | 62.2 tok/s | 149.0 tok/s |
| Prefill matmuls, 4096 rows | 0.8-1.0 TFLOP/s | | 3.1-3.7 TFLOP/s |
| Prefill attention per layer, 512 / 4096 rows | 2.6 / 156 ms | | 1.5 / 81 ms |
| Decode attention per layer, 512 / 4096 keys | 39 / 463 us | | 23 / 85 us |

- **Logits.** With the LM head alone, CUDA logits are unchanged (1.3e-5 vs. the independent corpus). The new attention kernels sum in a different order: 2.7e-5, the same tokens, and 13 of 16 steps still replay.
- **Short prompt.** Its TPOT moved from 4.4 to 4.5 ms, which is the combine launch the split attention adds to each layer.
- **What remains at 4096 tokens.** The prefill attention is about 1.9 of the 3.4 s: 0.37 TFLOP/s, with one key per step and a per-key rescale. The layer matmuls are about 0.9 s. These are the next targets.

Register-resident tiled prefill attention (GPU CI run 41883061198, CUDA Provider 5a8d6ac):
- **What changed.** The tiled kernel is now compiled per head dimension: 4, 8, 16, 32, 64, 96 and 128. Every per-query loop is unrolled, so each query's q, scores and output stay in registers instead of local memory. Keys are taken 16 at a time: their scores, their max and their normalized mean of V are computed together, then merged into the running mean. That replaces a per-key rescale with two divisions. Blocks serve 64 queries instead of 32, so each K/V tile read from global memory serves twice as many queries.
- **Kernel times.** Prefill attention per layer went from 1.54 to 0.50 ms at 512 rows, and from 80.5 to 20.7 ms at 4096 rows (about 1.4 TFLOP/s).
- **End to end.** Warm TTFT median went from 220 to 194 ms at 512 prompt tokens, and from 3372 to 1891 ms at 4096. TPOT is unchanged: 4.5, 4.8 and 6.7 ms.
- **Logits.** CUDA logits are 1.5e-5 from the independent corpus, with the same tokens; 13 of 16 steps replay.
- **What remains at 4096 tokens.** The layer matmuls (about 0.9 s) and the attention (about 0.5 s) now dominate.

Tensor Core prefill matmuls (CUDA Provider 67996fc; GPU CI runs 41901349874 and 41904874231):
- **Scope.** From 64 rows, `matmul`, `matmul-bias`, `matmul-residual` and `gated-matmul` against compact weights run `mma.sync.m16n8k16` (bfloat16 inputs, f32 accumulation) on Devices of compute capability 8.0 or later.
- **Separate module.** The Tensor Core kernels are compiled for `compute_80` into a module of their own, loaded only on such a Device, so the main kernel source still builds everywhere. A test requires that module to load on a compute capability 8.0+ Device, so a compilation failure cannot silently leave prefills on the CUDA cores.
- **Precision.** Compact weights enter the Tensor Cores exactly. Each f32 activation is split into two bfloat16 terms, hi = bf16(a) and lo = bf16(a - hi). Both terms are multiplied, which keeps about 16 significant bits of the activation and doubles the Tensor Core work.
  - Results match the CUDA-core kernels to within f32 accumulation order, not bit for bit. The conformance test compares rows within 1e-5 · k of each other, relative to max(1, |value|).
  - A dedicated test sets every activation to 1 + 2^-12, which bfloat16 alone rounds to 1, and checks that the low term is kept.
- **Disable control.** `MAGNETAR_CUDA_TENSOR_CORES=0` keeps the bit-identical CUDA-core tile kernels.
- **Kernel times.** Prefill matmuls went from 3.1-3.7 TFLOP/s to 7.0-7.4 TFLOP/s at 4096 rows, and from 3.0-3.6 to 4.7-6.3 TFLOP/s at 512 rows. The small K and V projections (896 x 128) gain least.

| Same run (41904874231), RTX 3060 | Tensor Cores off | Tensor Cores |
|---|---|---|
| Warm TTFT median, 512 prompt tokens | 194.6 ms | 98.6 ms |
| Warm TTFT median, 4096 prompt tokens | 1893 ms | 1008 ms |
| TPOT median, 5 / 512 / 4096 prompt tokens | 4.5 / 4.8 / 6.7 ms | 4.5 / 4.8 / 6.7 ms |
| Generated tokens, 5 / 512 / 4096 prompt tokens | | identical, 128 of 128 each |

- **Numerics.** The logit corpus covers a 5-token prompt, which stays below the Tensor Core threshold: 1.5e-5, unchanged. For long prompts, the campaign prints each prompt's generated tokens. In the same GPU job, the Tensor-Cores-off campaign produced the same 128 greedy tokens after 512- and 4096-token prompts, consistent across the 5 measured runs.
- **Limits.** The activation split doubles the Tensor Core work.
- **Rejected follow-ups.** Two measurements show these kernels are bound by something other than the `mma` work and the stage barriers. Each was reverted:
  - **One bfloat16 term per activation** (opt-in profile, run 41918957621). Tokens were identical to the two-term and Tensor-Cores-off campaigns. There was no speed-up: warm TTFT was 106.2 ms against 106.5 ms at 512 prompt tokens, and 1020 ms against 1038 ms at 4096. Prefill matmuls ran at the same TFLOP/s.
  - **32 inners per staged tile instead of 16** (run 41922809583). This halves the barriers and stage loops per inner, but gave no measurable gain. Prefill matmuls ran at 6.2-6.7 TFLOP/s for 4096 rows, against 6.4-7.4 in earlier runs. Warm TTFT was 98.6 ms at 512 prompt tokens and 1019 ms at 4096.
  - **What is left.** The remaining costs are the per-element f32-to-bfloat16 staging of `a`, which every column block repeats, and the transposed scalar stores of `b`. Going further means a pipelined design (`cp.async`, `ldmatrix`, larger tiles), not more tuning of this kernel.

Tensor Core prefill attention (CUDA Provider 286ea5d; GPU CI run 41986235007):
- **Kernel.** `attention_tc_kernel` serves 64 queries of one head per block: 4 warps of 16 queries, taking keys 64 at a time.
- **Arithmetic.** S = Q K^T and P V run as `m16n8k16` bfloat16 `mma` with f32 accumulation.
  - Every f32 operand enters as two bfloat16 terms, and each product is computed as hi*hi + hi*lo + lo*hi, keeping about 16 significant bits.
  - Softmax is online per query row.
  - Each key tile's probabilities are divided by the new running sum before they weight V, so the output stays a weighted mean of V rows. A large-finite-value test covers this kernel.
  - The 1/8 scale is folded into Q, which is exact for 64 dimensions.
- **Scope.** It handles causal, windowed, offset and grouped-query attention. It serves uncaptured attention of at least 16 query rows with 64-dimension heads, on Devices whose Tensor Core module loaded. `MAGNETAR_CUDA_TENSOR_CORES=0` keeps the CUDA-core kernels.
- **Tests.** The conformance test gains a 64-dimension case of 200 queries over 1000 keys with a 150-key window, and a 16-row case.

| RTX 3060, Qwen2.5-0.5B | Before (run 41922809583) | After (run 41986235007) |
|---|---|---|
| Prefill attention per layer, 512 / 4096 rows | 0.49 / 20.6 ms | 0.26 / 8.2 ms |
| Warm TTFT median, 512 / 4096 prompt tokens | 98.6 / 1019 ms | 96.1 / 715 ms |
| Tensor Cores off, same run | | 193.6 / 1889 ms |
| TPOT median, 5 / 512 / 4096 prompt tokens | 4.5 / 4.8 / 6.7 ms | 4.5 / 4.8 / 6.7 ms |

- **Numerics.** The generated tokens are identical with the Tensor Cores on and off, 128 of 128 each after 5-, 512- and 4096-token prompts. CUDA logits stay 1.5e-5 from the independent corpus, and 13 of 16 steps replay.

Pipelined Tensor Core prefill matmuls (CUDA Provider 4050f95; GPU CI run 41999569272):

- **Kernels.** `tc_split_kernel` splits `a` into its hi and lo bfloat16 terms once per matmul, instead of once per column block. `matmul_tcp_kernel` and `gated_matmul_tcp_kernel` then compute 128 x 64 output tiles on 8 warps. Two shared-memory stages are filled by `cp.async`, so the next 16 inners load while the current ones compute. Fragments are read with `ldmatrix`; the weight uses `.trans` and keeps its natural [k][n] layout, with no transposed scalar stores.
- **Precision.** The products and f32 accumulation are the same as in the plain Tensor Core kernel. The near-`f32::MAX` truncation of the hi term is kept. The activation-precision test now covers both kernels.
- **Scope.** The pipelined kernels need `k` and `n` to be multiples of 8 (16-byte rows) and an untransposed `a`. Other shapes keep the plain Tensor Core kernel. `MAGNETAR_CUDA_TENSOR_CORES=0` still keeps every prefill on the CUDA cores.

| RTX 3060 | Plain Tensor Core matmuls | Pipelined |
|---|---|---|
| Prefill matmuls, 4096 rows (896 x 896, x 4864, 4864 x 896) | 6.2-7.4 TFLOP/s | 9.2-10.7 TFLOP/s |
| Prefill matmuls, 512 rows (same shapes) | 4.7-6.3 TFLOP/s | 7.3-9.6 TFLOP/s |
| Warm TTFT median, 512 / 4096 prompt tokens | 96.1 / 715 ms | 64.4 / 553.8 ms |
| Tensor Cores off, same run | | 193.3 / 1890 ms |
| TPOT median, 5 / 512 / 4096 prompt tokens | 4.5 / 4.8 / 6.7 ms | 4.5 / 4.8 / 6.7 ms |

- **Numerics.** The generated tokens are identical with the Tensor Cores on and off, 128 of 128 each after 5-, 512- and 4096-token prompts. The CUDA logits against the Reference CPU corpus passed in the same run.
- **What remains at 4096 tokens.** The layer matmuls are now about 0.3 s (about 12 ms per layer, gate and up counted separately) and the prefill attention about 0.2 s (8.1 ms per layer). The K and V projections (896 x 128) stay at 4.3 TFLOP/s: 16 column blocks do not fill the Device.

Pipelined Tensor Core prefill attention (CUDA Provider 9bba019, then 311a549 without the superseded kernel; GPU CI run 42031696029):

- **Why attention, not a fused QKV matmul.** The pipelined matmuls already reach 10.7 TFLOP/s. With two bfloat16 terms per activation and f32 accumulation, the RTX 3060's ceiling is about 12.8 TFLOP/s, so a fused QKV matmul would save a few milliseconds at most. The prefill attention took 8.1 ms per layer at 4096 rows, about twice its Tensor Core bound.
- **Kernel.** `attention_tcp_kernel` keeps the arithmetic of the first Tensor Core attention: three bfloat16 products per f32 product, online softmax, and probabilities normalized by the running sum. K and V are split into their hi and lo terms once per attention by `tc_split_kernel`; before, every query block re-split them, and with grouped-query attention each of the 7 query heads sharing a KV head did so too. Key tiles of 32 stream through two `cp.async` stages. Fragments are read with `ldmatrix`, and V uses `.trans` from its natural layout, so the transposed scalar stores are gone.
- **Scope.** The kernel handles causal, windowed, offset and grouped-query attention, as before. It needs `kv_model_dim` to be a multiple of 8. Captured launches and other shapes use the tiled CUDA-core kernels. `MAGNETAR_CUDA_TENSOR_CORES=0` still keeps every prefill on the CUDA cores. The first Tensor Core attention kernel is removed.

| RTX 3060 | First Tensor Core attention | Pipelined |
|---|---|---|
| Prefill attention per layer, 512 / 4096 rows | 0.26 / 8.06 ms | 0.16 / 4.80 ms |
| Warm TTFT median, 512 / 4096 prompt tokens | 64.4 / 553.8 ms | 63.1 / 481.5 ms |
| Tensor Cores off, same run | | 193.5 / 1894.9 ms |
| TPOT median, 5 / 512 / 4096 prompt tokens | 4.5 / 4.8 / 6.7 ms | 4.5 / 4.8 / 6.7 ms |

- **Numerics.** The 76 CUDA Provider tests pass on the Device, including every attention conformance case and the large-finite-value test. The generated tokens are identical with the Tensor Cores on and off, 128 of 128 each after 5-, 512- and 4096-token prompts, and the CUDA logits against the Reference CPU corpus pass.
- **Long-context work overall.** Warm TTFT at 4096 prompt tokens went from 9381 to 482 ms, and at 512 from 743 to 63 ms. TPOT at 4096 went from 16.4 to 6.7 ms.

Opt-in one-term activations for the pipelined matmuls (CUDA Provider 8e2ec2d; GPU CI run 42054671809):

- **Profile.** `MAGNETAR_CUDA_TENSOR_CORE_ACTIVATIONS=bf16` feeds the pipelined matmuls only the hi bfloat16 term of each activation. This halves their `mma` work and their activation staging, and keeps about 8 significant bits of each activation instead of 16. Attention and the plain Tensor Core matmuls keep their split terms. The two-term split stays the default.
- **First attempts.** Instantiating the matmul for both profiles gave each kernel two copies of its static shared stages. The gated matmul needed 67.5 KB, over the 48 KB limit, and the Tensor Core module failed to load (run 42045694178). The profile is now a runtime flag. A test bounded at 1e-4 near `f32::MAX` then failed under the profile, at 2^-8 by design (run 42050195147); its bound now follows the profile.

| RTX 3060, same run | Two terms (default) | One term (opt-in) |
|---|---|---|
| Prefill matmuls, 4096 rows (896 x 896, x 4864, 4864 x 896) | 9.1-10.4 TFLOP/s | 10.1-11.9 TFLOP/s |
| Warm TTFT median, 512 / 4096 prompt tokens | 59.0 / 464.5 ms | 56.0 / 425.3 ms |
| Tensor Cores off | 193.6 / 1887.2 ms | |

- **Numerics.** The 73 Provider tests that apply pass under the profile; the activation-precision test skips itself. The generated tokens are identical across the three campaigns, 128 of 128 each after 5-, 512- and 4096-token prompts.
- **Finding.** Halving the `mma` work gained only 7-15% per matmul and 8% of TTFT at 4096 tokens. The pipelined matmul is therefore bound by staging, `ldmatrix` and synchronization, not by the Tensor Cores. The 12.8 TFLOP/s two-term ceiling assumed earlier does not describe it. A deeper pipeline with larger tiles should help both profiles without changing precision.

### Identity, eligibility and recovery

Inputs retain validated artifact/model/Component identities, dtype/layout, phase, shape and version provenance where relevant. Runtime checks trust, ResourceAffinity, capability support and MemoryManager admission before policy ranks candidates. Plans and state reference immutable generations and opaque leases. Cancellation stops future submissions; in-flight work retains its leases until completion. On Device loss or incompatible state, invalidate affected work and replan or fail explicitly; never silently rebind resources or restore an untrusted generation.

### Compatibility and WIT evolution

Preserve existing F32 and single-Device baseline fixtures and public request semantics. Reuse current WIT and Rust contracts first. Before implementation requires new graph operators, state envelopes or result types, record their exact versioned portable schema, negotiation, unsupported behavior and migration tests; native API types are prohibited. A new precision or model profile is opt-in until its validation evidence supports promotion. Vendor implementations stay in their repositories, followed by an explicit parent submodule pin update.

### Authority and isolation

Tools and service/transport adapters use caller-authorized inputs and bounded operations. Inference and Components gain no ambient filesystem, network, process, source-control or secret authority. Local source streaming checks artifact ranges and identity. External campaigns/artifacts are untrusted until verified. Snapshot or distributed data remains tenant-scoped and requires authorization in addition to checksums; signing keys and credentials never enter reports.

### Qualification and rollout

Use the shared real-checkpoint harness and baseline corpus rather than inventing isolated pass criteria. A profile records checkpoint revision/digests, code and Provider SHAs, hardware/driver, flags, dtype, shape, budgets and warmup. Exercise 1/4/16/32 sessions, 512/4096/16384-token contexts and 128-512 generated tokens only inside model/backend/capacity limits; report every skipped combination. Repeat at least five warm runs when affordable and document exceptions, uncertainty and cold costs. Check logits and downstream quality with declared thresholds; token identity alone is insufficient. Measure TTFT, TPOT/ITL p50/p95/p99, aggregate/per-request throughput, launch/phase/copy/I/O costs, RSS/VRAM, fragmentation, useful cache savings and energy only when valid telemetry exists.

Promotion requires appropriate real hardware and a real checkpoint, error/cancellation/memory-pressure/isolation tests, before/after evidence, a disable control and a qualified fallback or structured Unsupported result. Experimental features remain disabled by default until qualified. Run existing Rust, WIT, OpenSpec, license/SBOM/signature and artifact-reproducibility gates for affected code. Compare tail latency and total memory as well as throughput, under the same budget. Publish limitations and do not infer a blanket speedup from heterogeneous runs.

## Risks / Trade-offs

- Keep unfused and reference attention implementations available for unsupported or regressing profiles.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.
