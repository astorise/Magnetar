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
