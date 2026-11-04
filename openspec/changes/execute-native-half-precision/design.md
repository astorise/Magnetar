## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

## Goals / Non-Goals

Deliver the explicitly enumerated B01, B02 outcomes through Runtime typed resources; loaders; CUDA Provider, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

## Decisions

### Initial profile: lossless bfloat16 weight storage (task 1.1)

The first increment keeps every activation, intermediate, KV row and accumulation in f32, and changes only how weights are stored.

- **Why it comes first.** After `capture-and-replay-cuda-graphs`, a Qwen2.5-0.5B decode step on an RTX 3060 is bound by the Device: the f32 weights it reads (1.98 GB) take about 5.5 ms of its 8.0 ms at the card's bandwidth.
- **Why it is lossless.** The pinned checkpoint stores every weight as BF16. The f32 weights the Runtime materializes are exact upcasts of those values, so storing them as bfloat16 again loses nothing.

**Contract.** `ProviderExecutionApi::write_weight_tensor` writes a Model Instance weight. A weight is never written again while its instance lives, so a Provider may keep it in any representation that reproduces every value exactly. The default writes it like any tensor. Graphs, descriptors, Operators, WIT and MemoryManager admission are unchanged: weights stay f32 tensors to every caller.

**CUDA Provider.**
- A weight is stored compact, as its bfloat16 bits, when:
  - it is a matrix of at least 65 536 values;
  - every one of its values has the low 16 bits of its f32 encoding at zero.
- The matmul family reads a compact `b`, and the embedding lookup a compact table. Both convert each value back to the same f32 inside the Kernel, with the same accumulation order, so their results are bit-identical.
- Any other Kernel argument expands the weight to f32, for good, on first use: correct, but without the saving.
- Downloads, buffer copies, peer copies and step capture addresses handle compact buffers.

**Limits.**
- MemoryManager admission still counts 4 bytes per weight value, so it overstates the Device memory compact weights use.
- An F16 checkpoint, or a BF16 checkpoint whose weights are transformed before materialization so that they stop being bfloat16 values, keeps f32 weights.
- Native half activations and KV (the rest of B01) and Tensor Core matmul (B02) remain future work.

### Evidence for the initial profile

Real Qwen2.5-0.5B, greedy, on an RTX 3060, GPU CI run 41688972164 (Runtime e808b4a, CUDA Provider with this change). The baseline is run 41652697101, with f32 weights:

| | f32 weights | bfloat16 weights |
|---|---|---|
| Campaign TPOT median (128 tokens, 5 runs) | 8.0 ms (125.6 tok/s) | 5.3 ms (187.8 tok/s) |
| Warm TTFT median | 42.5 ms | 33.9 ms |
| Waiting for the Device per replayed step | 6.84 ms | 4.25 ms |
| Peak RSS | 1.38 GB | 1.38 GB |

- CUDA logits stay within 1.3e-5 of the independent corpus: the same maximum difference as with f32 weights.
- Replayed steps are still bit-identical to per-node ones, and 13 of 16 steps still replay.
- A first attempt converted each whole weight before its upload and raised peak RSS to 1.66 GB (run 41686324411). Uploading a chunk of 1 Mi values at a time removed that.

Device time of one decode matmul per projection shape (run 41718804397, `decode_matmul_bandwidth`, 200 back-to-back launches, so the smallest shapes read as their launch cost):

| `[1, k] x [k, n]` | f32 | bfloat16 |
|---|---|---|
| 896 x 896 (Q, O) | 13.1 us, 246 GB/s | 5.5 us, 293 GB/s |
| 896 x 128 (K, V) | 4.8 us | 4.9 us |
| 896 x 4864 (gate, up) | 55.4 us, 314 GB/s | 30.4 us, 286 GB/s |
| 4864 x 896 (down) | 55.6 us, 313 GB/s | 35.8 us, 243 GB/s |
| 896 x 151936 (LM head) | 1601 us, 340 GB/s | 823 us, 331 GB/s |

- The projections take about 117 us per layer in bfloat16 against 83 us at the card's 360 GB/s; the K and V matmuls are launch-bound and the down projection reaches the lowest bandwidth. They are the next decode GEMV targets.
- A compact Kernel reading 8 weights per load was not kept. Taken for every shape, it raised TPOT to 6.4 ms (run 41715792007): 256 columns per block leave 896-column projections 4 blocks for 28 SMs. On the LM head alone it saved 18 us (805 against 823 us), not worth its Kernel.

### Implementation boundary and data flow

Propagate storage, compute and accumulation dtypes independently through tensor metadata, graph validation and Kernel Registry eligibility. Materialize native half storage; keep F32 reductions explicit. CUDA owns Tensor Core kernels and alignment handling, and Reference CPU remains the correctness oracle.

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

- Do not turn the mandatory F32 first-native baseline into a mandatory half-precision requirement.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.
