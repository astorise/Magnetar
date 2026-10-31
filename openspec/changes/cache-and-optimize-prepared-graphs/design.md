## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

## Goals / Non-Goals

Deliver the explicitly enumerated C02, C03 outcomes through Runtime graph validation and prepared plans, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

## Decisions

### Implementation boundary and data flow

Cache immutable plan generations with a complete compatibility fingerprint and bounded storage. Run typed graph passes before publication; validate side effects and KV writes as well as numerical output. Recheck lifetimes and trust on reuse.

### Initial implemented profile

The first increment caches graphs, not plans:

- A loaded model (`ProductionLoadedModel`) keeps the execution graphs its Component built, keyed by prompt length, in a `ComponentGraphCache` shared with its execution engine. The model's Component, architecture and identity are fixed for its lifetime, so the prompt length is the only remaining input the graphs depend on. Request preparation builds or reuses them, prepares the plans against them, and every step of the request dispatches those same graphs instead of asking the Component to rebuild them, so a step cannot execute a graph other than the one its plan was fingerprinted against; every node is still validated against the plan.
- The cache holds at most 8 prompt lengths, least recently used first out, and is dropped with the model. Engines constructed outside `ProductionLoadedModel` keep rebuilding graphs per step.
- Plans are still prepared for every request (44 ms on Reference CPU once `attribute-redacted-inference-costs` removed their quadratic fingerprinting). Caching them needs the revalidation C02 requires (Kernel Registry generations, instance mutation version, capacity) and is not done yet, so tasks 2.1-2.3 stay open.

CUDA evidence (RTX 3060, run 41100147352, five measured 128-token requests on one resident instance, compared with run 41088755268 before this increment and the plan-fingerprint fix): warm TTFT median 1244 ms -> 128 ms, TPOT median 135 ms -> 102 ms, 7.4 -> 9.8 tok/s, latency growth 0.0 ms/token. Request preparation is now 19 ms, all of it plan preparation; the per-step graph build is gone from the decode profile.

### First fusion pass (C03)

- `magnetar-runtime::graph_fusion::fuse_matmul_bias` folds `add(matmul(a, b), bias)` into one `matmul-bias` node when the intermediate has exactly one consumer and no KV or prefix-cache semantics, and `bias` is a rank-1 weight matching the matmul output's last dimension. The fused node keeps the matmul's id and attributes and the add's output edge, so downstream nodes and resource identities are unchanged. On Qwen2.5 this is the Q, K and V projections of every layer: 72 of the ~459 nodes of a decode step.
- `matmul-bias` is a portable `LinearAlgebra` Operator (3 inputs, `matmul`'s attributes) produced only by this pass, never by a Component. Every implementation computes the matmul and then adds the bias row, the same operations in the same order as the pair it replaces, so fused and unfused graphs are bit-identical on a given Provider (Reference CPU and the CPU Provider: tested bit for bit; CUDA: the split-K kernel adds `bias[col]` to its finished sum, tested bit for bit against its own `matmul` + `add`).
- The pass runs where `ProductionLoadedModel` builds graphs (request preparation and, on a cache miss, the execution engine), before plans are prepared, so plans and steps always see the same graph. It only fuses when the instance's Provider has an active, prepared `matmul-bias` Kernel; any other Provider keeps the unfused graph. `ProductionLoadedModel::set_graph_fusion(false)` turns it off and forgets cached graphs.
- Tests: on the real Qwen Component's bias-bearing graphs every Q/K/V pair is fused, the fused graph keeps a valid execution order and fusing again finds nothing; an intermediate with a second consumer is not fused; an unbiased graph is unchanged. On Reference CPU with the real checkpoint a decode step dispatches 387 nodes instead of 459 and the logits still match the independent corpus (5.4e-5, same tokens).

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

- An identical shape does not imply an identical model, resource generation or authorized executable.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.
