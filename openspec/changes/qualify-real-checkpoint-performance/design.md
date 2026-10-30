## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

## Goals / Non-Goals

Deliver the explicitly enumerated A01, A02, A03, A04, A07, A08 outcomes through Qualification tooling; Runtime and Provider measurements, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

## Decisions

### Implementation boundary and data flow

Extend the existing production-loading integration harness with immutable run manifests and raw samples. Keep fast fixture checks separate from manual/nightly real-checkpoint campaigns. Compare internal baseline improvements separately from external-engine comparisons.

### Initial implemented profile

The first increment covers A01 and A04 fully and A03 for discovery and explicit skips:

- `tools/inference-qualification` (root workspace, serde/sha2 only) owns the report model: an immutable `RunManifest` whose SHA-256 names the report file, raw per-request `GenerationTrace`s (token arrival offsets at the public streaming boundary), derived TTFT/TPOT/ITL/throughput, procfs-based interval resource usage (peak RSS reset per interval when the kernel allows, `rchar`/`read_bytes`, CPU cores), nearest-rank p50/p95/p99 plus MAD dispersion, a noise-aware comparison that only promotes improvements clearing the combined noise band with at least five samples per side, and the A03 three-stage (discovery/compute/production) capability matrix.
- `integration-tests/production-loading/src/tests_real_checkpoint_qualification.rs` drives `ProductionLoadedModel`: one load, one cold request, one warmup and five measured greedy requests on the same resident instance. The per-PR lane runs the tiny synthetic bundle and must be labeled `fixture_only`; the `#[ignore]`d manual/nightly lane runs pinned `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775` (digest recomputed before use) on Reference CPU and, when a Device exists, CUDA.
- Profile: F32, batch 1, concurrency 1, plain-text prompt, 16 generated tokens by default. All 24 standard grid cells are recorded as skipped with their reason.
- GPU CI (`gpu-runner-smoke.yml`) runs the campaign on CUDA with 128 new tokens. It first ran 32: 7 x 128 tokens did not finish in 90 minutes on an RTX 3060 (run 41008921749, ~1.9 s/token already at 5-21 positions). `bound-memory-manager-bookkeeping` removed the growth with history (run 41051599580: TPOT 1.31-1.34 s, flat) and `attribute-redacted-inference-costs` removed most of the fixed per-step cost (run 41088755268: TPOT median 135 ms, warm TTFT median 1.24 s, 7.4 tok/s). The design's grid cells with 512-16384-token contexts stay recorded as skipped. Each finished request is printed and appended to `progress-<provider>.jsonl` so a timeout still leaves evidence, and the per-token latency growth (least-squares slope of inter-token latency over position) is reported with the warm statistics.
- A diagnostic, non-gating `real_checkpoint_decode_cost_profile` reports, for one 64-token request, inter-token latency per 8-position bucket, the time inside versus outside the Runtime's own prefill/decode steps, and Runtime events per token; it feeds `attribute-redacted-inference-costs` (A06) rather than qualifying anything.
- A02: `ProductionLoadedModel::generate_streaming_with_logits` (additive; observation only, existing entry points unchanged) hands each step's full pre-sampling logits to the harness. A `LogitReference` keeps per step the sampled token, the top 32 logits and a SHA-256 of the full vector. The checked-in corpus `integration-tests/production-loading/fixtures/qualification/qwen2.5-0.5b-instruct-7ae5576-reference-cpu-f32.json` (Reference CPU, pinned prompt token ids, 16 greedy tokens) is the reference every Provider is compared to, so GPU CI compares CUDA to Reference CPU without re-running it. Initial profile `f32-cross-provider-v1`: identical greedy tokens and the reference's top 8 logits within 5e-2 absolute; failures keep every step's differences. Lifecycle checks on the same instance: stream order and a single final `Finished`, capture not changing tokens, cancellation keeping exactly the delivered prefix, and resume from prompt + prefix reproducing the remaining tokens with logits inside the profile.
- No WIT change. Device memory is `not_measured` until a Provider exposes telemetry; A07 and A08 are pending gates and the A02 evidence is a separate artifact, so no campaign report from this increment can qualify production performance.

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

- No quality threshold is inferred from plausible text or a theoretical weight-size ratio.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.
