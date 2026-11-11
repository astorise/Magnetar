## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

## Goals / Non-Goals

Deliver the explicitly enumerated B03, B04 outcomes through Formats and loaders; Providers; Runtime validation, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

## Decisions

### Implementation boundary and data flow

Preserve validated packed bytes and quantization metadata from GGUF/GPTQ/AWQ ingestion through preparation. Qualify Q8_0 before Q4_K/Q5_K, then GPTQ/AWQ; model calibration identity and packing variants explicitly. Distinguish direct kernels from any opted-in dequantizing fallback.

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

- Malformed block metadata or silent F32 expansion must fail qualification before performance claims.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.

## Evidence: Q8_0 phase 1 on CUDA (RTX 3060)

Checkpoint: Qwen/Qwen2.5-0.5B-Instruct-GGUF `qwen2.5-0.5b-instruct-q8_0.gguf`
(revision `9217f5db79a29953eb74d5343926648285ec7e67`), loaded through the
Qwen Model Component (architecture family `qwen2`, artifact format `gguf`),
greedy decoding of 64 tokens after a warm-up request. GPU Runner Smoke Test
run 42243870149 (commit `f57c764`):

| Mode | Load | Warm TTFT | TPOT (64 tokens) |
|---|---|---|---|
| Packed weights stored (`MAGNETAR_CUDA_PACKED_WEIGHTS` unset) | 732 ms | 13.8 ms | 4.50 ms |
| Packed weights declined (`MAGNETAR_CUDA_PACKED_WEIGHTS=0`, dequantized to f32 at load) | 4276 ms | 14.3 ms | 7.20 ms |

Both modes generate the same 64 token ids (" Paris. The capital of the United
States is Washington, D. C. …"); the CI step fails if they differ.

What runs packed in phase 1: quantized projections are stored as the
checkpoint's transposed `[out, in]` blocks (`layout=transposed`); decode
matmuls of up to 64 rows read them packed (`q8_matmul_rows_kernel`), the
gated MLP reads both projections in one launch with no intermediate
(`q8_gated_matmul_rows_kernel`, so the decode step still captures into a
CUDA graph), and the embedding reads its table row by row. Longer prefill
matmuls dequantize the weight for that call only (phase 2: Tensor Core
prefill from packed weights).

Capture finding: a device buffer freed while a CUDA graph capture is in
progress records a driver error that cudarc reports at the next driver
call. The CUDA Provider now takes that deferred error before ending a
capture, so an aborted capture always ends and the error fails the capture
instead of a later, unrelated call.

### Mistral-7B-Instruct-v0.3 Q8_0 on CUDA (RTX 3060, 12 GB)

`bartowski/Mistral-7B-Instruct-v0.3-GGUF@61fd4167fff3ab01ee1cfe0da183fa27a944db48`
(`Mistral-7B-Instruct-v0.3-Q8_0.gguf`, 7.7 GB, Apache-2.0), served by the
Llama Model Component (the file declares `general.architecture = llama`;
the GGUF loader restores llama.cpp's query/key row permutation and
rebuilds the SentencePiece tokenizer). GPU Runner Smoke Test run
42260791015 (commit `0c4186c`): load 50.6 s, warm TTFT 13 prompt tokens
68.2 ms, TPOT 41.4 ms over 64 greedy tokens; text "Paris.\n\n## What is the
capital of France in 2021?\n\nParis…". It fits the Device only packed: the
f32 dequantized form (28 GB) cannot run there, so there is no declined
control for this profile.

## Evidence: packed Q4_K (with Q6_K dequantized) on the RTX 3060

GPU Runner Smoke Test run 42986314772 (commit `ee188ac`): the CUDA Q4_K
conformance test (decode matmul, bias and residual addends, gated matmul,
exact download) passes on the GPU, and Mistral-7B-Instruct-v0.3 `Q4_K_M`
loads and answers "Paris".

| Kernel, one token | `Q8_0` | `Q4_K` |
|---|---|---|
| `[4096, 4096]` | 66.4 us (269 GB/s) | 57.4 us (164 GB/s) |
| `[4096, 14336]` | 204.8 us (305 GB/s) | 176.4 us (187 GB/s) |
| `[14336, 4096]` | 224.0 us (279 GB/s) | 189.8 us (174 GB/s) |

| Mistral 7B, one GPU | TPOT | TTFT | Load |
|---|---|---|---|
| `Q8_0` | 28.09 ms | 74.3 ms | 50.7 s |
| `Q4_K_M` | 34.98 ms | 99.7 ms | 42.6 s |

`Q4_K` reads 47 % fewer bytes but its first decode Kernel is
compute-bound (integer-to-float conversion of every nibble, software f16
conversion per lane), and `Q4_K_M`'s `Q6_K` tensors (output head, half
of `ffn_down`, `attn_v`) are dequantized to bf16 at load, so the export
is slower per token than `Q8_0`. Next: exact cheap conversions in the
Q4_K Kernels, then packed `Q6_K`.
