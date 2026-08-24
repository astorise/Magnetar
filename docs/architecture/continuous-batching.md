# Continuous Batching

Continuous batching is the Runtime/Scheduler contract for coordinating many
generation operations across prefill, decode, streaming, cancellation, cache
reuse, and memory pressure.

Batching is Runtime-owned. Clients and Components may refer to Runtime-issued
`BatchId` and `BatchSlotId` values, but those identifiers do not grant authority
and do not encode Provider handles, Device handles, or memory pointers.

## Operation Lifecycle

Batched operations use explicit lifecycle states:

- `admitted`
- `queued`
- `prefill-pending`
- `prefilling`
- `decode-pending`
- `decoding`
- `streaming`
- `completed`
- `cancelled`
- `failed`
- `rejected`
- `evicted`

The lifecycle separates queued work from prefill and decode execution. Terminal
states are retained until Runtime policy removes them from batch slots.

## Prefill And Decode

Prefill and decode scheduling are distinct. Prefill advances prompt state and
may consume large temporary workspace. Decode advances active sequences by token
steps. Runtime policy may prioritize either phase or form mixed batches where a
future scheduler implementation supports that safely.

## Batch Slots

A batch slot is the Runtime-owned execution position for one operation inside a
batch. Slots bind operation identity, optional session identity, model and
tokenizer compatibility, sequence length, generated-token count, KV cache
references, Prefix Cache references, Provider/Device placement metadata, memory
reservation references, priority, deadline, and cancellation state.

Slots reference KV cache and Prefix Cache resources by Runtime-managed
identifiers. They never own raw KV cache memory or raw prefix contents.

## Compatibility

Only compatible operations may share a batch execution step. Compatibility
checks include model context, architecture, compute dtype, tokenizer, Provider,
Device, Resource Affinity, KV cache layout, sequence limits, memory placement,
and Provider-assisted sampling policy where that path is used.

## Memory Manager Relationship

Batching does not allocate directly. It produces `MemoryAdmissionRequest` values
for batch input buffers, output buffers, logits buffers, attention masks,
position buffers, sampling workspace, KV cache blocks, Prefix Cache lookup
workspace, temporary staging, and Provider-specific workspace. Runtime submits
those requests to the Memory Manager.

## Scheduling Policy

The current contract models FIFO, priority, deadline, fairness, latency target,
throughput target, decode-priority, and prefill-priority policy modes. Policy
also carries queue limits, active-operation limits, max batch tokens, max batch
sequences, starvation-prevention intent, browser feature requirements, and
prefill/decode enablement.

## Backpressure, Cancellation, And Failure

Backpressure may come from queue limits, Provider saturation, Device pressure,
memory pressure, streaming consumers, session limits, shutdown, or policy.
Errors are stable `BatchingErrorCode` categories.

Cancellation and failure are per slot where possible. One cancelled or failed
operation does not automatically fail unrelated slots unless a Runtime,
Provider, or Device failure makes continuation impossible.

## Streaming

Streaming output preserves per-operation token order even when batch execution
interleaves operations. Slow consumers are tracked as per-slot streaming
backpressure and must not corrupt other operation streams.

## Observability

Batch observations are redacted by default. Runtime observations must not expose
raw prompts, raw logits, raw KV cache contents, or raw Provider handles.

## Browser Compatibility

The contract is platform-neutral. It has no Wasmtime dependency and does not
require native Provider loading. Browser targets may return structured
unsupported-feature errors or use reduced batching policy depending on available
memory and future WebGPU/Provider capabilities.

## Non-Goals

This contract does not choose a concrete scheduling algorithm, implement paged
attention, speculative decoding, beam search, distributed routing, remote
serving protocols, or a browser batching engine.
