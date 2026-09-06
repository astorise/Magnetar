## Why

`enable-device-resident-kernel-chaining` fixed the *input* side of the
first-native dispatch loop's host round-trip but deliberately deferred the
*output* side: a Kernel's own output is always downloaded via
`ctx.provider.read_tensor(&output_resource.id)` under a Kernel-internal id
(e.g. `{operation_id}.out`), then re-uploaded under a *different* id
(`edge.{output_edge_id}`) by the caller. That deferral was reviewed with
product/architecture: the decision is that this must be fixed now, not
deferred further, because `device-resident-resource`'s "Same-Device
Pipeline Avoids Mandatory Host Copy" requirement is a Core contract
violation, not merely a performance gap -- and Magnetar's roadmap
explicitly adds more hardware Providers (Metal, OpenVINO, QNN, WebGPU)
soon, so the underlying resource-identity/admission/ownership confusion
that makes this hard to fix today must be stabilized *before* those
Providers exist, not migrated across all of them later.

Product/architecture explicitly rejected the simplest fix (bolt an `owner`
parameter onto `ProviderExecutionApi::submit_kernel`) as Provider-specific
patching of a Runtime-owned responsibility. Direct inspection of the
existing code during this review also found the admission mechanism this
change touches (`execute_invocation_with_memory_manager`, duplicated
identically across `providers/cpu`, `providers/cuda`, and this crate's own
in-crate `ReferenceCpuExecutor`) already leaked a `MemoryAllocationId` on
every single Kernel dispatch -- fixed separately, but confirming this seam
needs a real, generic redesign rather than another local patch.

## What Changes

- **Kernel output admission moves from Provider-side to Runtime/caller-side.**
  A new Core-level helper (Runtime-owned, Provider-agnostic, not a trait
  method) admits an output's `MemoryAllocationRequest` (class, byte size,
  placement, and a *caller-chosen* `MemoryAllocationOwner`) and records its
  `TensorResidency` *before* `submit_kernel` is called, under the
  resource's *final* identity (e.g. a graph edge's own id) rather than a
  Kernel-internal synthesized one (e.g. `{operation_id}.out`).
- `execute_invocation_with_memory_manager` (all three duplicated copies)
  checks `memory.tensor_residency(id)` for each output first: if already
  present (a caller pre-admitted it, the new path), it skips self-admission
  and writes directly into that identity; if absent (an old-style caller --
  conformance tests, direct Provider dispatch outside first-native), it
  falls back to today's self-admission behavior unchanged. **No
  `ProviderExecutionApi` trait signature changes** -- `submit_kernel`
  already receives `&mut MemoryManager`; this is a behavior/contract change
  inside the existing signature, backward compatible by construction.
- `magnetar-runtime/src/first_native_runtime.rs`'s dispatch loop threads
  the target graph edge's own resource id into
  `dispatch_reference_cpu_operator[_multi]`'s output parameter (instead of
  letting it synthesize `{operation_id}.out`), and pre-admits that id with
  `MemoryAllocationOwner::Session(cache_id)` before dispatch. The
  subsequent edge-level `write_tensor_value_admitted` call becomes a no-op
  confirmation (not a re-write) whenever the Kernel already wrote directly
  into the correct, final identity -- eliminating the physical D2H+H2D for
  every non-KV-concatenation edge. KV-history-concatenated edges (genuinely
  new, host-computed data) still explicitly write, correctly.
- A new benchmark suite measures decode latency/token, tokens/s, H2D/D2H
  volume and time, and the kernel-vs-transfer time split, with an explicit
  before/after comparison against this change -- the empirical baseline
  product asked for, independent of and not gating this change's own
  correctness decision (the contract violation is fixed regardless of
  measured cost).

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `cuda-provider`: "CUDA Provider Memory Manager Integration" requirement
  updated to describe the caller-pre-admits-then-Provider-writes contract
  for outputs, superseding the per-buffer self-admission
  `enable-device-resident-kernel-chaining` shipped.
- `device-resident-resource`: no requirement wording changes (the existing
  "Same-Device Pipeline Avoids Mandatory Host Copy" requirement is already
  correctly worded) -- this change makes the implementation actually
  satisfy it for the output side too, the same non-goal-closing pattern as
  `enable-device-resident-kernel-chaining`'s own Decision 1.

## Impact

- `magnetar-runtime/src/memory.rs`: new Runtime-owned output-admission
  helper.
- `magnetar-runtime/src/first_native_runtime.rs`: dispatch loop threads
  final resource ids through instead of synthesizing Kernel-internal ones;
  pre-admits outputs before dispatch.
- `magnetar-runtime/src/reference_cpu.rs`,
  `providers/cpu/src/lib.rs`, `providers/cuda/src/executor.rs`:
  `execute_invocation_with_memory_manager` gains the
  check-before-self-admit fallback.
- New benchmark crate/binary or `#[bench]`/criterion-based suite (exact
  location decided in design.md).
- No breaking change to `ProviderExecutionApi`; no submodule pin
  compatibility break expected (verify empirically in tasks).
- Depends on `enable-device-resident-kernel-chaining` (done) for the
  input-side passthrough this change extends to the output side.
