## Context

`implement-cuda-provider-baseline` (not yet archived) shipped a working CUDA
Provider, but its own spec delta describes behavior the implementation does
not have: explicit data movement, and device-owned Memory Manager
integration. Both were deliberate, documented non-goals in that change's own
design.md, written before its spec delta text was finalized -- the spec text
overshot what was actually built. `generalize-first-native-provider-dispatch`
(done) fixed the two Core P0s blocking any non-CPU Provider from being
selected at all (dynamic Provider/Device resolution instead of hardcoded
Reference CPU, and a structured `TensorValue` write-error channel). With that
prerequisite in place, this change is the next gate the intermediate audit
predicted: making device residency real end to end, not just representable.

Three already-archived Core capabilities define the target contract this
change must satisfy, none of which any Provider or the first-native dispatch
loop implements today:
- `device-resident-resource`: a Tensor Resource can exist and execute
  entirely device-side; compatible consecutive Device Kernels must not incur
  a mandatory host round-trip.
- `execution-stream`: a logical `ExecutionStream`/`CompletionToken`
  abstraction for ordered, potentially-asynchronous Provider submission.
- `device-memory-pool`: logical, accounted Device memory pools (explicitly
  **not** required for this change -- CUDA Provider's baseline spec already
  opts out of pooling in favor of direct per-resource allocation, and this
  change does not revisit that).

Direct code inspection (this session) traced exactly how the current
implementation violates the target contract:
- `providers/cuda/src/kernels.rs`: every kernel method does
  `clone_htod` -> compute -> `clone_dtoh`, always, regardless of where the
  input came from or where the output is headed next.
- `providers/cuda/src/executor.rs`: `CudaExecutor` stores every Tensor
  Resource as `HostTensor` in a `Mutex<BTreeMap>`; `MemoryPlacement::Device`
  is reported to the Memory Manager for data that is not actually on the
  device between calls.
- `magnetar-runtime/src/first_native_runtime.rs`'s `execute_qwen_graph_nodes`
  (the sole production Qwen dispatch loop): for each node, each input edge
  already computed by an earlier node is read via `read_tensor_value` and
  immediately forced through `TensorValue::into_host` (line ~3062), then
  handed to `dispatch_qwen_graph_node` as a `Vec<HostTensor>`. Downstream,
  `dispatch_reference_cpu_operator_multi` re-writes each of those host bytes
  under a **new** `TensorResourceId` (`{operation_id}.a`, `.b`, ...) via
  `ctx.provider.write_tensor(id, tensor)` before kernel selection/submission
  even happens. So today, even when the Kernel Registry correctly selects
  CUDA for two consecutive nodes (confirmed: `prepare_node_execution`
  resolves the real `binding.provider` from the Prepared Plan, not a
  hardcoded value), the runtime still forces a D2H copy of node N's output
  followed immediately by an H2D copy of the identical bytes as node N+1's
  input -- a round-trip that exists purely because the transport layer
  copies tensor *values* under fresh resource identities instead of passing
  an existing device-resident resource *reference* through.

That last point reframes the fix: this is not fundamentally a
"materialize less" problem, it is a "stop renaming/recopying a resource that
already lives where it needs to be" problem. `TensorValue::Opaque` already
exists (added by `define-provider-prepared-kernel-execution-contract`) as a
bare marker variant -- it carries no payload because the caller already has
the `TensorResourceId` it was read under. No enum shape change is needed;
what's missing is a code path that, given an `Opaque` value, passes the
existing `TensorResourceId` through to the next dispatch instead of
insisting on host bytes to re-wrap under a new one. (As implemented, this
change fixes exactly that -- the *input* side of the round-trip. The
*output* side, described above as "an H2D copy... as node N+1's input",
turned out on closer inspection to be two independently-admitted physical
slots today, not one; eliminating it safely is separate follow-up work --
see Decision 7.)

## Goals / Non-Goals

**Goals:**
- CUDA Provider allocates and retains real device memory per Tensor
  Resource, persisting across separate Kernel invocations, matching what
  `implement-cuda-provider-baseline`'s own spec already (prematurely)
  claims.
- CUDA Provider's kernel methods perform upload/download only when actually
  required (source not yet device-resident, or a host-visible result is
  requested), never unconditionally.
- A Kernel invocation whose eligible-operator input is already
  Provider-resident (from an earlier node's output, itself resident because
  of the same round-trip Decision 7 leaves in place -- see there) reuses the
  producer's existing `TensorResourceId` directly instead of downloading and
  re-uploading a copy under a new identity, halving the round-trip traffic
  between two chained device-resident Kernels.
- `KernelErrorCode` gains an out-of-device-memory category.
- `CudaProvider::health()` reports `HealthState::Degraded` (not `Available`)
  when a Device was found but kernel compilation failed, closing the
  observability gap where `health() == Available` and
  `execution_api() == None` could coexist unexplained.
- `.github/workflows/gpu-runner-smoke.yml` describes current reality and is
  re-dispatched against `main` on real hardware.

**Non-Goals:**
- Device Memory Pool (soft/hard reservation, watermarks): CUDA Provider
  keeps direct per-resource allocate/free, as its own baseline spec already
  states. Persisting a buffer across calls is not pooling.
- True asynchronous CUDA execution (multiple in-flight streams, overlap of
  compute and transfer): this change makes submission *stream-shaped*, not
  concurrent. CUDA Provider's `submit` still blocks until the kernel
  completes.
- Multi-Device / cross-GPU peer transfer, `device-resident-resource`'s
  replica requirements: single-GPU only, matching the existing baseline
  scope (P2 finding, tracked separately).
- Rewriting `dispatch_qwen_matmul`/`_rmsnorm`/`_attention`/etc.'s own
  signatures or the multi-output Kernel gap tracked in
  `define-provider-prepared-kernel-execution-contract` task group 3 --
  unrelated and already an open, separately-owned gap.
- Changing which Provider/Device the Kernel Registry or Prepared Plan
  selects. That resolution is already correct (verified: `binding.provider`
  is honored); this change only removes the unnecessary copy once a
  same-Provider/Device pair is already selected.

## Decisions

### Decision 1: Resource-reference passthrough, gated on operator eligibility, not Provider/Device matching

**Design correction made during implementation**, replacing the plan
originally written here: `execute_qwen_graph_nodes` resolves the *entire*
graph under one single `ctx.provider` instance for the whole call (set once
at the top of the function, confirmed by reading it) -- there is no
per-node Provider/Device variation within one execution to check a
Prepared Plan binding against. So when `read_tensor_value` returns
`TensorValue::Opaque` for an edge, that is already the complete and correct
signal that *this* Provider holds it device-resident; no separate
"same Provider/Device as producer" check is needed or possible at this
granularity.

What the passthrough is actually gated on: whether the *consuming* node's
own operator ever dereferences tensor bytes directly. `matmul`, `attention`,
`silu` (`unary`), `mul`/`residual-add` (`binary_same_shape`), and
`embedding` only forward values to a Kernel dispatch (confirmed by reading
each: they read at most `.shape` for descriptor/dimension math). `rmsnorm`
(weight-row broadcasting) and `rope` (per-head slicing) manipulate raw
`Vec<f32>` in Rust and always materialize, unconditionally -- see Non-Goals.
For an eligible operator's `Opaque` input, the loop builds
`NodeInputValue::Resident { id, shape }` (`shape` carried alongside since
`TensorValue::Opaque` itself has no payload) instead of calling
`into_host`; for an ineligible operator, or a `TensorValue::Host` value, or
no bound edge (a weight edge), it materializes exactly as before.

**Alternative considered (this session's own earlier draft)**: gate on
"the consuming node's Prepared Plan binding names the same Provider/Device
that produced this edge." Rejected once re-examined: with a single
`ctx.provider` per execution, this check is always trivially true when
reachable at all, so it added complexity (a Prepared Plan lookup) without
changing behavior versus the simpler "operator eligibility" gate.

**Alternative also considered**: give `TensorValue::Opaque` a payload (e.g.
a Provider-private boxed handle) and thread that value itself through
`dispatch_qwen_graph_node`'s signature. Rejected: `TensorResourceId` is
already the addressing scheme every `ProviderExecutionApi` method uses
(`read_tensor_value(&resource_id)`); inventing a second handle
representation would duplicate identity and reopen the "two tensor-access
pathways coexist" risk `define-provider-prepared-kernel-execution-contract`
already flagged, this time for identity rather than transport.

### Decision 2: `dispatch_qwen_graph_node`/`dispatch_reference_cpu_operator_multi` accept a per-input enum, not a bare `Vec<HostTensor>`

New type `NodeInputValue { Host(HostTensor), Resident { id:
TensorResourceId, shape: Vec<u64> } }` replaces `Vec<HostTensor>` in
`dispatch_qwen_graph_node`'s signature and the five passthrough-eligible
`dispatch_qwen_*` functions' input parameters (Decision 1). A second type,
`NodeInputResource { Fresh(TensorResourceId, TensorDescriptor, HostTensor),
Resident(TensorResourceId, TensorDescriptor) }`, replaces
`dispatch_reference_cpu_operator[_multi]`'s `inputs` parameter:
`ctx.provider.write_tensor(id, tensor)` is now called only for `Fresh`
entries; a `Resident` entry is passed to kernel selection/dispatch under
its existing `TensorResourceId` with no write at all. `rmsnorm`/`rope` keep
`HostTensor` parameters unchanged -- `dispatch_qwen_graph_node` calls
`NodeInputValue::into_host` (downloading via the existing `read_tensor` if
needed) immediately before invoking them, isolating the "always
materializes" non-goal to exactly those two call sites rather than
threading a conditional through their own bodies. Return types are
deliberately unchanged (`HostTensor` throughout, see Decision 3's note on
the output side being out of scope).

**Alternative considered**: eliminate the two-representation split entirely
by making the Kernel Registry/Provider always resolve resources by id and
never accept raw bytes. Rejected as out of scope: it would require
rewriting Reference CPU's entire compute path (which is genuine Rust
arithmetic over `Vec<f32>`, not a resource reference) and contradicts this
change's Non-Goal of leaving `dispatch_qwen_*` internals alone.

### Decision 7: Output-side round-trip left in place; `KernelMemoryClass` hardcoding fixed as a discovered prerequisite

Two further findings surfaced while implementing Decisions 1-2, both
requiring a scope correction (put to the user mid-implementation; the user
chose to proceed with the corrected scope in the same change):

1. **The output side has its own, separate round-trip**, distinct from the
   input-side one Decisions 1-2 fix:
   `dispatch_reference_cpu_operator_multi`'s output extraction always calls
   `ctx.provider.read_tensor(&output_resource.id)` (a forced download,
   under the Kernel's own internal id, e.g. `{node_id}.out`), and the outer
   loop then re-uploads that value under a *different* id
   (`edge.{output_edge_id}`) via `write_tensor_value_admitted`. Eliminating
   this too would require either unifying those two ids (so the Kernel
   writes directly under the edge's id) or making the edge-level
   `write_tensor_value_admitted(Opaque, ...)` call skip its own
   Memory-Manager admission -- both risk double-counting or dropping the
   Session-owned accounting that resource lifecycle (KV pending/promote,
   session release) depends on. **Deliberately left unfixed here**: the
   input-side fix alone already removes half of the round-trip traffic
   between two chained device-resident Kernels (the *consumer's* D2H+H2D),
   safely, without touching Memory Manager admission semantics at all
   (confirmed: `write_tensor`, the input-side-only path, performs no
   admission). The output side is real follow-up work, not silently
   solved.
2. **`KernelMemoryClass` was hardcoded to `Host`** for every
   `KernelResource` (both inputs and the output) this dispatch loop builds,
   regardless of which Provider is actually resolved.
   `providers/cuda/src/advertisements.rs` declares
   `KernelMemoryClass::Device` for every CUDA Kernel advertisement; `kernel.rs`'s
   `validate_resource` rejects a `KernelResource` whose memory class isn't
   in the advertisement's `memory_classes` set. This means every CUDA
   Kernel invocation through this dispatch loop would fail
   `KernelMemoryClassUnsupported` at `validate_invocation` -- a real,
   previously-undiscovered blocker independent of residency, found only by
   tracing what `KernelResource` actually gets built for a CUDA-resolved
   node. Fixed with `resolved_kernel_memory_class(prepared_plan, node)`: a
   small helper that peeks the Prepared Plan's own `PlanNodeBinding` for
   this node (already resolved before dispatch) and returns `Device` for
   any non-Reference-CPU binding, `Host` otherwise (including no-plan
   fallback). Without this fix, Decision 1's residency passthrough would be
   correct but unreachable for CUDA -- every CUDA dispatch through this
   loop would fail before residency ever mattered.

### Decision 3: CUDA device allocation table keyed by `TensorResourceId`

`CudaExecutor` replaces `storage: Mutex<BTreeMap<TensorResourceId, HostTensor>>`
with `Mutex<BTreeMap<TensorResourceId, CudaDeviceBuffer>>`, where
`CudaDeviceBuffer` wraps a `cudarc::driver::CudaSlice<f32>` plus its
`TensorDescriptor` (shape/dtype/layout needed to validate later reads
without re-deriving it from bytes). `write_tensor`/`write_tensor_admitted`
(host-typed) continue to accept `HostTensor` and now perform a real
`clone_htod` into a newly allocated slot in this table (previously: stored
the `HostTensor` as-is). `read_tensor`/`read_tensor_value` perform
`clone_dtoh` only when host bytes are actually requested
(`read_tensor`/`TensorValue::into_host` path); `read_tensor_value` alone
(no forced host materialization) returns `TensorValue::Opaque` when the
resource already has a live device buffer, letting Decision 1's passthrough
apply. Kernel methods in `kernels.rs` change from
"upload input, compute, download output" to "look up input in the
allocation table (uploading only if the caller handed raw bytes via
`write_tensor`), compute into a newly allocated output slot in the same
table, return the output `TensorResourceId`" -- no implicit download.
`release`/`complete` (already part of `ProviderExecutionApi`'s lifecycle)
free the corresponding table entry.

**Alternative considered**: keep `HostTensor` storage and only change
control flow (skip re-upload when hashes match). Rejected: does not satisfy
"CUDA Provider Memory Manager Integration"'s literal requirement ("SHALL
allocate device memory... directly") and does not remove the actual PCIe
transfer cost the audit is concerned with, only a redundant Rust-side copy.

### Decision 4: No new `ExecutionStream`/`CompletionToken` trait surface -- `ProviderExecutionHandle` already is one

Re-reading `provider.rs`'s existing `ProviderExecutionApi` and the already-
archived `provider` capability spec (not just `execution-stream`'s own
file) during design showed this gap does not exist: `submit`/`submit_kernel`
already return an opaque `ProviderExecutionHandle`, `status` already gives a
non-blocking poll, `complete`/`complete_kernel` already give an explicit
wait-and-fetch, and `provider` spec's own "Synchronous Provider Still Uses
Completion Contract" requirement already normatively covers exactly the
CUDA baseline's synchronous case ("Reference CPU Provider SHALL be allowed
to execute synchronously while preserving CompletionToken semantics").
`ProviderExecutionHandle` *is* this change's `CompletionToken` -- inventing
a second, parallel opaque-handle type (`open_stream`/`submit_on_stream`/
`poll_completion`) would duplicate existing, already-conformant
infrastructure rather than fill a real gap. No `ProviderExecutionApi` trait
changes are needed for completion semantics; this change's actual fix is
entirely in Decisions 1-3 (stop copying a resource that is already where it
needs to be) and Decision 5-6 (error/health categories). `provider` is
therefore **not** a Modified Capability for this change -- removed from the
proposal's Capabilities section.

**Alternative considered (this session's own earlier draft)**: add
`open_stream`/`submit_on_stream`/`poll_completion`/`wait_completion`/
`drain_stream`. Rejected once the existing trait and spec text were
re-checked: `execution-stream`'s multi-stream ordering/`ExecutionStream`-as-
a-first-class-object requirements remain genuinely unimplemented by any
Provider, but that is real *future* async work (already a Non-Goal here),
not something this change's synchronous, single-Provider-passthrough fix
needs or should invent early.

### Decision 5: `KernelErrorCode::OutOfDeviceMemory`

Additive enum variant on `magnetar-runtime/src/kernel.rs`'s
`KernelErrorCode`, mirroring `ProviderExecutionErrorCode`'s existing OOM
category. `providers/cuda/src/error.rs`'s
`From<CudaError> for KernelError` mapping changes so
`CudaErrorCode::OutOfDeviceMemory` maps to this new variant instead of the
generic `KernelExecutionFailed`.

### Decision 6: `CudaProvider::health()` reports `Degraded`, not `Available`, when kernels failed to compile

`HealthState::Degraded` already exists and is already reachable through
`ProviderHealthState::Degraded` mapping (`affinity.rs`). No new health
state is invented; `CudaProvider::health()` simply returns
`HealthState::Degraded` when `self.device.is_some() && self.executor.is_none()`,
instead of `Available`. This is a one-line implementation fix closing a real
observability gap identified by the audit, not a spec change.

## Risks / Trade-offs

- [Resource-reference passthrough (Decision 1) only reaches Kernel dispatch
  through `execute_qwen_graph_nodes`, not the synthetic-candidate/test-oracle
  call sites that invoke `dispatch_qwen_matmul`/etc. directly] → those
  oracle call sites always construct `NodeInputValue::Host(...)` explicitly
  (never resolve through `read_tensor_value` themselves), so they cannot
  reach the `Resident` branch at all -- this is a property of how they're
  written, not something that needs separate gating logic to preserve.
- [`CudaDeviceBuffer` table (Decision 3) can grow unbounded across a long
  generation run if `release`/`complete` are ever skipped on an error path] →
  mitigated by reusing the existing `ProviderExecutionApi` lifecycle
  contract (`release` is already a required call on every resource path);
  covered by a regression test that repeated write/release cycles keep the
  table bounded.
- [Two structurally different input representations
  (`NodeInputValue::Host`/`::Resident`, Decision 2) permanently coexist in
  `dispatch_reference_cpu_operator_multi`] → same category of risk
  `define-provider-prepared-kernel-execution-contract` already accepted for
  `TensorValue::Host`/`::Opaque`; mitigated more strongly here, since the two
  variants are consumed by an exhaustive match in exactly one function, so
  no future call site can reintroduce an unconditional `write_tensor` for a
  `Resident` input without a compile error forcing it to be handled.
- [The output-side round-trip (Decision 7) stays in place] → a future change
  chaining more than one Provider-resident node deep still pays one D2H+H2D
  per node boundary on the output side, even after this change; documented
  explicitly (Decision 7, Open Questions) rather than left to be
  rediscovered as a surprise.
- [Changing `CudaExecutor`'s storage type is a breaking change to
  `providers/cuda`'s internal API, requiring a paired submodule commit like
  `generalize-first-native-provider-dispatch` needed] → same
  cross-repository coordination already exercised in this session; update
  `SUBMODULES.md`'s compatibility matrix the same way.

## Migration Plan

1. Land `magnetar-runtime` changes (Decisions 1, 2, 4, 5, 6) behind the
   existing default-implementation pattern so `providers/cpu` keeps building
   unmodified against the new trait surface.
2. Update `providers/cuda` (Decision 3) in the same working tree, verify
   locally against real hardware (RTX 3070 Ti, as done for the baseline).
3. Re-verify `providers/cpu` still passes conformance with zero code changes
   (proves the additive claim in Decision 4).
4. Fix `.github/workflows/gpu-runner-smoke.yml`'s stale description; dispatch
   it manually against the branch before merging, matching the verification
   rigor used for the CUDA baseline's own CI fix.
5. Push `providers/cuda` and `providers/cpu` submodule commits first, then
   the parent `Magnetar` commit pinning both, mirroring
   `generalize-first-native-provider-dispatch`'s already-proven sequencing.
6. No rollback complexity beyond reverting the pin: submodules are
   independently versioned and the parent commit is the sole integration
   point.

## Open Questions

- The output-side round-trip (Decision 7) is real follow-up work: unifying
  the Kernel-internal output resource id with the graph edge's id (or
  otherwise making the edge-level write a true no-op) needs a Memory
  Manager admission model that can represent "this Provider-internal
  allocation is *the same physical resource* as this Session-owned edge,"
  not two independent admissions. Worth scoping as its own change once a
  real workload's profile shows this is the dominant remaining cost --
  premature to design the admission-unification model speculatively here.
- `device-resident-resource`'s "Residency Survives Asynchronous Execution"
  and most of `execution-stream`'s multi-stream/ordering requirements stay
  genuinely unimplemented by any Provider after this change (Decision 4).
  That is an accepted gap here (Non-Goal: true async execution), left for a
  future change once a Provider actually needs overlap -- flagging so it is
  not mistaken for something this change silently closed.
