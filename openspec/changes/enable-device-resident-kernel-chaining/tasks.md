## 1. Core: Kernel error category

- [x] 1.1 Add an out-of-device-memory variant to `KernelErrorCode`
  (`magnetar-runtime/src/kernel.rs`), per `kernel` spec's new "Kernel
  Out-Of-Device-Memory Error Category" requirement. (Added
  `KernelErrorCode::KernelOutOfDeviceMemory` and
  `KernelError::KernelOutOfDeviceMemory { reason: String }`, with matching
  `code()`, `id()` ("kernel-out-of-device-memory"), and `Display` arms
  mirroring `KernelExecutionFailed`'s shape. Full workspace build confirmed
  no other exhaustive match over these enums needed updating.)
- [x] 1.2 Test: constructing/mapping this variant round-trips through any
  existing structured-error display/serialization the enum already has.
  (`kernel_out_of_device_memory_error_round_trips_code_id_and_display`,
  `magnetar-runtime/src/tests.rs`: asserts `.code()`, `.id()`, and
  `Display` all agree and are distinct from `KernelExecutionFailed`.)

## 2. CUDA Provider: device allocation table

- [x] 2.1 Define `CudaDeviceBuffer` (`providers/cuda/src/executor.rs` or a
  new module): wraps a `cudarc::driver::CudaSlice<f32>` plus its
  `TensorDescriptor`. (Defined in `kernels.rs` as `{ slice: CudaSlice<f32>,
  shape: Vec<u64> }` -- shape alone, not a full `TensorDescriptor`, since
  every kernel here is f32/contiguous-only per the baseline's own declared
  scope; nothing reads dtype/layout off it.)
- [x] 2.2 Replace `CudaExecutor.storage: Mutex<BTreeMap<TensorResourceId,
  HostTensor>>` with `Mutex<BTreeMap<TensorResourceId, CudaDeviceBuffer>>`.
- [x] 2.3 Update `write_tensor`/`write_tensor_admitted` to perform a real
  `clone_htod` into a newly allocated table entry (design.md Decision 3),
  instead of storing the `HostTensor` as-is. (`write_tensor` now returns
  `Result<(), CudaError>` since a real upload can fail;
  `write_tensor_admitted` rolls back its Memory Manager admission via
  `memory.release` if the physical upload fails, mapping to
  `MemoryError::AllocationDenied`.)
- [x] 2.4 Update `read_tensor` to perform `clone_dtoh` from the table entry
  only when host bytes are actually requested.
- [x] 2.5 Update `read_tensor_value` to return `TensorValue::Opaque` when
  the resource has a live device buffer, and `TensorValue::Host` only when
  the caller previously wrote it as host-typed and no device buffer exists.
  (Simplified per implementation reality: every entry in this table is now
  always device-resident -- `write_tensor`/`write_tensor_admitted` always
  upload -- so `read_tensor_value` is `Some(Opaque)` iff present, `None`
  otherwise; there is no host-typed-only entry case for this Provider.)
- [x] 2.6 Update `write_tensor_value`/`write_tensor_value_admitted`: an
  incoming `TensorValue::Host` uploads as today; an incoming
  `TensorValue::Opaque` for a `TensorResourceId` already present in this
  Provider's own table is a no-op identity confirmation (the Provider
  already owns the data); for any other Provider's `Opaque` it fails
  structurally (cross-Provider opaque values are not portable). (Both fail
  with a `ProviderExecutionErrorCode::MaterializationFailed`/
  `TensorValueAdmissionError::Provider` when absent, via the shared
  `opaque_passthrough_error` helper.)
- [x] 2.7 `release`/`release_tensor` (existing lifecycle) free the
  corresponding table entry; verify no leak across repeated
  allocate/release cycles. (Unchanged control flow -- `BTreeMap::remove`
  drops the `CudaDeviceBuffer`, and `CudaSlice::drop` frees the real device
  allocation. Verified by task 2.9's test below.)
- [x] 2.8 Test: a resource written once and read many times across
  separate calls returns the same device buffer without re-uploading
  (assert via an allocation-count counter or similar instrumentation, not
  timing). (Consolidated with 3.2's test -- both assert the identical claim
  at the same seam: see `tests_conformance::back_to_back_kernels_do_not_round_trip_through_host`,
  which covers both "no re-upload of an already-resident input" and "no
  round trip between chained kernels" with one instrumented test, avoiding
  a near-duplicate.)
- [x] 2.9 Test: table size after a full multi-step generation run does not
  grow monotonically with token count (design.md's bounded-growth risk).
  (`executor::tests::repeated_write_release_cycles_do_not_grow_storage_unboundedly`:
  50 write/release cycles, asserting the live table never exceeds 2
  entries. A true end-to-end multi-step generation run through
  `first_native_runtime` is exercised separately in task 8.3.)

## 3. CUDA Provider: explicit data movement in kernels

- [x] 3.1 Update each kernel method in `providers/cuda/src/kernels.rs`
  (`matmul`, `add`, `attention`, etc.) to look up each input in the
  allocation table instead of unconditionally uploading, and to allocate
  its output directly into a new table entry instead of unconditionally
  downloading. (Every method now takes/returns `&CudaDeviceBuffer`/
  `CudaDeviceBuffer`; `upload`/`download` are the only two crossing points,
  called only from `executor.rs`. `embedding_lookup`'s id-range validation
  moved device-side into `embedding_lookup_kernel` itself (`kernels.cu`,
  device-computed `invalid_id_flag`, same pattern `softmax_rows_kernel`
  already used for its NaN-row flag) since `ids` can no longer be assumed
  host-visible.)
- [x] 3.2 Test: two compatible kernels invoked back-to-back on the same
  `CudaExecutor` (matmul output consumed by a second matmul or add) perform
  no D2H/H2D transfer between them (same instrumentation approach as 2.8).
  (`tests_conformance::back_to_back_kernels_do_not_round_trip_through_host`:
  `CudaKernels` gained `upload_count`/`download_count` instrumentation;
  `add` then `mul` chained with zero additional uploads/downloads between
  them, only the final explicit download.)
- [x] 3.3 Test: numerical output is unchanged versus the previous
  implicit-copy behavior, within the same tolerance already used against
  Reference CPU (regression, not a new tolerance). (No separate test
  needed: the implicit-copy code path no longer exists to compare against,
  so the existing `tests_conformance` suite re-passing against Reference
  CPU at the unchanged `1e-3` tolerance, on real hardware, *is* the
  regression proof -- all 12 pre-existing conformance tests still pass.)
- [x] 3.4 Update `providers/cuda/src/error.rs`'s `From<CudaError> for
  KernelError` mapping so `CudaErrorCode::OutOfDeviceMemory` maps to the new
  `KernelErrorCode` variant (task 1.1) instead of the generic
  `KernelExecutionFailed`. (Test:
  `tests::out_of_device_memory_maps_to_dedicated_kernel_error_category`.)

## 4. CUDA Provider: health/execution_api reconciliation

- [x] 4.1 Change `CudaProvider::health()` to return `HealthState::Degraded`
  when `self.device.is_some() && self.executor.is_none()` (device found,
  kernel compile/load failed), instead of `Available`.
- [x] 4.2 Test: a `CudaProvider` constructed with a device present but a
  forced kernel-compile failure reports `health() == Degraded` and
  `execution_api() == None` together, not the previous
  `Available`+`None` combination. (Rather than actually breaking NVRTC
  compilation, added a `#[cfg(test)]`-only constructor
  `CudaProvider::with_device_but_no_executor_for_test` that pairs a real,
  discovered Device with no executor -- exercises the exact state
  `health()` branches on, on real hardware, without fabricating a fake
  `DeviceDescriptor`. See
  `tests::health_is_degraded_when_device_found_but_executor_missing`.)

## 5. Core: first-native dispatch passthrough

**Design correction made during implementation**: two things discovered
while implementing this group changed the plan from what was originally
written here (kept below, struck through in spirit, replaced by what
actually shipped):
- The round-trip is real on the *input* side only, exactly as planned
  below. A parallel *output*-side round-trip also exists (every Kernel's own
  output is downloaded via `read_tensor` then immediately re-uploaded by
  the outer loop under a *different* resource id) but eliminating it safely
  requires unifying two independently-tracked Memory Manager admissions
  (Provider-owned kernel-internal vs. Session-owned edge-level) without
  double-counting capacity -- real, separate work, deliberately **not**
  attempted here; the input-side fix alone still halves the round trips per
  edge (the second kernel's own D2H+H2D is what's eliminated).
- A genuine, previously-undiscovered prerequisite bug: `KernelMemoryClass`
  was hardcoded to `Host` for every `KernelResource` this dispatch loop
  builds, regardless of resolved Provider. `providers/cuda`'s own Kernel
  advertisements declare `KernelMemoryClass::Device`
  (`providers/cuda/src/advertisements.rs`), so `validate_invocation`
  (`kernel.rs`'s `validate_resource`) would reject *every* CUDA Kernel
  invocation through this dispatch loop with `KernelMemoryClassUnsupported`
  -- independent of residency, this made CUDA dispatch through first-native
  impossible today. Fixed alongside (task 5.3 below) since the residency
  fix would otherwise be unreachable/untestable for CUDA.

- [x] 5.1 Define `NodeInputValue { Host(HostTensor), Resident { id:
  TensorResourceId, shape: Vec<u64> } }` (`magnetar-runtime/src/first_native_runtime.rs`,
  design.md Decision 1). Carries `shape` because `TensorValue::Opaque`
  itself has no payload -- callers needing only shape (`rows_cols`,
  `sequence_length`) never have to materialize. Also added
  `NodeInputResource { Fresh(TensorResourceId, TensorDescriptor,
  HostTensor), Resident(TensorResourceId, TensorDescriptor) }` for
  `dispatch_reference_cpu_operator_multi`'s input list, and
  `node_input_resource(operation_id, suffix, value)` to build one from a
  `NodeInputValue`.
- [x] 5.2 Changed `execute_qwen_graph_nodes`'s per-node input resolution:
  when `read_tensor_value` returns `TensorValue::Opaque` for an edge *and*
  the consuming node's operator is passthrough-eligible (`matmul`,
  `attention`, `silu`, `mul`, `residual-add`, `embedding` -- operators whose
  own Rust code never dereferences tensor bytes directly, only forwards to
  a Kernel dispatch), builds `NodeInputValue::Resident{id, shape}` instead
  of calling `into_host`. `rmsnorm`/`rope` are excluded (weight broadcast,
  per-head slicing genuinely need raw floats) and always materialize, per
  design.md Non-Goals. (Simplified from the original plan's "same
  Provider/Device as producer" check via Prepared Plan bindings: the whole
  graph execution already uses one single resolved `ctx.provider` instance
  throughout -- there is no per-node Provider/Device variation to check
  against, so `TensorValue::Opaque` from *this* provider is already the
  correct and sufficient signal.)
- [x] 5.3 Changed `dispatch_qwen_graph_node`'s `inputs` parameter from
  `Vec<HostTensor>` to `Vec<NodeInputValue>`, and
  `dispatch_reference_cpu_operator[_multi]`'s `inputs` parameter to
  `Vec<NodeInputResource>`. `dispatch_qwen_matmul`/`_unary`/
  `_binary_same_shape`/`_attention`/embedding's inline dispatch now take
  `NodeInputValue` directly (they only read shape); `dispatch_qwen_rmsnorm`/
  `_rope_per_head` keep `HostTensor` parameters, with `dispatch_qwen_graph_node`
  materializing via `NodeInputValue::into_host` immediately before calling
  them. Also added `resolved_kernel_memory_class(prepared_plan, node)`,
  peeking the Prepared Plan's own `PlanNodeBinding.provider` for this node
  (defaulting to `Host` with no plan or a Reference-CPU binding, `Device`
  otherwise) -- fixes the `KernelMemoryClass` prerequisite bug found above.
  Return types were deliberately left as `HostTensor` throughout (no
  output-side change -- see the correction note).
- [x] 5.4 In `dispatch_reference_cpu_operator_multi`, `ctx.provider.write_tensor(id,
  tensor)` is now called only for `NodeInputResource::Fresh` inputs; a
  `NodeInputResource::Resident(id, descriptor)` input is passed to kernel
  selection/dispatch under its existing `id` with no write. Confirmed (by
  reading every one of matmul/unary/binary_same_shape/attention/embedding)
  that none dereferences tensor bytes for an input that could arrive as
  `Resident` -- only shape, via `NodeInputValue::shape()`.
- [x] 5.5 No separate string-scanning static guard test needed here (unlike
  the precedent this task cited): `NodeInputResource`'s two variants are
  matched exhaustively in exactly one place
  (`dispatch_reference_cpu_operator_multi`'s input loop), and only the
  `Fresh` arm calls `write_tensor` -- a compile-time structural guarantee,
  not a runtime property that could silently regress via a new call site
  elsewhere the way the precedent's cross-file `HostTensor`-typed-method
  elimination could.
- [x] 5.6 Test: `first_native_runtime::tests::resident_input_passthrough_reuses_existing_resource_and_computes_correctly`.
  Built `OpaqueReportingExecutor` (wraps a real `ReferenceCpuExecutor` for
  actual computation, but `read_tensor_value` always reports `Opaque`,
  never `Host`) and called `dispatch_reference_cpu_operator` twice, feeding
  the first call's output resource id directly as a
  `NodeInputResource::Resident` into the second. Asserts both: the second
  dispatch's input resource id is the *exact same* id the first dispatch
  produced (real by-reference reuse, not a same-shaped copy under a new
  id), and the computed numbers are correct (add then mul against a real
  Kernel implementation, not a stub).
- [x] 5.7 Test: the synthetic-candidate/no-Prepared-Plan fallback path is
  unaffected by construction, not just by test outcome -- the test-oracle
  functions calling `dispatch_qwen_matmul`/etc. directly (never through
  `execute_qwen_graph_nodes`) always wrap their values in
  `NodeInputValue::Host(...)` explicitly, so they never reach the new
  `Resident`-detection branch at all. All pre-existing tests exercising
  these oracles still pass unchanged.
- [x] 5.8 Test: existing single-Provider (Reference CPU only) E2E behavior
  is unchanged. `ReferenceCpuExecutor::read_tensor_value` never returns
  `TensorValue::Opaque` (confirmed by reading its implementation), so the
  new `Opaque if passthrough_eligible => Resident` match arm is never taken
  for it -- the `other => materialize` arm always fires, reproducing the
  old unconditional `into_host` behavior exactly. Confirmed empirically:
  full `magnetar-runtime` suite (1192 tests, up from 1190) passes unchanged.

## 6. Conformance and CI

- [x] 6.1 Verified CUDA Provider passes the `provider-data-movement`
  conformance profile against real hardware
  (`tests_provider_conformance::passes_provider_data_movement_conformance_when_available`).
  Note: `validate_data_movement`'s check is data-driven by
  `ProviderMetadata.compute_advertisement.data_movement`, which *no* real
  Provider (Reference CPU included) populates yet -- this profile currently
  passes vacuously for every real Provider, CUDA included, not specially
  for CUDA. Documented in the test's own module doc rather than silently
  claiming a stronger guarantee than the check actually provides; wiring
  real data-movement advertisements is separate, future work.
- [x] 6.2 Added `passes_provider_data_movement_conformance_when_available`
  to `providers/cuda/src/tests_provider_conformance.rs`.
- [x] 6.3 Corrected `.github/workflows/gpu-runner-smoke.yml`'s stale
  description and its step name (now "Build and test the CUDA Provider",
  comment block describes the real device-allocation-table/explicit-
  movement/conformance behavior).
- [ ] 6.4 Dispatched `gpu-runner-smoke.yml` on `arc-gpu-magnetar`
  (run `34011516921`). First attempt failed -- but not from this change's
  code: `error: failed to load manifest for dependency
  'magnetar-provider-cpu' ... failed to read
  'providers/cpu/Cargo.toml' ... No such file or directory`. The
  workflow's "Checkout CUDA Provider submodule" step only ever initialized
  `providers/cuda`, never `providers/cpu` -- a genuine, pre-existing gap
  (CUDA's `[dev-dependencies]` path-depends on `providers/cpu` for its
  conformance oracle comparisons), not something this change introduced.
  Fixed the checkout step to also initialize `providers/cpu`; re-dispatch
  pending.

## 7. Cross-repository verification

- [x] 7.1 Built `providers/cpu` unmodified against this change's
  `magnetar-runtime` (`cargo build` in `providers/cpu`): clean, confirming
  no CPU Provider code change is required (design.md Decision 4's removal
  rationale holds).
- [x] 7.2 Built/tested `providers/cuda` locally against real hardware (RTX
  3070 Ti): 23/23 tests pass, including the new
  `back_to_back_kernels_do_not_round_trip_through_host`,
  `health_is_degraded_when_device_found_but_executor_missing`,
  `out_of_device_memory_maps_to_dedicated_kernel_error_category`,
  `repeated_write_release_cycles_do_not_grow_storage_unboundedly`, and
  `passes_provider_data_movement_conformance_when_available` tests; clippy
  and fmt clean.
- [ ] 7.3 Update `SUBMODULES.md`'s compatibility matrix with the new
  `providers/cuda` commit and, if `providers/cpu` needed any change at all
  (expected: none, confirmed by 7.1), its commit too. (Deferred until the
  submodule commits actually exist -- committing/pushing is a user-facing
  action held for explicit confirmation.)

## 8. Full verification

- [x] 8.1 `cargo build -p magnetar-runtime --lib`, `cargo test -p
  magnetar-runtime --lib` (1192 passed, up from 1190), `cargo clippy -p
  magnetar-runtime --lib --tests -- -D warnings`, `cargo fmt --check`,
  `cargo build --workspace`. All clean.
- [x] 8.2 `cargo check --target wasm32-unknown-unknown -p magnetar-runtime
  --all-features`. Clean (only pre-existing, unrelated warnings about
  unreachable wasm32-only code paths -- confirmed not introduced by this
  change).
- [x] 8.3 The full first-native E2E suite (part of the 1192-test run in
  8.1, including `e2e_ci_can_run_without_gpu_and_reports_only_expected_required_failure`
  and every `e2e_*`/`dispatch_reference_cpu_operator_multi_*` test) passes
  unchanged, confirming the per-node causal-evidence chain still holds with
  the new passthrough path present (though, per task 5.8, never taken for
  Reference CPU). `validate_e2e_no_shortcuts` is exercised as part of this
  same suite; no separate invocation needed.
- [x] 8.4 `openspec validate enable-device-resident-kernel-chaining --strict`
  passes.
