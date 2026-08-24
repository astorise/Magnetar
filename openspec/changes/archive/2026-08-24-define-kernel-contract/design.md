## Context

Magnetar already distinguishes portable Operators from Provider-owned native
execution. The missing layer is Kernel: a concrete implementation of an
Operator for a Provider and compatible Device context.

The current Runtime architecture requires Components to request portable
capabilities and prevents them from selecting Providers, Devices, native
handles, queues, streams, allocators, or function pointers directly. The Kernel
Contract must preserve that boundary while giving Runtime enough metadata for
future Kernel Registry and Dispatch work.

## Goals / Non-Goals

**Goals:**

- Introduce a first-class, platform-neutral Kernel contract in
  `magnetar-runtime`.
- Represent Kernel identity, advertisement metadata, invocation, result,
  fallback, conformance, observability, and structured errors.
- Validate Kernel compatibility against Operator metadata, tensor descriptors,
  memory class, workspace, Resource Affinity, execution mode, determinism, and
  precision policy before Provider dispatch.
- Keep the contract independent from direct client Provider or Device
  selection.
- Avoid raw memory, raw model data, raw prompts, raw KV cache contents, raw
  Provider handles, raw Device handles, and raw function pointers in public
  Kernel APIs.

**Non-Goals:**

- Define the full Kernel Registry or Dispatch algorithm.
- Implement CUDA, Metal, OpenVINO, QNN, WebGPU, or other native kernels.
- Add a Provider ABI for Kernel execution.
- Add graph optimization or fusion optimization.
- Move workspace allocation ownership away from Memory Manager.

## Decisions

1. Kernel is a metadata contract, not a Provider API.

   The implementation adds a `kernel` module with public contract types and
   validation helpers. Providers can advertise Kernel metadata later, but the
   current change does not add a native dispatch ABI. This keeps the contract
   useful for planning without coupling it to a backend implementation.

   Alternative considered: add Kernel methods to `ProviderExecutionApi`
   immediately. That would force ABI and dispatch decisions before the Kernel
   Registry change has defined selection semantics.

2. Kernel identity includes Provider and Operator identity.

   `KernelId` contains Provider ID, Kernel name and version, implemented
   Operator ID, Operator version range, feature flags, implementation family,
   optional build fingerprint, and optional conformance profile. It intentionally
   has no field for function pointers or native handles.

   Alternative considered: identify Kernels only by name. That would be
   ambiguous across Providers and would not be enough for compatibility checks.

3. Runtime validation is metadata-driven.

   `KernelAdvertisement::validate_invocation` validates Operator identity,
   Operator version range, Operator attributes, arity, shape, dtype, layout,
   memory class, workspace availability, Device constraints, Resource Affinity,
   execution mode, determinism, precision, and aliasing metadata.

   Alternative considered: let Providers reject incompatible invocations. That
   would make planning opaque and would allow Provider behavior to define
   portable semantics.

4. Workspace remains Memory Manager-owned.

   Kernel metadata declares workspace requirements and Kernel invocations only
   carry workspace allocation references. Kernels do not allocate workspace
   directly through this contract.

   Alternative considered: embed workspace allocation in Kernel invocation.
   That would blur Runtime and Provider ownership and weaken memory admission.

5. Browser compatibility is represented as metadata.

   Browser-compatible modes, memory classes, and structured unsupported-feature
   errors are part of the contract. The module does not depend on Wasmtime or
   native Provider loading, and it compiles for `wasm32-unknown-unknown`.

   Alternative considered: split browser Kernel contracts into a separate
   module. That would duplicate the same compatibility concepts before a real
   browser dispatch implementation exists.

## Risks / Trade-offs

- Broad metadata surface -> Mitigation: keep types plain, stable, and local to
  the `kernel` module until Registry/Dispatch needs narrower abstractions.
- Validation is not a full dispatcher -> Mitigation: explicitly scope this
  change to the contract; Registry and Dispatch remain separate future changes.
- Some metadata is optional or descriptive -> Mitigation: hard validation covers
  identity, Operator compatibility, tensor constraints, memory class, workspace,
  Resource Affinity, determinism, precision, and redaction-sensitive errors.
- Provider integration remains future work -> Mitigation: exports are available
  from the crate root so Provider advertisement can adopt them without changing
  public names.

## Migration Plan

1. Add the `kernel` module and root exports.
2. Add contract tests for Kernel identity, advertisement validation, mismatch
   errors, workspace handling, determinism, precision, and redacted
   observability.
3. Keep existing Provider, Scheduler, Memory Manager, Operator, and Execution
   Graph APIs source-compatible.
4. Future changes can add Kernel Registry and Provider advertisement plumbing
   using the public contract types.
