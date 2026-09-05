## Why

`reach-architecture-freeze-1` made first-native graph dispatch generically
Resource-ID-*addressed* (Correctif 5): every node-to-node value in
`execute_qwen_graph_nodes` is referenced by `TensorResourceId`, never held in
a private cache. But the *value* at each Resource ID is still `HostTensor` --
Reference CPU's own host-visible tensor representation -- because
`ProviderExecutionApi::read_tensor`/`write_tensor`/`write_tensor_admitted`
are `HostTensor`-typed on the trait itself. That change's own investigation
(tasks 3.3, 5.4-5.6) confirmed this blocks three concrete things it could not
close within its own scope:

- A genuinely device-resident (non-host) Provider cannot implement
  `ProviderExecutionApi` meaningfully -- `device-resident-resource`'s
  existing spec already requires a Tensor Resource be able to "exist and
  execute entirely Device-side without an authoritative host byte buffer,"
  which a `HostTensor`-typed trait method structurally cannot satisfy.
- Multi-output Kernel support (Correctif 5's task 5.4/5.5) has no natural
  home: every Reference CPU Kernel today calls `store_output` at index 0
  only, and there is no portable, Provider-agnostic way to describe "this
  Kernel produced N outputs" without first deciding what a portable output
  value looks like.
- Model Loading (task group 8's remaining gap) writes weight bytes through
  the same `HostTensor`-shaped path, coupling "how weights get into Provider
  storage" to Reference CPU's own representation even for artifacts that
  will eventually be parsed by non-Reference-CPU-specific format loaders.

No existing spec commits to a concrete answer for what a Provider-agnostic
tensor *value* representation looks like at the `ProviderExecutionApi`
boundary -- `tensor` and `device-resident-resource` specify Tensor Resource
*metadata*, residency, and lifecycle requirements, not the Rust-level
data-movement contract Kernels and Providers exchange values through. This
is that decision: what the audit's post-freeze review named Change B
(`define-provider-prepared-kernel-execution-contract`), deliberately left
out of `reach-architecture-freeze-1` (see that change's `design.md`,
"One Change, not three, for this audit round") because it is a new semantic
decision, not an implementation catching up to an already-correct spec.

## What Changes

- Define a Provider-agnostic tensor **value** contract for prepared Kernel
  execution: a data-movement interface `ProviderExecutionApi` exposes that
  does not require any Provider (Reference CPU included) to expose or
  consume `HostTensor` as the *only* representable value shape, while
  keeping the existing `TensorResourceId`-addressed transport
  (`reach-architecture-freeze-1`, Correctif 5) unchanged.
- Define how a submitted Kernel invocation reports **multiple** outputs
  under this contract (not just index 0), so a graph node can declare more
  than one output edge and have each resolve to its own Resource.
- Define the boundary precisely: Reference CPU's own `HostTensor` becomes
  one concrete, host-visible implementation of the new value contract
  (an adapter), not a rewrite of Reference CPU's internals -- this is a
  contract-level change to `ProviderExecutionApi`, not a new Kernel
  execution engine.
- Additive, not breaking: new `ProviderExecutionApi` methods
  (`read_tensor_value`/`write_tensor_value`/`write_tensor_value_admitted`)
  carry the new value type; the existing `HostTensor`-typed
  `read_tensor`/`write_tensor`/`write_tensor_admitted` keep their current
  signatures for hand-written test oracles and any caller that already
  knows it only talks to a host-visible Provider (see `design.md`,
  Decision 1). Reference CPU and the `MockKernelProvider` test double
  (added in `reach-architecture-freeze-1`) each implement both.

## Capabilities

### New Capabilities
- `provider-prepared-kernel-execution`: the Provider-agnostic tensor value
  and multi-output data-movement contract Kernels and Providers exchange
  values through during prepared execution, independent of any one
  Provider's internal tensor representation.

### Modified Capabilities
(none -- `provider`, `tensor`, and `device-resident-resource`'s existing
requirements describe behavior this change implements toward, not a
Rust-level API shape; none of their requirement text needs to change to
accommodate this contract.)

## Impact

- `magnetar-runtime/src/provider.rs`: `ProviderExecutionApi` trait --
  `read_tensor`/`write_tensor`/`write_tensor_admitted`/`release_admitted_tensor`
  signatures, and (for multi-output) how a Kernel's produced outputs are
  reported back to the caller.
- `magnetar-runtime/src/reference_cpu.rs`: `ReferenceCpuExecutor`'s storage
  and these trait methods' implementations adapt to the new value type;
  `HostTensor` itself is not removed, only no longer required at the trait
  boundary.
- `magnetar-runtime/src/first_native_runtime.rs`: `execute_qwen_graph_nodes`,
  `dispatch_reference_cpu_operator`, and the `dispatch_qwen_*` family read
  and write values through the new contract instead of `HostTensor`
  directly; `QwenDispatchContext`'s `provider` field type is unaffected
  (`Arc<dyn ProviderExecutionApi>`), only what flows through it.
- `magnetar-runtime/src/kernel.rs` / `kernel_dispatch.rs`: `KernelInvocation`
  and `KernelResult`'s output representation, for multi-output support.
- Test doubles: `MockKernelExecutor`/`MockKernelProvider`
  (`first_native_runtime/tests.rs`) implement the new contract too, so the
  non-Reference-CPU substitutability tests `reach-architecture-freeze-1`
  added keep exercising a real, independent implementation.
- Unblocks (in `reach-architecture-freeze-1`, tracked there once this
  Change lands): tasks 3.3, 5.4, 5.5, 5.6, and the "generic weight write"
  half of task group 8's remaining gap.
