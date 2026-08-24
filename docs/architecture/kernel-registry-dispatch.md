# Kernel Registry And Dispatch

Kernel Registry is a Runtime-owned metadata index for validated Kernel
advertisements. Providers may advertise Kernels, and Runtime test fixtures may
seed Kernels for contract tests. Clients and Components do not register Kernels
directly.

Kernel Registry is separate from the Kernel Contract. The Kernel Contract
defines portable metadata, validation, invocation, result, conformance, and
error structures. Kernel Registry owns admission, indexing, invalidation,
candidate lookup, selection, and policy-aware ranking of those Kernel
advertisements.

Kernel Dispatch is the Runtime-owned submission path. It turns a selected
Kernel Candidate into a Dispatch Plan and a Runtime-created Kernel Invocation,
then revalidates Provider, Device, memory, Resource Affinity, lifecycle,
cancellation, and policy state immediately before dispatch. Dispatch fails
closed if the selected Kernel becomes stale.

The Scheduler may use Kernel metadata for batching, deadlines, backpressure,
and pressure-aware planning, but final Kernel validation and invocation creation
remain in Runtime Dispatch. Execution Graph planning produces Operator
invocations and Kernel requirements; graphs do not embed raw native Kernel
pointers.

Selection preserves Resource Affinity. If resources are Provider-bound or
Device-bound, Runtime selects only compatible Kernels unless explicit movement,
conversion, or rebuild steps are present in the Dispatch Plan. Memory Manager
integration is represented through output, workspace, staging, movement, dtype
conversion, and layout conversion feasibility metadata.

Fallback is explicit. A fallback chain may include alternate Kernels, alternate
Providers, alternate Devices, explicit dtype conversion, explicit layout
conversion, explicit data movement, host execution, or rejection. Fallback does
not silently override Resource Affinity, dtype, layout, memory, determinism,
precision, conformance, or Provider policy.

Dispatch results return structured metadata: selected Kernel, Provider, Device,
success or failure, output readiness, updated resources, timing, fallback,
cancellation, determinism, precision, diagnostics, and structured errors. The
registry and dispatch APIs do not expose raw Provider handles, Device handles,
native function pointers, memory pointers, raw tensor values, prompts, weights,
or KV cache contents.

Browser targets use the same platform-neutral metadata. Browser-compatible
Providers can advertise WebAssembly linear memory, JavaScript-mediated
execution, or future WebGPU buffer metadata. Native-only dispatch features
return structured unsupported-feature errors when no browser-compatible path is
available.

This contract does not define concrete CUDA, Metal, QNN, OpenVINO, WebGPU, or
CPU kernels; Provider Kernel ABI; a graph optimizer; model architecture
Components; distributed Kernel dispatch; remote execution; or any client-visible
raw execution handle.
