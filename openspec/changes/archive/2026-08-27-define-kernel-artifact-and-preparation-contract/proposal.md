# Define Kernel Artifact And Preparation Contract

## Why

Magnetar currently models a Kernel as a concrete Provider-owned implementation
of a portable Operator.

This is sufficient for statically known kernels but does not yet model the
lifecycle required by dynamically generated or externally produced kernel
implementations.

Modern kernel-generation systems can produce hardware-specialized code or
intermediate representations for multiple execution targets.

Magnetar should be able to consume these generated implementations without
changing its core architectural invariants.

The Runtime must remain independent of:

- the system that generated the kernel
- the source language used
- the compiler implementation
- native Provider handles
- native function pointers
- hardware-specific executable object representation

This change introduces a first-class artifact and preparation model.

## What Changes

This change defines three distinct kernel lifecycle entities:

```text
KernelSourceArtifact
CompiledKernelArtifact
PreparedKernel
```

It also defines:

- kernel artifact identity
- artifact provenance
- source format identity
- target compatibility metadata
- compilation metadata
- artifact validation
- Provider preparation
- Prepared Kernel lifecycle
- Kernel Registry integration
- cold-path versus hot-path separation
- artifact replacement semantics
- observability and redaction
- conformance requirements

This change does not define how a Provider compiles source code.

That capability is defined by a later Provider Kernel Compilation change.

## Core Model

The canonical lifecycle is:

```text
external producer
      |
      v
KernelSourceArtifact
      |
      | compile / translate
      v
CompiledKernelArtifact
      |
      | Provider preparation
      v
PreparedKernel
      |
      | registration / selection
      v
Kernel Registry
      |
      v
Kernel Dispatch
```

These three entities SHALL remain semantically distinct.

## Kernel Source Artifact

A Kernel Source Artifact represents source code or an intermediate
representation that is not yet ready for direct execution by a Provider.

Examples MAY include:

```text
Triton source / IR
CUDA C++
PTX source
HIP source
WGSL
Metal Shading Language
SPIR-V text/intermediate representation
provider-specific IR
future generated kernel DSLs
```

The exact source format vocabulary SHALL be extensible.

Magnetar SHALL NOT define a closed enum containing all possible kernel
languages.

## Kernel Source Format

Kernel source formats SHALL use an extensible identity.

A conceptual representation MAY be:

```text
namespace:name@version
```

Examples:

```text
triton:source@3
nvidia:ptx@9
nvidia:cuda-cpp
amd:hip
webgpu:wgsl
apple:msl
khronos:spirv@1.6
vendor:custom-ir@2
```

Source format identity SHALL NOT imply Provider compatibility.

## Kernel Source Artifact Identity

Kernel Source Artifact identity SHOULD be content-addressed.

Identity SHOULD include at least a digest of immutable source bytes.

Metadata SHOULD include:

- artifact digest
- source format
- declared Operator or fused Operator group
- Operator semantic version requirements
- dtype constraints
- layout constraints
- shape specialization metadata
- target requirements
- compiler requirements
- generator provenance
- trust metadata
- creation metadata
- optional human-readable name

Human-readable names SHALL NOT be authoritative identity.

## Generated Kernel Provenance

Kernel Source Artifacts MAY record provenance describing how they were
produced.

Provenance MAY include:

- human-authored
- AI-generated
- optimizer-generated
- compiler-generated
- CI-generated
- vendor-provided
- imported

Provenance SHALL be descriptive only.

AI-generated status SHALL NOT grant or reduce trust automatically.

## Compiled Kernel Artifact

A Compiled Kernel Artifact represents a kernel implementation transformed into
a Provider-consumable executable or lower-level representation.

Examples MAY include:

```text
CUBIN
PTX
HSACO
SPIR-V
metallib
compiled WebGPU shader representation
native object code
Provider-specific binary blobs
```

Compiled Kernel Artifact SHALL remain data.

It SHALL NOT expose an executable pointer through Runtime public contracts.

## Compiled Artifact Identity

Compiled Kernel Artifact identity SHOULD be content-addressed.

Its metadata SHOULD record enough information to determine whether the artifact
can be reused safely.

Metadata SHOULD include:

- compiled artifact digest
- source artifact digest where known
- source format
- compiled format
- compiler identity
- compiler version
- compiler flags digest
- target architecture
- Provider compatibility
- runtime/driver compatibility requirements
- dtype constraints
- layout constraints
- shape specialization
- Operator semantic compatibility
- precision metadata
- determinism metadata
- trust/integrity state

## Prepared Kernel

A Prepared Kernel represents a Provider-owned executable kernel prepared for a
specific Provider and execution context.

Prepared Kernel is ephemeral Runtime state.

Prepared Kernel SHALL NOT be treated as a portable artifact.

Prepared Kernel MAY correspond internally to:

```text
CUDA function handle
loaded GPU module
Metal compute pipeline
Vulkan pipeline
WebGPU compute pipeline
CPU function pointer
OpenVINO compiled operation
QNN graph/kernel handle
other Provider-native execution state
```

These native objects SHALL remain Provider-private.

## Prepared Kernel Identifier

Runtime MAY identify a Prepared Kernel using an opaque identifier.

Example conceptual type:

```rust
PreparedKernelId
```

The identifier MAY internally be represented by an integer.

Its numeric value SHALL NOT expose or encode a native pointer.

Runtime SHALL treat it as opaque.

## Provider Ownership

Prepared Kernel native state SHALL be owned by the Provider.

Runtime SHALL NOT:

- dereference native kernel pointers
- store native function pointers
- reinterpret Provider handles
- expose native handles to Components
- expose native handles through WIT
- expose native handles through Runtime Inference API
- expose native handles through diagnostics by default

## Kernel Registry Ownership

Kernel Registry SHALL own logical metadata and selection state.

Kernel Registry MAY associate a Kernel candidate with:

```text
KernelId
KernelAdvertisement
KernelArtifactId
PreparedKernelId
ProviderBinding
DeviceBinding
```

Kernel Registry SHALL NOT own Provider-native executable pointers.

## Device Boundary

Device SHALL remain a hardware metadata and status abstraction.

Device SHALL NOT gain responsibilities for:

- source compilation
- source parsing
- artifact loading
- executable linking
- kernel preparation
- native function pointer resolution

Compilation and preparation belong to Provider-level capabilities.

## Scheduler Boundary

Scheduler SHALL NOT compile or prepare kernels.

Scheduler MAY depend on readiness information indicating whether required
kernels are prepared.

Scheduler MAY delay admission when kernel preparation is incomplete according
to Runtime policy.

## Execution Graph Boundary

Execution Graph SHALL continue to express portable Operators.

Execution Graph SHALL NOT embed:

- Provider-specific source code
- raw kernel source
- native binaries
- native kernel handles
- Device-specific executable pointers

Graph nodes MAY carry semantic requirements used by Kernel Registry to select
compatible Kernel implementations.

## Cold Path

Compilation and preparation SHALL be cold-path operations.

Cold-path operations MAY include:

```text
artifact discovery
source validation
compilation
translation
specialization
binary validation
qualification
benchmarking
Provider module loading
pipeline creation
kernel preparation
registry publication
```

These operations SHALL NOT occur synchronously inside an active token decode
hot path unless explicitly allowed by a future policy.

## Hot Path

Hot-path execution SHOULD consist only of operations such as:

```text
select or resolve already-prepared Kernel
bind Runtime-owned resources
submit Provider invocation
observe completion
```

Kernel compilation SHALL NOT happen in the normal token-generation hot path.

## Model Instance Readiness

Runtime MAY require all mandatory kernels for a Model Instance execution plan
to be prepared before marking that Model Instance ready.

Example:

```text
loading
  -> graph validated
  -> kernels resolved
  -> artifacts prepared
  -> Model Instance READY
```

A Model Instance SHALL NOT silently become ready when required Kernel
preparation has failed.

## Lazy Preparation

Runtime MAY support lazy preparation.

If lazy preparation is used:

- the operation SHALL be explicit in policy
- inference SHALL receive structured admission/backpressure state
- compilation SHALL not be silently inserted into the hot path
- readiness semantics SHALL remain explicit

## Artifact Trust

Kernel Source Artifacts and Compiled Kernel Artifacts SHALL have explicit trust
and integrity status.

Artifact format SHALL NOT imply trust.

Generated-by-AI status SHALL NOT imply trust.

Local origin SHALL NOT imply trust.

Cache presence SHALL NOT imply trust.

Trusted status SHALL be policy-controlled.

## Artifact Validation

Before preparation, Runtime and/or Provider SHALL validate artifact metadata.

Validation SHOULD cover as applicable:

- artifact digest
- format identity
- Operator compatibility
- dtype compatibility
- layout compatibility
- shape constraints
- target architecture
- Provider compatibility
- compiler/runtime compatibility
- precision requirements
- determinism requirements
- required device features
- trust/integrity state

## Semantic Compatibility

A Kernel Artifact SHALL declare the portable Operator semantics it implements.

For fused kernels, it SHALL declare the ordered or structured Operator group
whose semantics it preserves.

A generated kernel SHALL NOT redefine Operator semantics.

## Shape Specialization

Kernel Artifacts MAY be shape-specialized.

Specialization metadata MAY describe:

- exact dimensions
- bounded dimensions
- batch ranges
- sequence length ranges
- head counts
- head dimensions
- tile geometry
- alignment
- layout assumptions

Kernel Registry SHALL use specialization metadata during compatibility
selection.

## DType And Layout Specialization

Kernel Artifacts MAY specialize for dtype and layout.

Such specialization SHALL be explicit.

Runtime SHALL NOT silently reinterpret tensor dtype or layout to satisfy a
Kernel Artifact.

Required conversions SHALL remain explicit graph/runtime operations.

## Precision Metadata

Kernel Artifact metadata SHOULD describe numerical behavior where relevant.

Metadata MAY include:

- accumulation dtype
- approximate math
- tolerance profile
- deterministic tolerance
- reduction ordering assumptions
- fused operation semantics
- quantization behavior

## Compilation Cache Compatibility

Compiled Kernel Artifacts SHOULD contain enough metadata for future cache
compatibility decisions.

A future cache key MAY include:

```text
source digest
operator semantic version
compiler identity/version
compiler options
Provider version
target architecture
runtime/driver compatibility
dtype/layout
shape specialization
device features
```

This change does not define the cache policy itself.

## Artifact Replacement

A newer Kernel Artifact MAY coexist with an older artifact.

Replacement SHALL NOT require unloading the Provider.

Conceptually:

```text
Kernel Artifact v1
Kernel Artifact v2
Kernel Artifact v3
```

MAY all exist while Runtime controls which one is selected.

## Prepared Kernel Generations

Multiple Prepared Kernel generations MAY coexist temporarily.

This supports future hot replacement.

Example:

```text
PreparedKernel generation 17
    -> in-flight requests

PreparedKernel generation 18
    -> new requests
```

Older Prepared Kernels MAY be destroyed only after no active operation
references them.

## Provider Lifetime Independence

Replacing a Kernel Artifact SHALL NOT require unloading and reloading the
Provider.

Provider lifecycle and Kernel lifecycle SHALL remain separate.

## Memory Manager Boundary

Kernel preparation MAY create Provider-owned executable memory.

This SHALL NOT grant Provider ownership of Runtime Tensor Resources.

Memory Manager SHALL continue to own Runtime allocation and residency policy
for inference data.

Executable Kernel memory and Runtime Tensor memory SHALL remain distinct
concepts.

## Runtime API Boundary

Runtime Inference API SHALL remain Kernel Artifact independent.

Inference callers SHALL NOT provide:

- raw kernel source
- compiled binary blobs
- PreparedKernelId
- native handles
- compiler options

through normal generation requests.

Kernel artifact management belongs to Runtime initialization, deployment,
loading, tooling, or future authorized management APIs.

## Component Boundary

Portable Components SHALL NOT supply arbitrary executable kernel source
directly to Providers during inference.

Future explicitly authorized kernel-generation or kernel-import workflows MAY
exist outside ordinary inference Component authority.

## External Generator Boundary

Kernel generation MAY occur outside Magnetar Runtime.

Possible producers include:

```text
human engineers
CI systems
AI coding agents
optimization services
vendor toolchains
Tachyon services
future Magnetar tooling
```

Magnetar Runtime SHALL consume artifacts rather than depend on any specific
generator.

## Error Model

Kernel Artifact and Preparation failures SHALL use structured errors.

Error categories SHOULD include:

```text
kernel-artifact-invalid
kernel-artifact-digest-mismatch
kernel-artifact-format-unsupported
kernel-artifact-untrusted
kernel-artifact-operator-incompatible
kernel-artifact-dtype-incompatible
kernel-artifact-layout-incompatible
kernel-artifact-shape-incompatible
kernel-artifact-target-incompatible
kernel-artifact-provider-incompatible
kernel-artifact-driver-incompatible
kernel-artifact-compiler-incompatible
kernel-preparation-unavailable
kernel-preparation-failed
kernel-prepared-handle-invalid
kernel-prepared-generation-in-use
kernel-prepared-destroy-failed
kernel-prepared-not-ready
kernel-hot-path-compilation-denied
internal-kernel-artifact-error
```

Errors SHALL NOT expose native pointers or executable code contents by default.

## Observability

Observability MAY report:

- Kernel Source Artifact discovered
- artifact validated
- compiled artifact selected
- preparation started
- preparation completed
- preparation failed
- Prepared Kernel registered
- Prepared Kernel selected
- Prepared Kernel retired
- Prepared Kernel destroyed
- artifact replacement occurred
- hot-path compilation denied

Observability SHALL redact:

- raw kernel source by default
- raw executable binary bytes
- native Provider handles
- native function pointers
- raw device pointers
- secrets
- credentials
- local filesystem paths where policy requires redaction

## Conformance

Conformance SHALL validate:

- source, compiled, and prepared kernel entities remain distinct
- Device does not compile kernels
- Scheduler does not compile kernels
- Runtime does not expose native Provider handles
- PreparedKernelId is opaque
- Kernel Registry does not own native executable pointers
- Provider owns native prepared state
- hot-path compilation is rejected by default
- artifact format does not imply trust
- AI-generated provenance does not imply trust
- Kernel Artifact semantics match Operator semantics
- shape/dtype/layout specialization is explicit
- Provider lifetime is independent from Kernel replacement
- multiple Prepared Kernel generations may coexist safely

## Non-Goals

This change does not:

- implement Triton compilation
- implement CUDA compilation
- implement WGSL compilation
- implement Metal compilation
- define Provider compilation ABI
- define generated-kernel qualification
- define kernel benchmark policy
- define Kernel Artifact cache eviction
- define hot reload of Providers
- expose native kernel handles
- allow Components to inject arbitrary executable code during inference
- place kernel generation agents inside Runtime

## Impact

Magnetar gains a first-class lifecycle for generated and externally produced
Kernel implementations:

```text
source
  -> compiled artifact
  -> prepared Provider state
  -> Registry
  -> hot-path execution
```

while preserving the existing architecture:

```text
Operator = portable semantics
Kernel = concrete implementation
Provider = hardware/runtime integration
Device = hardware metadata/status
Runtime = inference orchestration
```