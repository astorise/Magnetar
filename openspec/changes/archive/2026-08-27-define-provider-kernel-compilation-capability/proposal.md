# Define Provider Kernel Compilation Capability

## Why

Magnetar now distinguishes:

```text
KernelSourceArtifact
CompiledKernelArtifact
PreparedKernel
```

but does not yet define how a Provider advertises and performs the cold-path
transformation between these states.

Generated kernels introduce a heterogeneous compilation landscape:

```text
Triton       -> PTX/CUBIN or Provider-specific output
CUDA C++     -> PTX/CUBIN
HIP          -> HSACO
WGSL         -> WebGPU pipeline representation
MSL          -> Metal library/pipeline
SPIR-V       -> Vulkan pipeline
vendor IR    -> accelerator executable
```

The Runtime must support this without:

- teaching Device how to compile code
- teaching Scheduler how to invoke compilers
- exposing compiler internals through inference APIs
- exposing native Provider handles
- requiring one global `TargetLang` enum
- compiling during token decode
- coupling Magnetar to Triton, CUDA, KernelEvolve, or any particular generator

This change introduces an optional, versioned Provider Kernel Compilation
Capability.

## What Changes

This change defines:

- Provider compilation capability discovery
- accepted source formats
- produced compiled formats
- compilation modes
- compilation target description
- compilation jobs
- asynchronous compilation
- cancellation and deadlines
- resource limits
- compiler identity
- deterministic/reproducible compilation metadata
- controlled compiler authority
- compilation isolation
- compiled artifact validation
- preparation boundary
- Provider ABI extension model
- structured failures
- observability
- conformance

## Architectural Rule

Compilation belongs to Provider capability, not Device.

```text
Runtime
   |
   | KernelSourceArtifact + semantic requirements
   v
Provider Kernel Compilation Capability
   |
   | Provider-specific toolchain
   v
CompiledKernelArtifact
```

Device continues to represent only hardware identity, capabilities, health,
availability, limits, and pressure.

## Compilation Versus Preparation

Compilation and preparation SHALL remain distinct.

Compilation:

```text
KernelSourceArtifact
     ->
compiler / translator
     ->
CompiledKernelArtifact
```

Preparation:

```text
CompiledKernelArtifact
     ->
Provider loader / driver / pipeline creation
     ->
PreparedKernel
```

Execution:

```text
PreparedKernel
     ->
Provider execution
```

A Provider MAY internally combine compilation and preparation where the native
platform requires it, but Magnetar's logical contract SHALL preserve the three
states.

## Capability Discovery

A Provider MAY advertise a Kernel Compilation Capability.

Compilation support SHALL NOT be assumed merely because the Provider can
execute Kernels.

Example:

```text
Reference CPU Provider
    execution: yes
    compilation: no

CUDA Provider
    execution: yes
    compilation:
        nvidia:ptx -> nvidia:cubin
        optional triton/source -> nvidia:ptx

WebGPU Provider
    execution: yes
    compilation/preparation:
        webgpu:wgsl -> browser pipeline

Metal Provider
    execution: yes
    compilation:
        apple:msl -> apple:metallib/pipeline
```

Exact support is Provider-specific.

## Compilation Capability Identity

Compilation capability SHALL be independently versioned.

A conceptual capability identifier MAY be:

```text
magnetar:provider/kernel-compilation@1
```

Its version SHALL NOT automatically track:

- Magnetar crate version
- Provider version
- Provider ABI version
- WIT versions
- Kernel Artifact manifest version

## Compilation Capability Descriptor

A Provider compilation capability SHOULD advertise:

- capability version
- accepted Kernel Source Formats
- emitted Compiled Kernel Formats
- supported Devices or Device classes
- supported target architectures
- compilation modes
- asynchronous support
- cancellation support
- deadline support
- maximum source size
- maximum output size
- maximum concurrent compilation jobs
- compiler identity availability
- reproducibility metadata support
- isolation model
- required external authorities
- specialization support

## Compilation Modes

Providers MAY advertise one or more compilation modes.

Suggested portable categories:

```text
SourceCompilation
IntermediateTranslation
BinarySpecialization
ShaderCompilation
PipelineCompilation
OfflineAot
LoadTimeJit
ProviderManaged
```

These describe behavior rather than specific languages.

`LoadTimeJit` SHALL NOT mean token-loop compilation.

It means JIT during an explicit cold-path preparation/loading phase.

## Source Format Negotiation

Runtime SHALL compare Kernel Source Artifact format against Provider advertised
accepted formats.

Unsupported formats SHALL be rejected before compilation.

Runtime SHALL NOT infer support from filename extensions.

Runtime SHALL NOT infer support from Provider name.

## Output Format Negotiation

A Provider SHALL identify the produced Compiled Kernel Artifact format.

Compilation result SHALL not be an untyped binary blob.

Example:

```text
input:
    triton:source@3

output:
    nvidia:ptx@9
```

or:

```text
input:
    apple:msl

output:
    apple:metallib
```

## Compilation Input

Compilation SHALL consume an explicit immutable artifact input.

Conceptually:

```text
KernelCompilationRequest
    request_id
    source_artifact
    target
    specialization
    policy
```

The source payload SHOULD be represented as bytes rather than `String`,
because not all input formats are textual.

The request SHALL NOT carry arbitrary host filesystem paths.

## Compilation Target

Runtime MAY supply a portable compilation target derived from the selected
Provider and Device.

Target metadata MAY include:

- Provider binding
- Device binding
- hardware architecture
- hardware feature set
- target ABI
- target execution environment
- requested dtype/layout specialization
- shape specialization
- precision requirements
- determinism requirements

It SHALL NOT contain native Device handles or pointers.

Provider SHALL resolve DeviceBinding to its private native Device state.

## Runtime Selection Authority

The Provider SHALL NOT use compilation as a mechanism to choose arbitrary
Providers or Devices.

Runtime remains authoritative for Provider and Device selection.

The compilation capability receives the Runtime-selected target.

## Compiler Identity

Compilation result SHALL record compiler identity where available.

Compiler identity SHOULD include:

- compiler name
- compiler version
- backend version
- toolchain fingerprint
- relevant flags fingerprint

Examples MAY include:

```text
triton 3.x
nvcc 14.x
ptxas version
shaderc version
Metal compiler version
Provider internal compiler version
browser implementation/compiler fingerprint
```

Exact strings are Provider-specific metadata.

## Compiler Flags

Compiler options affecting executable behavior or compatibility SHALL
participate in compiled artifact identity.

Runtime SHOULD record a deterministic fingerprint rather than logging complete
compiler command lines by default.

Raw command lines may contain paths or environment information and SHALL be
redacted by default.

## Specialization Inputs

Compilation MAY specialize kernels for:

- shapes
- batch size ranges
- sequence lengths
- attention dimensions
- dtype
- layout
- alignment
- quantization
- device features

Specialization SHALL be explicit and represented in resulting artifact
metadata.

Hidden specialization SHALL NOT be allowed.

## Asynchronous Compilation

Compilation SHOULD support asynchronous execution where the Provider/toolchain
can do so.

Compilation SHALL be modeled as a job lifecycle.

Suggested states:

```text
queued
compiling
succeeded
failed
cancelled
timed-out
```

A compilation job SHALL have an opaque Runtime-visible identifier.

It SHALL NOT expose process IDs, thread pointers, compiler object pointers, or
native driver handles.

## Compilation Job Ownership

Provider owns native compilation-job state.

Runtime owns orchestration and policy around the job.

Runtime MAY:

- submit
- poll
- await
- cancel
- time out
- discard result

Runtime SHALL NOT access Provider-native job internals.

## Compilation Deadlines

Runtime MAY impose a compilation deadline.

Provider capability SHALL declare whether deadlines are enforceable.

When Runtime policy requires enforceable deadlines and Provider cannot enforce
them, compilation SHALL fail closed.

Compilation timeout SHALL NOT leave a Compiled Kernel Artifact marked ready.

## Compilation Cancellation

Provider SHALL declare cancellation semantics.

Suggested categories:

```text
NotSupported
BeforeStartOnly
Cooperative
Interruptible
ProviderSpecific
```

A cancelled job SHALL NOT publish a partially generated artifact as valid.

## Compilation Resource Limits

Runtime policy MAY impose limits such as:

- source bytes
- output bytes
- wall-clock duration
- concurrent jobs
- temporary workspace
- host memory
- device compiler memory

Provider SHALL reject requests exceeding enforced limits.

## Compiler Authority

Kernel compilation may require more host authority than kernel execution.

Examples include:

```text
temporary files
subprocess execution
compiler toolchain files
driver APIs
shader compiler APIs
```

This authority SHALL be explicit.

Provider compilation SHALL NOT automatically inherit unrestricted:

- filesystem access
- network access
- environment access
- shell access
- arbitrary process execution
- secret access

## Compilation Isolation Model

Provider capability SHALL describe its compilation isolation model.

Suggested values:

```text
InProcessTrustedCompiler
RestrictedSubprocess
SandboxedSubprocess
ExternalCompilationService
PlatformManagedCompiler
BrowserManagedCompiler
Unavailable
```

These categories describe trust boundaries, not implementation requirements.

Runtime policy MAY reject a compilation path whose isolation model is
insufficient for the source artifact trust level.

## Untrusted Kernel Source

Kernel Source Artifact SHALL be treated as potentially untrusted executable
input.

A trusted compiler does not make untrusted source safe automatically.

Compilation of untrusted source SHOULD happen through a controlled compilation
boundary.

## Network Authority

Kernel compilation SHALL NOT require arbitrary network access during normal
Runtime model loading unless explicitly declared and policy-authorized.

Compiler dependency fetching SHALL NOT happen implicitly.

Toolchains, headers, libraries, and dependencies SHOULD already be available
through controlled deployment mechanisms.

## Filesystem Authority

Runtime SHOULD pass source bytes and metadata rather than arbitrary source file
paths.

A Provider MAY use a private temporary workspace internally.

Temporary paths SHALL remain Provider-private and redacted.

## Process Execution

A Provider MAY invoke compiler processes if declared by its isolation model.

It SHALL NOT construct shell command strings from untrusted source metadata.

Arguments SHALL be passed structurally where possible.

## Environment Variables

Compiler behavior SHALL NOT depend silently on ambient environment variables.

Environment inputs affecting compilation SHOULD be:

- denied
- sanitized
- allowlisted
- or captured in the compilation fingerprint

according to Provider policy.

Secrets SHALL NOT become compiler inputs unless an explicit non-inference
management policy authorizes them.

## Compilation Result

Successful compilation SHALL produce a validated candidate
CompiledKernelArtifact.

The result SHALL include:

- output bytes or Provider-managed artifact content reference
- compiled format
- digest
- source artifact digest
- target metadata
- compiler identity
- compiler options fingerprint
- specialization metadata
- compatibility metadata
- compilation observations
- trust/integrity state

## Compilation Does Not Grant Trust

Successful compilation SHALL NOT automatically mark the resulting artifact as
trusted or qualified.

```text
compiled != trusted
compiled != semantically correct
compiled != qualified
compiled != production eligible
```

Qualification is defined by a later change.

## Output Integrity

Runtime or Provider SHALL calculate and verify a digest for compiled output.

A compiled artifact SHALL have immutable identity before it is admitted for
preparation or caching.

## Compilation Failure Atomicity

Compilation failure SHALL NOT mutate an existing known-good Compiled Kernel
Artifact or Prepared Kernel.

Partial output SHALL remain non-ready.

## Preparation

Once a compatible Compiled Kernel Artifact exists, Provider MAY prepare it.

Preparation MAY involve:

```text
module loading
driver linking
pipeline creation
runtime specialization
graph compilation
native handle creation
```

Preparation returns an opaque PreparedKernelId.

## Preparation Capability

A Provider that can execute artifact-backed Kernels SHALL expose preparation
behavior independently from source compilation.

This allows:

```text
Provider A:
    source compilation + preparation

Provider B:
    preparation only from AOT artifacts

Provider C:
    static built-in kernels only
```

All three remain valid Providers.

## AOT-Only Platforms

A Provider that cannot compile source at Runtime MAY still consume
CompiledKernelArtifacts produced elsewhere.

Example:

```text
CI / build farm
    -> compiled artifact
    -> mobile deployment
    -> Provider.prepare()
```

Therefore Runtime compilation support SHALL NOT be required for Provider
execution support.

## Platform-Managed Shader Compilation

Platforms that compile shader source while creating a pipeline MAY logically
combine compile and prepare internally.

The Provider SHALL still expose enough metadata to preserve Magnetar's logical
artifact lifecycle and cold/hot path separation.

## Hot Path Prohibition

Provider execute APIs SHALL NOT trigger source compilation.

Provider execution SHALL accept only a previously prepared logical Kernel.

If Provider discovers that native compilation is unexpectedly required during
execute, it SHALL return a structured error rather than silently blocking the
decode loop.

## Model Instance Readiness

Runtime MAY require successful compilation and preparation of all mandatory
Kernel implementations before marking a Model Instance ready.

Compilation jobs MAY run concurrently during Model Loading.

## Compilation Concurrency

Runtime MAY submit multiple independent compilation jobs concurrently.

Provider SHALL advertise a concurrency limit or unbounded/unknown status.

Runtime SHALL respect Provider limits and global resource policy.

## Compiler Failure Containment

A compiler crash SHALL be normalized into a Provider compilation failure.

It SHALL NOT be reported as a successful artifact.

Where the compilation isolation model permits, a compiler crash SHOULD NOT
crash the Magnetar Runtime process.

## ABI Boundary

The stable native Provider ABI SHALL NOT expose Rust trait objects for kernel
compilation.

Compilation capability SHALL use versioned C-compatible or otherwise
explicitly stable ABI descriptors.

Conceptual ABI:

```c
struct MagnetarKernelCompilationCapabilityV1;

query_kernel_compilation_capability(...);

submit_kernel_compilation(...);

poll_kernel_compilation(...);

cancel_kernel_compilation(...);

release_kernel_compilation_job(...);
```

Exact C declarations are implementation-defined by this change's implementation,
but ownership and versioning semantics SHALL follow the existing Provider ABI
policy.

## Optional ABI Extension

Kernel Compilation Capability SHOULD be an optional ABI extension.

A Provider implementing Provider ABI v1 but not Kernel Compilation SHALL remain
valid.

Runtime SHALL discover the extension explicitly.

Absence SHALL produce:

```text
kernel-compilation-unavailable
```

when compilation is required.

It SHALL NOT be treated as Provider corruption.

## ABI Ownership

All buffers crossing the ABI SHALL have explicit ownership.

The ABI SHALL define:

- who allocates request buffers
- how Provider reads them
- who allocates result buffers
- how Runtime copies/consumes them
- how Provider releases result storage
- how error strings are released

No allocator ownership SHALL be implicit.

## No Unwinding Across ABI

Provider compiler implementation SHALL NOT unwind across the Provider ABI.

Panics, exceptions, or compiler failures SHALL be normalized into structured
failure status.

## Prepared Kernel ABI

PreparedKernelId MAY cross the ABI as an opaque numeric identifier.

Its value SHALL NOT be a public native pointer.

Example:

```c
typedef uint64_t MagnetarPreparedKernelId;
```

Provider alone maps the ID to native executable state.

## Observability

Compilation observability MAY include:

- request submitted
- queued
- compiler started
- compiler completed
- compiler failed
- compiler cancelled
- compiler timed out
- compiled artifact created
- output validated
- preparation started
- preparation completed

Observability MAY report:

- source digest
- source format
- compiled format
- compiler identity
- target architecture
- elapsed duration
- output size
- specialization summary

Observability SHALL NOT expose by default:

- raw kernel source
- compiled binary bytes
- native handles
- compiler temporary paths
- arbitrary compiler stdout/stderr
- environment contents
- secrets
- credentials

## Compiler Diagnostics

Provider MAY return compiler diagnostics.

Diagnostics SHALL be classified and redacted.

Runtime SHOULD prefer:

```text
error category
compiler stage
source-location metadata
redacted diagnostic
```

rather than unrestricted raw compiler output.

## Error Model

Structured errors SHOULD include:

```text
kernel-compilation-unavailable
kernel-compilation-capability-version-unsupported
kernel-compilation-source-format-unsupported
kernel-compilation-output-format-unsupported
kernel-compilation-target-unsupported
kernel-compilation-specialization-unsupported
kernel-compilation-policy-denied
kernel-compilation-isolation-insufficient
kernel-compilation-source-too-large
kernel-compilation-output-too-large
kernel-compilation-concurrency-limit
kernel-compilation-deadline-unsupported
kernel-compilation-timeout
kernel-compilation-cancellation-unsupported
kernel-compilation-cancelled
kernel-compilation-compiler-unavailable
kernel-compilation-compiler-crashed
kernel-compilation-failed
kernel-compilation-output-invalid
kernel-compilation-output-integrity-failed
kernel-compilation-job-not-found
kernel-compilation-job-state-invalid
kernel-compilation-abi-incompatible
kernel-compilation-buffer-ownership-violation
kernel-compilation-hot-path-denied
internal-kernel-compilation-error
```

## Conformance

Provider Kernel Compilation conformance SHALL validate:

- compilation capability discovery
- absence of compilation capability is valid
- accepted source format negotiation
- output format declaration
- unsupported formats rejected
- Device does not compile
- Scheduler does not compile
- Provider does not choose arbitrary Device
- compilation jobs have explicit lifecycle
- async completion behavior
- cancellation behavior
- deadline behavior
- size/resource limits
- untrusted source does not imply trusted output
- source compilation never occurs in normal execute path
- ABI buffers have explicit ownership
- no Rust ABI leakage
- no native pointer leakage
- compiler errors are structured and redacted
- failed compilation leaves no ready artifact
- preparation returns opaque PreparedKernelId

## Non-Goals

This change does not:

- implement Triton
- select Triton as Magnetar's canonical language
- implement CUDA/NVCC
- implement HIP
- implement WebGPU
- implement Metal
- implement a compiler sandbox
- implement an AI kernel generator
- define Kernel qualification
- define benchmarking/ranking
- define content-addressed Kernel cache
- define hot-swap policy
- unload/reload Providers
- allow compilation in token decode
- change Device into an execution/compiler API

## Impact

Magnetar gains a hardware-neutral contract for Provider-managed kernel
compilation while preserving the existing architecture:

```text
Generator
    ->
KernelSourceArtifact
    ->
Provider Compilation Capability
    ->
CompiledKernelArtifact
    ->
Provider Preparation
    ->
PreparedKernel
    ->
Kernel Registry / Dispatch
```

The Runtime can therefore consume future generated kernels without depending
on any specific generator or compiler technology.