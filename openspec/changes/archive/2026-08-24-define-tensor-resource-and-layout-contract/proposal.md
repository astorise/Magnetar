# Define Tensor Resource And Layout Contract

## Why

Magnetar now defines:

- Memory Manager
- Operator Contract
- Execution Graph
- Kernel Contract
- Kernel Registry and Dispatch
- Reference CPU Provider
- first operator implementation scope

The next required foundation is the Tensor Resource and Layout Contract.

Operators describe computation.

Kernels execute implementations.

Memory Manager owns allocation and residency.

But both Operators and Kernels need a precise, portable way to describe tensor
shape, dtype, layout, view, aliasing, memory class, and Resource Affinity.

Without this contract, tensor handling may become ambiguous or unsafe:

- Components may assume raw pointers
- Providers may assume hidden layouts
- Kernels may mutate tensors unexpectedly
- Memory Manager may not track views or aliases
- Graph planning may silently insert conversions
- CPU fallback may silently move data
- quantized or paged layouts may leak into portable APIs incorrectly

This change defines Tensor Resource and Layout as first-class Runtime concepts.

## What Changes

This change introduces the Tensor Resource and Layout Contract.

It defines:

- Tensor Descriptor
- Tensor Resource
- Tensor Resource ID
- Tensor View
- Tensor Shape
- Tensor Rank
- Tensor DType
- Storage DType versus Compute DType
- Tensor Layout
- Layout Descriptor
- Strides
- Contiguous layout
- Strided layout
- Blocked layout placeholder
- Paged layout placeholder
- Packed quantized layout placeholder
- Provider-owned opaque layout boundary
- Tensor memory class
- Tensor residency
- Resource Affinity
- aliasing rules
- mutation rules
- lifetime rules
- conversion rules
- observability rules
- conformance requirements

The exact Rust type names are implementation-defined.

## Tensor Descriptor

A Tensor Descriptor SHALL describe tensor metadata without implying that tensor
storage is allocated.

A Tensor Descriptor SHOULD include:

- shape
- rank
- dtype
- optional storage dtype
- optional compute dtype
- layout descriptor
- memory class intent
- mutability intent
- aliasing intent
- Resource Affinity constraints
- semantic role where relevant

A Tensor Descriptor SHALL NOT contain raw memory pointers.

## Tensor Resource

A Tensor Resource SHALL represent Runtime-managed tensor storage or
Provider-owned opaque storage with Runtime-visible metadata.

A Tensor Resource SHOULD include:

- Tensor Resource ID
- descriptor
- allocation or provider-owned resource reference
- residency metadata
- memory class
- Resource Affinity
- lifecycle state
- readiness state
- aliasing metadata
- view metadata where applicable
- owner subsystem metadata
- observability correlation

Tensor Resources SHALL be owned by Runtime and Memory Manager.

## Tensor Resource Identity

Tensor Resource identity SHALL be Runtime-issued and opaque.

Tensor Resource IDs SHALL NOT encode:

- raw pointers
- Provider handles
- Device handles
- allocation addresses
- file paths
- model secrets
- prompt data

A Tensor Resource ID alone SHALL not grant unauthorized access.

## Tensor Resource Lifecycle

Tensor Resource lifecycle states SHOULD include:

```text
declared
planned
allocating
ready
in-use
view
mutating
released
evicted
invalid
failed
```

Semantics:

- `declared`: metadata exists but no allocation is guaranteed
- `planned`: Runtime has planned allocation or binding
- `allocating`: Memory Manager is allocating
- `ready`: data is available according to metadata
- `in-use`: resource is currently used by an operation
- `view`: resource represents a view over another resource
- `mutating`: resource is being modified by a Runtime-authorized operation
- `released`: resource was released
- `evicted`: resource was evicted according to policy
- `invalid`: resource is no longer safe to use
- `failed`: creation or update failed

## Readiness

Tensor Resource readiness SHALL be distinct from lifecycle.

Readiness SHOULD include:

```text
not-ready
ready
pending-transfer
pending-conversion
pending-compute
invalid
failed
```

Kernels SHALL only dispatch on resources whose readiness is compatible with the
Kernel requirements.

## Tensor Shape

Tensor Shape SHALL be explicit.

Shape metadata SHOULD include:

- rank
- dimensions
- symbolic dimensions where allowed
- dynamic dimension markers where allowed
- maximum dimension constraints where relevant
- batch dimension role where relevant
- sequence dimension role where relevant
- hidden dimension role where relevant
- head dimension role where relevant

Shape validation SHALL happen before Kernel dispatch where possible.

## Tensor DType

Tensor DType SHALL be explicit.

DType metadata SHALL distinguish:

```text
storage dtype
compute dtype
accumulation dtype
output dtype
index dtype
mask dtype
```

No dtype conversion SHALL occur silently.

DType conversion SHALL be represented as explicit graph/operator planning or
explicit Runtime movement/conversion plan.

## Tensor Layout

Tensor Layout SHALL be explicit.

Initial layout categories SHOULD include:

```text
contiguous
strided
blocked
paged
packed-quantized
attention-specific
provider-owned-opaque
browser-compatible
```

A Tensor Layout SHALL describe how logical tensor dimensions map to storage.

Unsupported layouts SHALL fail with structured errors or explicit conversion
plans.

## Contiguous Layout

Contiguous layout SHALL describe dense row-major or explicitly declared dense
storage order.

The exact dimension order SHALL be explicit where needed.

Reference CPU Provider SHOULD support contiguous layout for the first operator
scope.

## Strided Layout

Strided layout SHALL describe dimension strides explicitly.

Strided layout MAY be placeholder in the first implementation scope.

If unsupported, Runtime SHALL reject or plan explicit contiguous materialization
where policy allows.

## Blocked Layout

Blocked layout SHALL be reserved for tiled or block-structured storage.

Blocked layout MAY be future or Provider-specific.

Blocked layout SHALL not be assumed by portable Components unless represented
through portable metadata.

## Paged Layout

Paged layout SHALL represent page/block-based tensor storage, especially for
future KV cache and attention paths.

Paged layout MAY be placeholder in the first implementation scope.

Paged layout metadata SHOULD include:

- page size
- block size
- logical-to-physical mapping metadata
- capacity
- current length
- append behavior where relevant

Raw page pointers SHALL not be exposed.

## Packed Quantized Layout

Packed quantized layout SHALL represent quantized packed storage.

Metadata SHOULD include:

- quantization method
- bits per value
- group size
- scale dtype
- zero point dtype
- packing order
- block/group metadata
- dequantization requirements

Packed quantized layout MAY be future or placeholder initially.

## Provider-Owned Opaque Layout

Provider-owned opaque layout SHALL represent storage that cannot be interpreted
portably.

Opaque layout metadata SHALL NOT expose raw Provider handles.

Components SHALL NOT receive opaque native layout internals.

Runtime may use opaque layout only through Provider-owned execution and
validated Resource Affinity.

## Tensor View

A Tensor View SHALL represent a Runtime-authorized view over a Tensor Resource.

View metadata SHOULD include:

- base Tensor Resource ID
- shape
- offset
- strides
- layout
- dtype compatibility
- mutability
- aliasing relationship
- lifetime dependency
- Resource Affinity inheritance

A Tensor View SHALL not outlive its base resource.

## Aliasing

Aliasing SHALL be explicit.

Aliasing metadata SHOULD distinguish:

```text
no-alias
read-only-alias
mutable-alias
input-output-alias
view-alias
internal-temporary-alias
```

Runtime SHALL validate aliasing before Kernel dispatch.

A Kernel SHALL not mutate an input unless mutation is declared and allowed.

## Mutability

Tensor mutability SHALL be explicit.

A Tensor Resource may be:

```text
immutable
mutable
single-writer
multi-reader
runtime-internal
provider-owned
```

Mutability policy SHALL be validated before scheduling and dispatch.

## Memory Class

Tensor Resources SHALL declare memory class.

Initial memory classes SHOULD include:

```text
host
pinned-host
device
unified
shared
provider-owned
browser-linear-memory
future-webgpu-buffer
```

A Kernel may require specific memory classes.

Memory Manager SHALL validate compatibility.

## Residency

Tensor Resource residency SHALL be tracked by Memory Manager.

Residency metadata SHOULD include:

- memory class
- Provider affinity
- Device affinity
- host visibility
- transfer state
- conversion state
- eviction eligibility
- size estimate
- ownership metadata

Residency SHALL drive Resource Affinity.

## Resource Affinity

Tensor Resource Affinity SHALL be Runtime-derived and authoritative.

A caller SHALL NOT forge Provider or Device affinity.

If a Tensor Resource is bound to a Provider or Device, Kernel selection SHALL
respect that affinity unless explicit movement or conversion is planned and
authorized.

## Tensor Size Accounting

Runtime SHALL compute or estimate tensor size from shape, dtype, layout, and
packing metadata.

Size accounting SHALL be used by Memory Manager.

Unknown size SHALL force conservative admission or rejection according to policy.

## Tensor Conversion

Tensor conversion SHALL be explicit.

Conversions MAY include:

- dtype conversion
- layout conversion
- memory class movement
- device transfer
- host staging
- materialization from provider-owned opaque layout
- dequantization
- quantization

Conversion SHALL be represented in graph planning or Runtime dispatch plan.

## Tensor Materialization

Tensor materialization SHALL be Runtime-controlled.

Materialization may occur from:

- Model Artifact weights
- Adapter Artifact tensors
- tokenizer or input tokens
- KV cache output
- operator output
- Provider-owned output
- test fixture data

Materialization SHALL be tracked by Memory Manager.

## Tensor Access Boundary

Components SHALL not access raw tensor storage unless a specific portable
Capability explicitly allows a safe representation.

Default Component access SHALL be descriptor-level, not pointer-level.

Provider execution may access native storage internally through Runtime-created
invocations.

## Runtime Tensor APIs

Runtime Tensor APIs SHALL expose stable metadata and controlled resource
references.

They SHALL not expose:

- raw pointers
- native handles
- allocation addresses
- Provider internals
- Device internals
- raw KV cache contents
- raw model weights
- raw prompts by default

## Execution Graph Relationship

Execution Graph tensor edges SHALL reference Tensor Descriptors or Tensor
Resources.

Graph validation SHALL verify tensor shape, dtype, layout, memory behavior,
aliasing, Resource Affinity, and lifecycle constraints.

## Operator Relationship

Operators SHALL consume and produce Tensor Descriptors or Tensor Resources
according to their contracts.

Operator validation SHALL check shape, dtype, layout, aliasing, and memory
behavior before Kernel selection.

## Kernel Relationship

Kernels SHALL receive Runtime-created resource references, not raw public
pointers.

Kernel compatibility SHALL validate tensor metadata before dispatch.

Kernel results SHALL update Tensor Resource readiness, residency, and aliasing
metadata where relevant.

## Provider Relationship

Providers may own opaque tensor storage.

Provider-owned tensor storage SHALL remain opaque to Components and clients.

Runtime SHALL track enough metadata to validate future operations.

Provider-owned tensor resources SHALL still participate in Memory Manager
accounting where possible.

## Reference CPU Relationship

Reference CPU Provider SHALL primarily support host contiguous Tensor Resources
for the first operator scope.

Unsupported layouts, dtypes, or memory classes SHALL fail explicitly or require
explicit conversion.

## Browser Target

Tensor Resource and Layout contracts SHALL be platform-neutral.

Browser targets may support:

- browser-linear-memory
- JavaScript-mediated buffers
- future WebGPU buffers
- reduced layout set
- reduced dtype set

Browser targets SHALL not require native Provider loading or Wasmtime.

Unsupported browser tensor features SHALL return structured errors.

## Error Model

Tensor errors SHALL be structured.

Error categories SHOULD include:

- tensor descriptor invalid
- tensor resource not found
- tensor resource not ready
- tensor resource invalid
- tensor resource released
- tensor shape invalid
- tensor shape mismatch
- tensor rank unsupported
- tensor dtype unsupported
- tensor dtype conversion required
- tensor dtype conversion unsupported
- tensor layout unsupported
- tensor layout conversion required
- tensor layout conversion unsupported
- tensor memory class unsupported
- tensor residency unavailable
- tensor Resource Affinity conflict
- tensor aliasing violation
- tensor mutability violation
- tensor view invalid
- tensor view base unavailable
- tensor size unknown
- tensor materialization failed
- tensor transfer failed
- tensor browser feature unsupported
- internal tensor error

## Observability

Runtime SHOULD emit observations for:

- tensor descriptor created
- tensor resource planned
- tensor resource allocated
- tensor resource ready
- tensor resource view created
- tensor resource used
- tensor resource mutated
- tensor conversion planned
- tensor conversion completed
- tensor conversion failed
- tensor transfer planned
- tensor transfer completed
- tensor transfer failed
- tensor released
- tensor evicted
- tensor invalidated
- tensor aliasing violation
- tensor Resource Affinity conflict

Observability SHALL not expose raw tensor values, raw prompts, raw model
weights, raw KV cache contents, raw Provider handles, raw Device handles, or
memory pointers by default.

## Non-Goals

This change does not:

- define every possible tensor layout
- implement optimized layout transforms
- implement full quantized layout support
- implement paged attention
- implement WebGPU tensors
- expose raw tensor pointers
- expose Provider-owned buffers to Components
- define a general ndarray library
- define training tensor gradients
- require GPU hardware
- require browser tensor implementation

## Impact

Magnetar gains a precise tensor boundary.

Execution becomes safer:

```text
Operator
    |
    v
Tensor Descriptor validation
    |
    v
Tensor Resource planning
    |
    v
Memory Manager allocation/residency
    |
    v
Kernel Dispatch with validated resources
```

This prepares:

- Qwen model component baseline
- Runtime inference API
- end-to-end local inference conformance
- future CUDA/Metal/OpenVINO/QNN layout support