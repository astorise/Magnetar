# Define Adapter Loading Contract

## Why

Magnetar now has contracts for:

- Model Artifacts
- Model Loading
- Model Residency
- Tokenizer
- Generation
- Sampling
- Inference Sessions
- KV Cache
- Prefix Cache
- Continuous Batching
- Memory Manager
- Providers and Devices

The next missing foundation is adapter loading.

Adapters such as LoRA or related low-rank overlays are common inference-time
extensions to a base model.

Adapters must not be hidden inside:

- Model Loading
- Generation
- Provider execution
- Kernel dispatch
- Session state
- Memory Manager internals
- Model Artifact validation

Without a first-class Adapter Loading Contract, Magnetar cannot safely define:

- adapter artifact identity
- base model compatibility
- architecture compatibility
- tokenizer compatibility where relevant
- dtype compatibility
- rank metadata
- target module metadata
- adapter residency
- activation/deactivation
- merge versus overlay behavior
- memory pressure
- KV cache invalidation
- session policy
- batching compatibility
- kernel requirements

This change defines the Adapter Loading Contract.

## What Changes

This change introduces Adapter Artifacts and Adapter Loading as first-class
inference concepts.

An Adapter Artifact SHALL represent adapter data that modifies or augments a
base model during inference.

Adapter Loading SHALL transform a validated Adapter Artifact into
Runtime-managed adapter residency.

Adapter Activation SHALL associate loaded adapter residency with a generation
context, session, or model instance according to policy.

The exact Rust type names are implementation-defined.

## Adapter Artifact

An Adapter Artifact SHALL be a Model Artifact kind or related inference artifact
kind.

Adapter Artifact metadata SHOULD include:

- adapter identity
- base model compatibility
- model architecture compatibility
- target module metadata
- adapter method
- rank metadata
- alpha/scaling metadata
- dtype metadata
- tensor metadata
- quantization metadata where applicable
- shard metadata where applicable
- tokenizer compatibility where applicable
- required Runtime features
- required Provider Capabilities
- license metadata
- provenance metadata
- signature metadata

Adapter Artifact identity SHALL be digest-based.

Logical adapter names or paths SHALL not be sufficient identity.

## Adapter Method

Adapter method SHALL be explicit.

Initial methods SHOULD include:

```text
lora
qlora
ia3
prompt-tuning
prefix-tuning
custom
```

This change does not require implementation of every method.

Unsupported adapter methods SHALL produce structured errors.

## Adapter Is Not Provider

An adapter SHALL not define Provider identity.

Invalid reasoning:

```text
customer-support-lora -> CustomerSupportProvider
qwen-lora              -> QwenLoraProvider
```

Correct reasoning:

```text
base model + adapter artifact
        |
        v
Runtime adapter loading
        |
        v
Provider executes compatible operators/kernels
```

Providers remain CPU, CUDA, Metal, OpenVINO, QNN, Candle temporary, or other
native execution implementations.

## Adapter Is Not Kernel

An adapter may affect which operators or kernels are needed.

But the adapter itself is not a kernel.

For example, a LoRA adapter may require the execution graph to include low-rank
projection operations or fused adapter kernels.

Kernel contracts are defined separately.

Adapter metadata SHALL inform graph/operator/kernel planning.

## Base Model Compatibility

Adapter Loading SHALL validate compatibility with the base model.

Compatibility SHOULD include:

- base model identity
- base model revision
- architecture family
- architecture implementation
- hidden size
- layer count
- target module names
- tensor shapes
- tokenizer compatibility where relevant
- dtype compatibility
- quantization compatibility
- position encoding compatibility where relevant
- generation mode compatibility
- Provider Capability compatibility

An incompatible adapter SHALL not be activated.

## Target Module Metadata

Adapters SHOULD declare target modules.

Examples:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
lm_head
embedding
```

Target module names are architecture-specific.

Runtime SHALL validate target module metadata against the loaded model
architecture implementation.

Missing or incompatible targets SHALL produce structured errors.

## Adapter Loading Request

An AdapterLoadingRequest SHOULD include:

- request ID
- adapter artifact reference
- base model context reference
- target usage
- adapter method
- requested compute dtype
- residency policy
- activation policy
- merge policy
- memory budget
- required Capabilities
- optional session association
- priority
- timeout
- observability correlation ID

Provider and Device selection SHALL remain Runtime-owned.

## Adapter Loading Lifecycle

A loaded adapter SHALL have lifecycle state.

Initial states SHOULD include:

```text
requested
validating
planning
allocating
materializing
ready
active
inactive
merging
merged
unmerging
draining
unloading
unloaded
failed
invalid
```

Semantics:

- `requested`: loading was requested
- `validating`: artifact and compatibility validation are running
- `planning`: memory and execution impact are being planned
- `allocating`: memory is being allocated or queued
- `materializing`: adapter tensors are being materialized
- `ready`: adapter can be activated
- `active`: adapter is currently applied to inference
- `inactive`: adapter is loaded but not active
- `merging`: adapter is being merged into base weights
- `merged`: adapter has been merged according to policy
- `unmerging`: merged adapter state is being reverted where supported
- `draining`: adapter refuses new use while active use completes
- `unloading`: adapter resources are being released
- `unloaded`: resources released
- `failed`: loading or use failed
- `invalid`: adapter is no longer safe to use

## Adapter Residency

Adapter residency is Runtime-owned state indicating where adapter data lives.

Residency MAY include:

- host memory
- pinned host memory
- device memory
- unified/shared memory
- provider-owned opaque memory
- browser linear memory
- future WebGPU buffer
- sharded residency
- mixed residency

Residency SHALL be tracked by Memory Manager.

Residency SHALL imply Resource Affinity where applicable.

## Memory Manager Relationship

Adapter Loading SHALL use Memory Manager.

Memory Manager SHALL evaluate:

- adapter tensor memory
- quantized adapter storage
- compute-ready materialization memory
- temporary transform workspace
- merge workspace
- unmerge workspace where supported
- transfer staging
- pinned memory where applicable
- session adapter budget
- memory pressure
- residency placement
- pending allocation

Generation SHALL not allocate adapter memory directly.

Provider code SHALL not bypass Memory Manager accounting.

## Activation

Adapter Activation SHALL be explicit.

Activation MAY be scoped to:

```text
operation
session
model-instance
runtime
```

Default activation SHOULD be scoped narrowly.

Activation SHALL validate:

- loaded adapter lifecycle
- base model compatibility
- session policy
- memory residency
- Resource Affinity
- Provider/Device compatibility
- batching compatibility
- KV cache compatibility
- Prefix Cache compatibility

## Multiple Adapters

Runtime MAY support multiple active adapters.

Multiple adapter behavior SHALL be explicit.

Policies MAY include:

```text
single-adapter-only
multiple-adapters-ordered
weighted-adapter-composition
reject-multiple-adapters
```

If multiple adapters are active, their order and composition SHALL be
deterministic and policy-controlled.

## Merge Versus Overlay

Adapter execution may use different strategies.

Initial strategies SHOULD include:

```text
overlay
merge-on-load
merge-on-activation
provider-fused
disabled
```

Semantics:

- `overlay`: adapter remains separate and is applied during execution
- `merge-on-load`: adapter is merged into model residency during loading
- `merge-on-activation`: adapter is merged when activated
- `provider-fused`: Provider executes fused adapter path without permanent merge
- `disabled`: adapter is loaded but not used

Merge behavior SHALL be explicit.

Silent mutation of base model residency SHALL be forbidden unless policy
explicitly allows it.

## Base Model Mutation

Adapter activation SHALL NOT silently mutate base model residency.

If merge is used, Runtime SHALL track:

- merge source adapter
- affected base residency
- reversible status
- new residency state
- invalidated KV caches
- invalidated Prefix Cache entries
- unload/unmerge policy

If merge cannot be safely reversed, Runtime SHALL record that state.

## KV Cache Relationship

Activating or changing adapters may invalidate KV cache.

KV cache compatibility SHALL include active adapter set.

A KV cache created with one adapter set SHALL not be reused with another
incompatible adapter set.

Runtime policy decides whether to:

- reject cache reuse
- rebuild cache
- invalidate cache
- maintain adapter-specific cache entries
- use adapter-aware prefix cache keys

## Prefix Cache Relationship

Prefix Cache fingerprints SHALL include active adapter set where adapter changes
affect model outputs.

A Prefix Cache entry created with adapter A SHALL not be reused without adapter
A unless policy and compatibility prove it is safe.

Adapter changes may invalidate Prefix Cache entries.

## Generation Relationship

Generation SHALL use active adapter context when executing model forward.

Generation requests may reference adapter activation according to Runtime and
session policy.

Generation SHALL not load adapters implicitly unless policy explicitly allows
it.

Generation SHALL not silently activate an adapter.

## Sampling Relationship

Sampling usually remains independent from adapters, but adapter changes may
affect logits and therefore deterministic outputs.

Determinism metadata SHOULD include adapter set identity when applicable.

## Continuous Batching Relationship

Batched operations may share a batch step only when adapter compatibility allows
it.

Compatibility SHOULD consider:

- active adapter set
- adapter execution strategy
- Provider fused adapter support
- memory residency
- Resource Affinity
- merge state
- sampling compatibility where relevant

## Provider Relationship

Providers may support adapter-specific execution.

Provider advertisements MAY include:

- supported adapter methods
- maximum adapter rank
- supported adapter dtypes
- supported merge strategies
- supported fused adapter kernels
- supported target modules
- supported quantized adapter formats
- adapter memory requirements
- adapter activation cost

Provider support SHALL be validated through Runtime policy.

Provider-owned adapter resources SHALL remain opaque.

## Device Relationship

Adapter residency may be Device-bound.

Device-bound adapter residency SHALL constrain future execution.

Runtime SHALL not silently move adapter data between Devices.

Movement, copy, transform, or re-materialization SHALL be explicit.

## Session Relationship

An Inference Session may reference loaded adapters and active adapter policy.

Session policy may define:

- allowed adapters
- maximum active adapters
- default adapter
- activation allowed
- deactivation allowed
- merge allowed
- adapter memory budget
- adapter sharing policy
- adapter privacy policy
- adapter unload on session close

A session SHALL not gain arbitrary filesystem access to adapters.

## Adapter Privacy

Adapters may encode private or fine-tuned behavior.

Adapter metadata, names, and activation status may be sensitive.

Runtime SHALL not expose raw adapter tensors by default.

Observability SHALL redact adapter metadata according to policy.

## Browser Target

Adapter Loading SHALL be platform-neutral.

Browser targets may support a reduced adapter model depending on:

- browser memory limits
- WebAssembly linear memory
- future WebGPU buffers
- available Provider capabilities
- session policy

Unsupported adapter features SHALL return structured errors.

Browser adapter loading SHALL not require Wasmtime or native Provider loading.

## Error Model

Adapter errors SHALL be structured.

Error categories SHOULD include:

- adapter artifact not found
- adapter artifact invalid
- adapter artifact untrusted
- adapter artifact revoked
- adapter method unsupported
- base model incompatible
- architecture incompatible
- target module missing
- target tensor mismatch
- tokenizer incompatible
- storage dtype unsupported
- compute dtype unsupported
- quantization unsupported
- adapter rank unsupported
- adapter shape mismatch
- memory feasibility failed
- memory allocation failed
- adapter loading queued
- adapter loading timeout
- Provider capability unavailable
- Provider adapter unsupported
- Provider not ready
- Provider saturated
- Device unavailable
- Device memory insufficient
- activation denied
- activation conflict
- multiple adapters unsupported
- merge unsupported
- merge failed
- unmerge unsupported
- unmerge failed
- KV cache incompatible
- Prefix Cache invalidated
- unload failed
- browser feature unsupported
- internal adapter error

## Observability

Runtime SHOULD emit observations for:

- adapter loading requested
- adapter artifact validated
- adapter compatibility checked
- adapter loading validation failed
- adapter residency planning started
- adapter residency planning completed
- adapter memory allocation requested
- adapter memory allocation queued
- adapter materialization started
- adapter materialization completed
- adapter ready
- adapter activated
- adapter deactivated
- adapter merge started
- adapter merge completed
- adapter unmerge started
- adapter unmerge completed
- adapter load failed
- adapter unload started
- adapter unloaded
- adapter cache invalidation
- adapter batching compatibility failed

Observability SHALL not expose raw adapter tensors, raw model weights, raw
Provider handles, or raw prompts by default.

## Non-Goals

This change does not:

- define kernel implementation
- define full operator graph contract
- define LoRA math in detail
- define all adapter methods
- require multiple adapter support
- require merge support
- require Provider-fused adapter support
- define training or fine-tuning
- define adapter download protocol
- define Hugging Face integration
- define adapter marketplace
- expose raw adapter tensors
- expose Provider adapter handles
- allow adapters to select Providers directly
- allow adapters to select Devices directly
- require GPU hardware
- require browser adapter implementation

## Impact

Magnetar gains a clear adapter boundary.

The inference model becomes:

```text
Base Model Artifact
        |
        v
Model Loading
        |
        v
Base Model Instance
        |
        +-- Adapter Artifact
        |       |
        |       v
        |   Adapter Loading
        |       |
        |       v
        |   Adapter Residency
        |
        v
Generation with active adapter context
```

This prepares:

- Model Instance lifecycle
- Execution Graph and Operator contract
- Kernel contract
- Kernel registry and dispatch
- Model Component contract