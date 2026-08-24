# Define Model Loading Contract

## Why

Magnetar now distinguishes Model Artifacts from Component Artifacts, Providers,
Devices, Sessions, Generation, and KV cache.

A Model Artifact is validated model data.

It is not automatically executable.

Before inference can run, model data must be loaded, materialized, placed,
validated against Provider capabilities, and represented as Runtime-owned
resident state.

Without a Model Loading Contract, loading logic may become hidden inside:

- Generation
- Provider execution
- Memory Manager internals
- Session creation
- model architecture code
- Scheduler
- Component implementation

That would make model residency, memory pressure, dtype conversion,
quantization, sharding, Provider compatibility, Device placement, unload, reload,
and cache invalidation unsafe.

This change defines the Model Loading Contract.

## What Changes

This change introduces Model Loading as a first-class Runtime process.

Model Loading SHALL transform a validated and trusted Model Artifact into a
Runtime-owned loaded model context.

A loaded model context may later be formalized as a Model Instance.

This change defines the loading contract and residency semantics.

It SHALL cover:

- loading request
- loading policy
- artifact validation preconditions
- architecture implementation compatibility
- Memory Manager feasibility
- storage dtype and compute dtype handling
- quantization compatibility
- sharded loading
- Provider Capability validation
- Provider and Device placement
- model residency plan
- loading lifecycle
- unload behavior
- reload behavior
- errors
- observability

## Model Loading Is Runtime-Owned

Model loading SHALL be owned by the Runtime.

Clients may request loading.

Components may participate only through authorized inference-scoped contracts.

Providers may materialize native resources.

Memory Manager owns memory feasibility and residency.

The Runtime coordinates all of them.

## Model Artifact Preconditions

A Model Artifact SHALL be validated before loading.

Loading SHALL require:

- valid manifest
- verified digest
- required parts present
- required shards validated
- architecture metadata valid
- tokenizer association validated where required
- dtype metadata valid
- quantization metadata valid
- trust policy accepted
- license policy accepted where enforced
- revocation checks passed

A loading request SHALL fail before memory allocation if artifact preconditions
are not met.

## Model Loading Request

A ModelLoadingRequest SHALL describe a request to load model data.

It SHOULD include:

- request ID
- model artifact reference
- target usage
- requested compute dtype
- requested storage handling
- quantization policy
- sharding policy
- residency policy
- memory budget
- placement preference or constraints
- required Capabilities
- optional session association
- optional cache policy
- priority
- timeout
- observability correlation ID

Provider and Device selection SHALL not be directly controlled by user input.

Any placement preference SHALL be interpreted through Runtime policy,
Resource Affinity, and Resolution Policy.

## Loading Lifecycle

A loaded model context SHALL have lifecycle state.

Initial states SHOULD include:

```text
requested
validating
planning
allocating
materializing
ready
active
draining
unloading
unloaded
failed
invalid
```

Semantics:

- `requested`: loading was requested
- `validating`: artifact and compatibility checks are running
- `planning`: memory and placement plans are being computed
- `allocating`: memory is being allocated or queued
- `materializing`: weights/config are being materialized
- `ready`: model context may be used for inference
- `active`: model context is currently used
- `draining`: no new use; existing use may finish
- `unloading`: resources are being released
- `unloaded`: resources released
- `failed`: loading or use failed
- `invalid`: context is no longer safe to use

## Architecture Implementation

A Model Artifact declares architecture metadata.

Loading SHALL validate that a compatible model architecture implementation
exists.

The implementation MAY be:

- Runtime-native
- Component-based
- Provider-assisted
- test fixture

Architecture implementation SHALL NOT be confused with Provider.

Invalid reasoning:

```text
llama -> LlamaProvider
qwen  -> QwenProvider
```

Correct reasoning:

```text
qwen artifact
    + qwen architecture implementation
    + Runtime Resolution
    + CPU/CUDA/Metal/OpenVINO/QNN/Candle Provider
    + Device placement
```

## Provider Compatibility

Loading SHALL validate Provider Capability compatibility.

Provider compatibility MAY include:

- supported Compute Capability version
- supported operation families
- supported dtypes
- supported layouts
- supported quantization formats
- supported memory placements
- supported data movement
- supported model materialization mode
- supported KV cache layout where relevant later
- Device availability
- Provider readiness
- Provider pressure
- Provider loading policy

Provider compatibility SHALL be evaluated through Runtime Resolution and
Provider advertisements.

A Model Artifact SHALL not directly select a Provider.

## Device Placement

Loading MAY produce Device-bound residency.

Device placement SHALL be Runtime-owned.

Placement SHALL consider:

- Resource Affinity
- memory capacity
- memory pressure
- required dtype
- quantization support
- data movement support
- sharding
- Provider readiness
- Device readiness
- policy

A Model Artifact SHALL not directly select a Device.

## Memory Manager Relationship

Model Loading SHALL use the Memory Manager.

Memory Manager SHALL evaluate:

- model weights memory
- config memory
- quantized storage memory
- compute-ready materialization memory
- temporary dequantization workspace
- sharded loading memory
- adapter overlay memory placeholder
- transfer staging memory
- pinned memory where applicable
- browser memory constraints
- loading pending queues
- memory pressure
- residency plan

Generation SHALL not allocate model residency directly.

Provider code SHALL not bypass Memory Manager accounting.

## Model Residency Plan

Loading SHALL produce a Model Residency Plan before materialization.

A Model Residency Plan SHOULD describe:

- artifact reference
- model architecture
- target compute dtype
- storage dtype
- quantization handling
- shard placement
- memory placements
- Provider/Device bindings where resolved
- required data movement
- temporary workspace
- expected resident size
- loading phases
- fallback options
- unload policy
- diagnostics

A plan SHALL not expose raw native handles.

## Model Residency

Model residency is Runtime-owned state indicating where loaded model data lives.

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

## Loading Phases

Loading MAY occur in phases.

Phases MAY include:

```text
read-manifest
validate-parts
open-artifact-bytes
validate-shards
plan-memory
allocate-host
allocate-device
materialize-weights
dequantize-or-transform
transfer-to-device
initialize-provider-state
validate-ready
publish-model-context
```

Implementations may combine phases, but diagnostics SHOULD preserve meaningful
phase information.

## DType Handling

Loading SHALL distinguish:

```text
storage_dtype
compute_dtype
```

Loading SHALL validate:

- storage dtype supported
- requested compute dtype supported
- conversion allowed
- temporary workspace available
- Provider can execute requested compute dtype
- quantization compatibility

If conversion is required, it SHALL be explicit in the residency plan.

## Quantization Handling

Quantized model loading SHALL validate quantization metadata.

Loading SHALL determine whether quantized weights are:

- used directly by Provider
- dequantized during load
- dequantized lazily
- transformed into Provider-specific layout
- rejected as unsupported

The chosen behavior SHALL be explicit in the Model Residency Plan.

## Sharded Loading

Model Loading SHALL support sharded artifacts.

Loading SHALL validate all required shards.

Loading MAY materialize shards independently.

Sharding policy MAY determine whether shards are:

- loaded sequentially
- loaded in parallel
- placed on one Device
- split across Devices
- kept on host and transferred lazily
- rejected due to unsupported placement

## Lazy Loading

Runtime MAY support lazy loading.

Lazy loading SHALL be explicit policy.

If enabled, not all weights are materialized during initial load.

Lazy loading SHALL still validate artifact identity, required parts, trust, and
compatibility before the model context is considered loadable.

Lazy loading SHALL report pending residency state.

## Partial Loading

Partial loading MAY be allowed only by explicit policy.

Partial loading SHALL not produce a ready model context unless the missing parts
are not required for the requested usage.

A partially loaded context SHALL expose clear status.

## Loading And Sessions

A session may reference a loaded model context.

Session creation MAY require a model to be already loaded, or may request
implicit loading if Runtime policy allows it.

Implicit loading SHALL still follow the same Model Loading Contract.

Closing a session SHALL not necessarily unload a model.

Model unload is governed by model residency policy.

## Loading And KV Cache

Loading a model SHALL not create KV cache.

KV cache is created by generation prefill/decode.

Model unload may invalidate KV caches associated with that model context.

## Loading And Adapters

Loading SHALL prepare for adapter overlays.

Adapters may be separate Model Artifacts.

This change does not define full adapter merge or activation behavior.

It establishes that adapter residency and compatibility belong to model loading
or future adapter loading contracts, not Provider naming.

## Loading And Browser Target

Model Loading SHALL be platform-neutral.

Browser targets may support a reduced loading model due to:

- WebAssembly linear memory
- browser memory limits
- WebGPU buffer constraints
- lack of native Provider loading
- no native pinned memory
- no mmap

Unsupported loading features SHALL return structured errors.

Browser loading SHALL not require Wasmtime.

## Unload

Runtime SHALL support model unloading.

Unload SHALL:

- prevent new inference use
- drain or reject active sessions according to policy
- invalidate or release associated KV caches according to policy
- release Memory Manager resources
- release Provider-owned resources
- update residency state
- emit observability

Unload SHALL not leave dangling session references.

## Reload

Runtime MAY support reload.

Reload MAY be used for:

- policy changes
- Provider changes
- Device recovery
- dtype change
- quantization mode change
- artifact update
- memory pressure recovery

Reload SHALL create a new validated loading process.

Reload SHALL not silently mutate existing model contexts unless policy permits.

## Failure Handling

Loading failures SHALL be structured and phase-aware.

If loading fails after partial allocation, Runtime SHALL clean up or mark
resources according to policy.

A failed loaded model context SHALL not be used for inference.

## Trust And Security

Model Loading SHALL not bypass Model Artifact trust.

Provider materialization SHALL not make untrusted model bytes trusted.

Runtime SHALL prevent raw loaded weights or Provider memory handles from being
exposed to Components or clients by default.

## Error Model

Model loading errors SHALL be structured.

Error categories SHOULD include:

- model artifact not found
- model artifact invalid
- model artifact untrusted
- model artifact revoked
- architecture unsupported
- architecture implementation missing
- tokenizer incompatible
- required part missing
- shard missing
- shard digest mismatch
- storage dtype unsupported
- compute dtype unsupported
- dtype conversion unsupported
- quantization unsupported
- quantization transform failed
- memory feasibility failed
- memory allocation failed
- loading queued
- loading timeout
- Provider capability unavailable
- Provider not ready
- Provider saturated
- Device unavailable
- Device memory insufficient
- placement unsupported
- data movement unsupported
- materialization failed
- Provider initialization failed
- unload failed
- reload failed
- browser feature unsupported
- internal loading error

## Observability

Runtime SHOULD emit observations for:

- model loading requested
- artifact preconditions checked
- loading validation failed
- residency planning started
- residency planning completed
- memory allocation requested
- memory allocation queued
- memory allocation failed
- shard loading started
- shard loading completed
- materialization started
- materialization completed
- Provider state initialized
- model ready
- model load failed
- model unloading started
- model unloaded
- model reload requested
- model reload completed
- model residency pressure

Observability SHALL not expose raw model weights, secrets, raw memory handles,
or private Provider handles.

## Non-Goals

This change does not:

- define full Model Instance public lifecycle
- define sampling
- define logits processing
- define adapter activation fully
- define LoRA merge behavior
- define KV cache layout
- define continuous batching
- define model registry download protocol
- define Hugging Face integration
- define Tachyon model distribution
- define remote model loading
- define cross-node sharding
- define out-of-process Provider memory
- require GPU hardware
- require browser model loading implementation
- expose raw model weights
- allow Model Artifacts to select Providers directly
- allow Model Artifacts to select Devices directly

## Impact

Magnetar gains a clear transition from validated model data to loaded inference
state.

The model inference path becomes:

```text
Model Artifact
    |
    v
Model Loading Contract
    |
    +-- artifact preconditions
    +-- architecture implementation
    +-- Memory Manager feasibility
    +-- Provider/Device compatibility
    +-- residency plan
    +-- materialization
    |
    v
Loaded Model Context / future Model Instance
    |
    v
Inference Session
    |
    v
Generation
```

This prepares:

- sampling and logits processing contract
- full Model Instance lifecycle
- adapter loading contract
- prefix cache model
- continuous batching contract