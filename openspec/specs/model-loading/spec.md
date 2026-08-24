# model-loading Specification

## Purpose
TBD - created by archiving change define-model-loading-contract. Update Purpose after archive.
## Requirements
### Requirement: Model Loading Contract

Magnetar SHALL define Model Loading as the Runtime-owned process that
materializes validated Model Artifacts into loaded inference state.

#### Scenario: Load model

Given a Model Artifact is valid and trusted

When Runtime loads it

Then Runtime creates a loaded model context or returns a structured loading
error.

---

### Requirement: Loading Requires Validated Artifact

Model Loading SHALL require Model Artifact validation and trust before memory
allocation or materialization.

#### Scenario: Untrusted model

Given a Model Artifact is untrusted

When loading is requested

Then loading fails before allocation.

---

### Requirement: Loading Request

Model Loading SHALL accept a structured loading request.

The request SHOULD include artifact reference, target usage, compute dtype,
quantization policy, sharding policy, residency policy, memory budget, required
Capabilities, priority, timeout, and observability correlation.

#### Scenario: Missing artifact

Given a loading request has no artifact reference

When Runtime validates it

Then loading fails with model-artifact-not-found or invalid-request.

---

### Requirement: Loading Lifecycle

Loaded model context SHALL expose lifecycle state.

States SHOULD include requested, validating, planning, allocating,
materializing, ready, active, draining, unloading, unloaded, failed, and invalid.

#### Scenario: Model ready

Given materialization succeeds

When Runtime publishes the loaded context

Then lifecycle becomes ready.

---

### Requirement: Architecture Implementation Compatibility

Loading SHALL validate that a compatible model architecture implementation
exists.

#### Scenario: Missing architecture implementation

Given a Model Artifact declares architecture `custom-x`

And no compatible implementation exists

When loading is requested

Then loading fails with architecture-implementation-missing.

---

### Requirement: Architecture Is Not Provider

Model architecture SHALL not create Provider identity.

#### Scenario: Qwen model

Given a Model Artifact declares architecture `qwen`

When loading resolves execution compatibility

Then Runtime looks for a Qwen-compatible implementation and compatible Providers

And does not create `QwenProvider`.

---

### Requirement: Provider Compatibility Validation

Loading SHALL validate Provider Capability compatibility through Runtime
Resolution and Provider advertisements.

#### Scenario: Provider lacks dtype

Given requested compute dtype is BF16

And no compatible Provider supports BF16 for required operations

When loading is planned

Then loading fails or remains queued according to policy.

---

### Requirement: Device Placement Validation

Loading SHALL validate Device placement through Runtime policy, Resource
Affinity, Memory Manager, and Provider/Device status.

#### Scenario: Device memory insufficient

Given model residency requires more Device memory than available

When loading is planned

Then loading fails, queues, or falls back according to policy.

---

### Requirement: Model Residency Plan

Loading SHALL produce a Model Residency Plan before materialization.

The plan SHALL describe memory placement, dtype handling, quantization handling,
Provider/Device bindings where resolved, expected resident size, loading phases,
and diagnostics.

#### Scenario: Plan before allocation

Given loading request is valid

When Runtime prepares to allocate

Then a residency plan exists before materialization begins.

---

### Requirement: Model Residency

Runtime SHALL track Model Residency as loaded model data placement.

Residency SHALL be distinct from Model Artifact identity.

#### Scenario: Artifact versus residency

Given the same Model Artifact is loaded on two Devices

When Runtime records state

Then both contexts reference the same artifact identity but have distinct
residency.

---

### Requirement: Memory Manager Owns Loading Memory

Model Loading SHALL use Memory Manager for feasibility, allocation, residency,
pressure, staging, and release.

#### Scenario: Memory denied

Given Memory Manager denies allocation due to pressure

When loading runs

Then loading is queued, retried, or failed according to policy.

---

### Requirement: DType Handling

Loading SHALL distinguish storage dtype from compute dtype and validate any
conversion or workspace requirements.

#### Scenario: INT8 storage BF16 compute

Given weights are stored as INT8

And BF16 compute is requested

When loading is planned

Then Runtime validates conversion support and workspace availability.

---

### Requirement: Quantization Handling

Loading SHALL explicitly handle quantized artifacts.

#### Scenario: Unsupported quantization

Given a Model Artifact uses unsupported quantization format

When loading is requested

Then Runtime rejects loading with quantization-unsupported.

---

### Requirement: Sharded Loading

Loading SHALL validate and support sharded Model Artifacts.

#### Scenario: Missing shard

Given a Model Artifact requires shard 3

And shard 3 is unavailable

When loading is requested

Then loading fails with shard-missing.

---

### Requirement: Lazy Loading Policy

Lazy loading SHALL be available only when explicitly enabled by policy.

#### Scenario: Lazy loading enabled

Given lazy loading is enabled

When model loading completes initial validation

Then the loaded context may report pending residency for not-yet-materialized
parts.

---

### Requirement: Partial Loading Policy

Partial loading SHALL be explicit and SHALL NOT produce ready state if required
parts are missing.

#### Scenario: Required part missing

Given required weights are missing

When partial loading is not explicitly allowed for the target usage

Then loading fails.

---

### Requirement: Loading Does Not Create KV Cache

Model Loading SHALL NOT create KV cache.

KV cache is created by generation prefill or decode.

#### Scenario: Model loaded

Given model loading completes

When no generation has run

Then no KV cache is created.

---

### Requirement: Model Unload

Runtime SHALL support unloading loaded model contexts.

Unload SHALL release Memory Manager resources, Provider-owned resources, update
residency state, and invalidate dependent runtime state according to policy.

#### Scenario: Unload model

Given a loaded model context is ready

When unload is requested

Then Runtime drains or rejects active use and releases resources according to
policy.

---

### Requirement: Model Reload

When reload is supported, Runtime SHALL treat it as a new validated loading process.

Reload SHALL NOT silently mutate existing loaded contexts unless policy permits.

#### Scenario: Reload dtype

Given a model is loaded with FP16 compute

When reload requests BF16 compute

Then Runtime performs a new loading process or rejects the reload.

---

### Requirement: Loading Failure Cleanup

If loading fails after partial allocation or materialization, Runtime SHALL
clean up or mark resources according to policy.

#### Scenario: Materialization failure

Given memory was allocated

And materialization fails

When loading fails

Then Runtime releases or invalidates allocated resources.

---

### Requirement: Browser-Compatible Loading Contract

Model Loading SHALL be platform-neutral and SHALL not require Wasmtime or native
Provider loading.

#### Scenario: Browser target

Given Runtime runs on browser target

When a loading feature requires native Provider loading

Then Runtime returns browser-feature-unsupported or equivalent structured error.

---

### Requirement: Model Loading Errors

Model Loading failures SHALL use structured error categories.

#### Scenario: Provider not ready

Given no compatible ready Provider exists

When loading requires Provider materialization

Then Runtime returns provider-not-ready or provider-capability-unavailable
according to the failure.

---

### Requirement: Model Loading Observability

Runtime SHALL define structured observations for loading request, validation,
planning, allocation, shard loading, materialization, readiness, unload, reload,
and failure.

Observability SHALL not expose raw model weights or raw native memory handles.

#### Scenario: Model ready observation

Given model loading succeeds

When Runtime publishes the loaded context

Then it may emit a model-ready observation.

### Requirement: Model Loading Prepares Adapter Compatibility

Model Loading SHALL expose metadata needed for later adapter validation.

#### Scenario: Loaded model target modules

Given a model is loaded

When an adapter targets its modules

Then Runtime can validate target modules against loaded model metadata.

---

### Requirement: Model Loading Does Not Implicitly Activate Adapter

Loading a base model SHALL NOT implicitly activate adapters.

#### Scenario: Base model ready

Given a base model is loaded

When no adapter activation request exists

Then the loaded model runs without adapters.

### Requirement: Model Loading Produces Model Instance

Successful Model Loading SHALL produce or update a Runtime-owned Model Instance
when target usage requires inference execution.

#### Scenario: Loading complete

Given model materialization succeeds

When Runtime publishes the result

Then a Model Instance is created or updated.

---

### Requirement: Model Loading Does Not Bypass Instance Readiness

Successful materialization alone SHALL not imply Model Instance readiness.

#### Scenario: Provider warmup pending

Given model weights are materialized

But required Provider warmup is pending

When Runtime reports instance state

Then the Model Instance is not yet ready.

---

### Requirement: Loading Failure Prevents Ready Instance

If Model Loading fails, Runtime SHALL not expose a ready Model Instance.

#### Scenario: Materialization failed

Given materialization fails

When loading ends

Then Runtime reports failed loading and no ready instance is available.

### Requirement: Model Loading Resolves Compatible Model Component

Model Loading SHALL resolve a compatible Model Component or Runtime-native
architecture implementation before materializing architecture-specific model
state.

#### Scenario: Resolve Qwen component

Given a Model Artifact declares architecture family `qwen`

When Model Loading validates the artifact for inference

Then Runtime resolves a compatible Model Component or Runtime-native
architecture implementation before materialization.

---

### Requirement: Model Loading Uses Model Component Without Bypassing Trust

Model Loading SHALL be allowed to use Model Component metadata for architecture compatibility,
config validation, target module declaration, graph metadata preparation, and
warmup graph construction.

Model Loading SHALL NOT allow a Model Component to bypass Model Artifact trust
validation, memory admission, Runtime policy, or Provider resolution.

#### Scenario: Compatible component with untrusted artifact

Given a compatible Model Component exists

And the Model Artifact is untrusted

When Model Loading validates the artifact

Then loading fails before materialization.

---

### Requirement: Model Loading Uses Authorized Config Data

Model Loading SHALL provide Model Components only Runtime-authorized artifact
metadata and config data.

Model Components SHALL NOT read arbitrary filesystem paths during loading.

#### Scenario: Config path denied

Given a Model Component attempts to read an arbitrary config file path

When Model Loading checks Component authority

Then Runtime denies filesystem authority.

