# adapter Specification

## Purpose
TBD - created by archiving change define-adapter-loading-contract. Update Purpose after archive.
## Requirements
### Requirement: Adapter Artifact

Magnetar SHALL define Adapter Artifacts as inference data that modify or
augment a compatible base model during inference.

#### Scenario: Classify LoRA file

Given an artifact contains LoRA adapter weights

When Runtime classifies it

Then the artifact is an Adapter Artifact or adapter Model Artifact kind.

---

### Requirement: Adapter Artifact Identity

Adapter Artifact identity SHALL be digest-based.

Logical names, paths, aliases, or tags SHALL not be sufficient identity.

#### Scenario: Same adapter name different digest

Given two adapter artifacts share a logical name

But have different digests

When Runtime records them

Then they are distinct adapter artifacts.

---

### Requirement: Adapter Is Not Provider

An Adapter Artifact SHALL NOT define Provider identity.

#### Scenario: Adapter loaded

Given a LoRA adapter is loaded

When Runtime resolves execution

Then Runtime uses compatible Providers through Runtime Resolution

And does not create a LoRA Provider.

---

### Requirement: Adapter Is Not Kernel

An Adapter Artifact SHALL NOT be treated as a kernel implementation.

#### Scenario: Adapter affects execution

Given a LoRA adapter requires additional low-rank projection operations

When Runtime plans execution

Then adapter metadata informs graph/operator/kernel planning

And the adapter itself is not the kernel.

---

### Requirement: Adapter Method

Adapter method SHALL be explicit.

Unsupported adapter methods SHALL produce structured errors.

#### Scenario: Unsupported method

Given an adapter declares method `custom-x`

And Runtime does not support it

When adapter loading runs

Then loading fails with adapter-method-unsupported.

---

### Requirement: Base Model Compatibility

Adapter Loading SHALL validate adapter compatibility with the base model.

#### Scenario: Wrong base model

Given an adapter targets base model A

And loading is requested for base model B

When validation runs

Then Runtime rejects loading with base-model-incompatible.

---

### Requirement: Target Module Metadata

Adapters SHALL declare target module metadata when the adapter method targets model modules.

Runtime SHALL validate target modules against the base model architecture
implementation.

#### Scenario: Missing target module

Given an adapter targets module `q_proj`

And the base model implementation exposes no compatible module

When validation runs

Then Runtime rejects loading with target-module-missing.

---

### Requirement: Adapter Loading Request

Adapter Loading SHALL accept a structured request including adapter artifact
reference, base model context, target usage, adapter method, dtype, residency,
activation, merge policy, memory budget, capabilities, priority, timeout, and
observability correlation.

#### Scenario: Missing base model

Given an adapter loading request lacks a base model context

When validation runs

Then loading fails with base-model-incompatible or invalid-request.

---

### Requirement: Adapter Lifecycle

Loaded adapters SHALL have lifecycle state.

States SHOULD include requested, validating, planning, allocating,
materializing, ready, active, inactive, merging, merged, unmerging, draining,
unloading, unloaded, failed, and invalid.

#### Scenario: Adapter ready

Given adapter materialization succeeds

When Runtime publishes adapter residency

Then lifecycle becomes ready.

---

### Requirement: Adapter Residency

Runtime SHALL track Adapter Residency separately from base model residency.

#### Scenario: Adapter loaded on Device

Given adapter tensors are materialized on Device memory

When Runtime records residency

Then Adapter Residency is tracked independently from base model residency.

---

### Requirement: Adapter Memory Managed

Adapter memory SHALL be allocated, tracked, admitted, pressured, and released
through Memory Manager.

#### Scenario: Adapter memory denied

Given Memory Manager denies adapter allocation

When adapter loading runs

Then loading fails, queues, or retries according to policy.

---

### Requirement: Adapter Activation Is Explicit

Adapter activation SHALL be explicit and policy-controlled.

#### Scenario: Loaded but inactive

Given an adapter is loaded and ready

When no activation request exists

Then generation does not use the adapter.

---

### Requirement: Adapter Deactivation

Adapter deactivation SHALL be explicit and policy-controlled.

#### Scenario: Deactivate adapter

Given an adapter is active in a session

When deactivation is requested

Then Runtime stops applying it according to policy and lifecycle constraints.

---

### Requirement: Multiple Adapter Policy

Multiple active adapters SHALL be governed by explicit policy.

#### Scenario: Multiple unsupported

Given one adapter is active

And a second adapter activation is requested

When policy is single-adapter-only

Then Runtime rejects the activation.

---

### Requirement: Merge Strategy

Adapter merge or overlay strategy SHALL be explicit.

Silent mutation of base model residency SHALL be forbidden.

#### Scenario: Merge requested

Given merge-on-activation is requested

When Runtime applies the adapter

Then Runtime records affected base residency and invalidates dependent state
according to policy.

---

### Requirement: Adapter Changes Affect KV Cache Compatibility

KV cache compatibility SHALL include active adapter set where adapter changes
affect model outputs.

#### Scenario: Adapter mismatch cache

Given KV cache was created with adapter A active

When generation runs with adapter B

Then Runtime rejects or rebuilds cache according to policy.

---

### Requirement: Adapter Changes Affect Prefix Cache Compatibility

Prefix Cache fingerprints SHALL include active adapter set where relevant.

#### Scenario: Prefix reuse with different adapter

Given a prefix entry was created with adapter A

When request uses no adapter

Then Runtime rejects reuse unless compatibility policy proves it safe.

---

### Requirement: Adapter-Aware Generation

Generation SHALL apply active adapter context during model execution.

Generation SHALL NOT silently activate or load adapters.

#### Scenario: Generation with adapter

Given a session has adapter A active

When generation runs

Then model execution uses adapter A according to Runtime plan.

---

### Requirement: Adapter-Aware Batching

Batch compatibility SHALL include active adapter set and adapter execution
strategy where relevant.

#### Scenario: Incompatible adapters

Given operation A uses adapter X

And operation B uses adapter Y

When Provider cannot batch different adapters

Then Scheduler does not place them in the same batch step.

---

### Requirement: Provider Adapter Capabilities

Providers SHALL advertise adapter capabilities when Provider-supported adapter execution is exposed.

Capabilities MAY include supported adapter methods, maximum rank, supported
dtypes, merge strategies, fused kernels, target modules, quantized adapter
formats, and activation cost.

#### Scenario: Provider lacks LoRA support

Given an adapter requires LoRA execution

And no compatible Provider path exists

When Runtime plans activation

Then activation fails with Provider-adapter-unsupported.

---

### Requirement: Provider Adapter Resources Are Opaque

Provider-owned adapter resources SHALL remain opaque to Components and public
portable APIs.

#### Scenario: Provider adapter handle

Given Provider materializes adapter data into native memory

When Runtime reports adapter status

Then it exposes stable Runtime metadata and not raw Provider handles.

---

### Requirement: Session Adapter Policy

Inference Sessions SHALL define or reference adapter policy.

#### Scenario: Adapter not allowed

Given session policy does not allow adapter A

When activation is requested

Then Runtime rejects activation.

---

### Requirement: Browser-Compatible Adapter Loading

Adapter Loading SHALL be platform-neutral and SHALL not require Wasmtime or
native Provider loading.

#### Scenario: Browser unsupported adapter feature

Given browser target lacks required adapter execution support

When adapter activation is requested

Then Runtime returns browser-feature-unsupported or equivalent structured error.

---

### Requirement: Adapter Error Categories

Adapter failures SHALL use structured error categories.

#### Scenario: Shape mismatch

Given adapter tensor shape is incompatible with target module

When validation runs

Then Runtime returns adapter-shape-mismatch or target-tensor-mismatch.

---

### Requirement: Adapter Observability

Runtime SHALL define adapter observations for loading, validation, materialization,
activation, deactivation, merge, unmerge, unload, cache invalidation, and batching
compatibility.

Observability SHALL not expose raw adapter tensors, raw model weights, raw
Provider handles, or raw prompts by default.

#### Scenario: Adapter activated

Given adapter activation succeeds

When observability records it

Then Runtime emits redacted adapter activation metadata.

### Requirement: Adapter State Belongs To Model Instance Context

Runtime SHALL support associating active adapter state with a Model Instance, session, or
operation according to policy.

#### Scenario: Instance-level adapter

Given adapter A is activated at model-instance scope

When generation uses that instance

Then adapter A is part of the active instance context.

---

### Requirement: Adapter Mutation Affects Model Instance Lifecycle

Adapter merge, unmerge, activation, or deactivation SHALL affect Model Instance
readiness, mutability, cache compatibility, and batching compatibility.

#### Scenario: Merge adapter

Given adapter merge mutates model residency

When merge occurs

Then Model Instance records semantic mutation and invalidates dependent state
according to policy.

### Requirement: Adapters May Modify Execution Graphs

Adapters SHALL represent any adapter modification or extension to Execution Graphs through explicit graph metadata,
additional operators, modified paths, merge graphs, or fused adapter metadata.

#### Scenario: LoRA active

Given LoRA adapter is active

When Runtime builds an MLP graph

Then LoRA path or fused adapter metadata is represented explicitly.

---

### Requirement: Adapter Graph Changes Affect Cache Compatibility

Adapter-induced graph semantic changes SHALL affect KV Cache and Prefix Cache
compatibility where model outputs change.

#### Scenario: Adapter changed graph

Given Prefix Cache entry was created without adapter graph path

When adapter graph path is active

Then Runtime rejects reuse unless policy proves compatibility.

### Requirement: Adapter Validation Uses Model Component Metadata

Adapter Loading SHALL use Model Component target module metadata where available.

#### Scenario: LoRA target validation

Given adapter targets `q_proj`

When Runtime validates the adapter

Then it checks target module metadata from the compatible Model Component.

---

### Requirement: Adapter Graph Changes May Be Produced By Model Component

Model Component SHALL define adapter overlay or merge graph production metadata where supported.

#### Scenario: LoRA overlay graph

Given LoRA adapter is active

When Runtime requests graph production

Then Model Component may emit explicit adapter overlay graph metadata.

