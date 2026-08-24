# Define Model Instance Lifecycle

## Why

Magnetar now distinguishes:

- Model Artifact
- Model Loading
- Model Residency
- Adapter Loading
- Tokenizer
- Generation
- Inference Session
- KV Cache
- Prefix Cache
- Continuous Batching
- Memory Manager
- Provider
- Device

The remaining gap before operators and kernels is the Model Instance lifecycle.

A Model Artifact is immutable data.

Model Loading materializes data.

Model Residency records where loaded data lives.

But inference needs a Runtime-owned model context that is ready, active,
draining, unloadable, observable, and safe to reference from sessions and
generation operations.

Without a Model Instance lifecycle, model state may become hidden inside:

- Model Loading
- Session state
- Generation
- Provider resources
- Memory Manager residency
- Adapter activation
- KV Cache compatibility
- Scheduler state

That would make unload, reload, adapter activation, cache invalidation,
concurrency, readiness, Provider pressure, Resource Affinity, and failure
handling unsafe.

This change defines Model Instance as the Runtime-owned loaded model context.

## What Changes

This change introduces Model Instance as a first-class Runtime concept.

A Model Instance SHALL represent a loaded model context that can be used for
inference when ready.

A Model Instance SHALL bind:

- Model Artifact identity
- model architecture implementation
- Model Residency records
- tokenizer compatibility metadata
- Provider/Device placement metadata
- Resource Affinity
- Runtime policy
- adapter activation state
- associated sessions
- associated KV cache and Prefix Cache dependencies
- lifecycle state
- readiness state
- usage accounting
- observability correlation

The exact Rust type names are implementation-defined.

## Model Instance Definition

A Model Instance is not raw model data.

A Model Instance is not merely a memory allocation.

A Model Instance is the Runtime-owned context created after successful Model
Loading.

Conceptually:

```text
Model Instance =
    Model Artifact identity
  + architecture implementation
  + Model Residency
  + Runtime policy
  + Provider/Device compatibility
  + inference readiness
  + lifecycle state
```

## Model Instance Is Runtime-Owned

Model Instance identity, lookup, lifecycle, readiness, authorization, and cleanup
SHALL be owned by the Runtime.

Clients and Components SHALL NOT forge Model Instance identity.

Model Instance identifiers SHALL be opaque Runtime-issued identifiers.

A Model Instance ID SHALL NOT expose Provider handles, Device handles, memory
pointers, raw model weights, or internal architecture pointers.

## Model Instance Is Not Model Artifact

A Model Artifact may be referenced by many Model Instances.

A Model Instance SHALL reference exactly one primary base Model Artifact.

Multiple Model Instances may exist for the same artifact with different:

- Provider placement
- Device placement
- compute dtype
- residency strategy
- quantization handling
- adapter state
- policy
- browser/native execution path

## Model Instance Is Not Model Residency

Model Residency describes where model data lives.

Model Instance owns or references residency records but also includes lifecycle,
readiness, architecture implementation, policy, and inference coordination.

A Model Instance may have one or more residency records.

## Model Instance Is Not Session

An Inference Session may reference a Model Instance.

A Model Instance may be shared by multiple sessions according to policy.

Closing a session SHALL not automatically unload the Model Instance unless
policy requires it.

## Model Instance Lifecycle

A Model Instance SHALL have lifecycle state.

Initial states SHOULD include:

```text
creating
loading
warming
ready
active
idle
draining
suspended
reloading
unloading
unloaded
failed
invalid
removed
```

Semantics:

- `creating`: Runtime has accepted instance creation
- `loading`: Model Loading is materializing required residency
- `warming`: optional warmup or Provider initialization is running
- `ready`: instance can accept inference operations
- `active`: one or more operations are using the instance
- `idle`: instance is ready but not currently active
- `draining`: instance refuses new operations while existing operations finish
- `suspended`: instance is retained but temporarily unavailable by policy
- `reloading`: instance is being replaced or refreshed
- `unloading`: resources are being released
- `unloaded`: residency and Provider resources were released
- `failed`: instance cannot be used due to failure
- `invalid`: instance is no longer safe for inference
- `removed`: Runtime registry entry was removed

The exact serialized names are implementation-defined.

## Readiness

Lifecycle and readiness SHALL be distinct.

A Model Instance may exist but not be ready.

Readiness SHOULD include:

```text
not-ready
ready
read-only
draining
suspended
failed
```

Readiness SHALL consider:

- Model Residency availability
- Provider readiness
- Device readiness
- adapter state
- memory pressure
- Runtime policy
- required Capabilities
- architecture implementation readiness
- browser/native support availability

## Instance Creation

Model Instance creation SHALL require successful Model Loading or an explicit
policy-controlled loading path.

Creation SHALL validate:

- Model Artifact validation
- Model Artifact trust
- architecture implementation
- Model Residency Plan
- Memory Manager admission
- Provider/Device compatibility
- tokenizer compatibility metadata
- Runtime policy
- browser/native capability constraints

A Model Instance SHALL not become ready until all required readiness checks
succeed.

## Instance Warmup

Model Instance warmup MAY be supported.

Warmup may include:

- Provider initialization
- kernel cache preparation placeholder
- operator graph preparation placeholder
- shape plan preparation placeholder
- tokenizer/model metadata validation
- small test execution
- memory residency verification
- adapter readiness verification

Warmup SHALL be policy-controlled.

Warmup failure SHALL not produce a ready instance.

## Instance Usage

Generation, Sessions, and Scheduler SHALL use Model Instances through
Runtime-managed references.

A Model Instance usage reference SHALL be acquired before operation execution and
released after operation completion.

Runtime SHALL prevent unloading while active references exist unless forced
policy applies.

## Usage Counting

Runtime SHALL track Model Instance usage.

Usage MAY include:

- active operation count
- active session count
- queued operation count
- total request count
- token counts
- last used timestamp
- memory residency size
- KV cache dependencies
- adapter dependencies
- failure count

Usage status SHALL not expose raw prompts, raw weights, or raw handles.

## Instance Sharing

A Model Instance MAY be shared across sessions according to policy.

Sharing policy SHOULD consider:

- tenant/user isolation where available
- model trust
- adapter state
- memory residency
- KV cache privacy
- Prefix Cache privacy
- Provider/Device Resource Affinity
- Runtime policy
- browser/native constraints

Default policy SHOULD allow safe Runtime-local sharing only when privacy and
state mutation boundaries are clear.

## Instance Mutability

Model Instance mutability SHALL be explicit.

Mutability may come from:

- adapter merge
- Provider-specific preparation
- quantization transform
- residency relocation
- reload
- warmup state
- future operator/kernel preparation

Silent mutation affecting inference semantics SHALL be forbidden.

## Adapter Relationship

A Model Instance may have active adapters according to Runtime and session
policy.

Adapter state SHALL be represented explicitly.

Adapter activation may create an instance-specific view or modify residency only
when merge policy explicitly allows it.

Adapter changes may affect:

- readiness
- batching compatibility
- KV cache compatibility
- Prefix Cache compatibility
- determinism metadata
- Provider compatibility
- kernel planning

## KV Cache Relationship

KV cache compatibility SHALL reference Model Instance identity or compatible
instance metadata.

Unloading, reloading, invalidating, or mutating a Model Instance may invalidate
dependent KV caches.

A KV cache created for one incompatible Model Instance SHALL not be reused with
another.

## Prefix Cache Relationship

Prefix Cache entries SHALL bind to Model Instance identity or compatible model
context metadata where needed.

Model Instance unload, reload, adapter change, tokenizer mismatch, or policy
change may invalidate Prefix Cache entries.

## Generation Relationship

Generation SHALL require a ready Model Instance or policy-controlled implicit
load path.

Generation SHALL not run against a merely valid Model Artifact.

Generation SHALL acquire a usage reference to the Model Instance before prefill
or decode begins.

If the Model Instance becomes draining, invalid, failed, or unloaded during
generation, Runtime policy determines whether the operation completes, fails,
cancels, or retries.

## Continuous Batching Relationship

Continuous Batching compatibility SHALL include Model Instance compatibility.

Operations using incompatible Model Instances SHALL not share the same execution
step unless Provider and Runtime policy explicitly support it.

Batching SHALL respect Model Instance readiness, pressure, Resource Affinity,
and active adapter state.

## Provider Relationship

A Model Instance may hold Provider-owned opaque resources.

Provider-owned model resources SHALL remain internal to Runtime and Provider.

Provider health, readiness, pressure, admission, and Device status SHALL affect
Model Instance readiness and scheduling.

Provider failure may move the Model Instance to failed, invalid, suspended, or
draining according to policy.

## Device Relationship

A Model Instance may be Device-bound through residency.

Device-bound residency SHALL imply Resource Affinity.

Device loss, reset, pressure, or unavailability may suspend, invalidate, reload,
or unload a Model Instance according to policy.

## Memory Manager Relationship

Memory Manager SHALL track all Model Instance residency.

Model Instance lifecycle transitions SHALL coordinate with Memory Manager for:

- allocation
- residency update
- pressure
- suspension
- unload
- eviction
- relocation placeholder
- browser memory limits
- Provider-owned memory accounting

A Model Instance SHALL not own raw memory outside Memory Manager accounting.

## Suspension

Runtime MAY suspend a Model Instance.

Suspension may occur due to:

- memory pressure
- Provider pressure
- Device pressure
- administrative policy
- browser lifecycle event
- temporary resource loss

A suspended instance SHALL not accept new inference operations.

Runtime may resume, reload, unload, or fail the instance according to policy.

## Draining

A draining Model Instance SHALL reject new operations while allowing active
operations to complete according to policy.

Draining may be triggered by:

- unload request
- reload request
- policy change
- Provider drain
- Device pressure
- Runtime shutdown
- adapter mutation
- failure isolation

## Unload

Model Instance unload SHALL:

- stop new operation admission
- drain or cancel active operations according to policy
- invalidate or release dependent KV caches according to policy
- invalidate dependent Prefix Cache entries according to policy
- release adapter associations according to policy
- release Memory Manager residency
- release Provider-owned resources
- update lifecycle
- emit observability

Unload SHALL not leave dangling session references.

## Reload

Reload SHALL be treated as a new validated loading process.

Reload MAY produce:

- replacement instance
- updated residency
- different Provider/Device placement
- different compute dtype
- different quantization handling
- different adapter compatibility
- new Resource Affinity

Reload SHALL not silently mutate active inference semantics.

Policy SHALL define whether active sessions migrate, fail, drain, or continue
using the old instance.

## Failure Handling

Model Instance failures SHALL be structured and stateful.

Failures may occur in:

- loading
- warmup
- Provider initialization
- Device residency
- Memory Manager allocation
- adapter activation
- generation execution
- unload
- reload
- cache dependency handling

A failed or invalid Model Instance SHALL not accept new inference operations.

## Browser Target

Model Instance lifecycle SHALL be platform-neutral.

Browser targets may support reduced lifecycle behavior depending on:

- browser memory limits
- WebAssembly linear memory
- future WebGPU buffers
- browser-compatible Provider path
- session policy

Unsupported lifecycle features SHALL return structured errors.

Browser Model Instance lifecycle SHALL not require Wasmtime or native Provider
loading.

## Error Model

Model Instance errors SHALL be structured.

Error categories SHOULD include:

- model instance not found
- model instance not ready
- model instance loading
- model instance warming
- model instance draining
- model instance suspended
- model instance unloading
- model instance unloaded
- model instance failed
- model instance invalid
- model instance removed
- model instance active
- model instance busy
- model instance sharing denied
- model instance policy denied
- model instance reload required
- model instance reload failed
- model instance unload failed
- model instance warmup failed
- model instance Provider unavailable
- model instance Provider not ready
- model instance Provider failed
- model instance Device unavailable
- model instance Device lost
- model instance memory pressure
- model instance residency missing
- model instance adapter incompatible
- model instance KV cache invalidated
- model instance Prefix Cache invalidated
- model instance browser feature unsupported
- internal model instance error

## Observability

Runtime SHOULD emit observations for:

- model instance creation requested
- model instance created
- model instance loading
- model instance warming
- model instance ready
- model instance active
- model instance idle
- model instance draining
- model instance suspended
- model instance reloading
- model instance unloading
- model instance unloaded
- model instance failed
- model instance invalidated
- model instance removed
- model instance usage acquired
- model instance usage released
- model instance sharing denied
- model instance cache invalidation
- model instance memory pressure
- model instance Provider pressure
- model instance Device unavailable

Observability SHALL not expose raw model weights, raw prompts, raw Provider
handles, raw Device handles, raw memory pointers, or raw KV cache contents by
default.

## Non-Goals

This change does not:

- define operator graph semantics
- define kernel contracts
- define kernel registry
- define full Provider kernel ABI
- define distributed model serving
- define remote model instance migration
- define persistent model instances across Runtime restarts
- define model download protocol
- define adapter math
- define sampling behavior
- define chat conversation storage
- expose raw model weights
- expose Provider handles
- expose Device handles
- require GPU hardware
- require browser model loading implementation

## Impact

Magnetar gains a stable boundary between loaded model data and inference use.

The architecture becomes:

```text
Model Artifact
    |
    v
Model Loading
    |
    v
Model Residency
    |
    v
Model Instance
    |
    +-- Sessions
    +-- Generation
    +-- Adapters
    +-- KV Cache
    +-- Prefix Cache
    +-- Continuous Batching
```

This prepares:

- execution graph and operator contract
- kernel contract
- kernel registry and dispatch
- model component contract