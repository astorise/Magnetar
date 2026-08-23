# Define KV Cache Model

## Why

Magnetar is an inference Runtime.

Generation now has a token-based contract, and Inference Sessions provide a
Runtime-owned context for reusable inference state.

The next missing foundation is the KV cache.

Transformer generation depends on reusing attention key/value state across
decode steps.

Without a first-class KV cache model, the cache may become hidden inside:

- Generation
- Scheduler
- Provider execution
- Device-specific code
- model implementation code
- batching code

That would make ownership, memory pressure, eviction, cancellation, resource
affinity, session lifecycle, and Provider placement unsafe.

The KV cache must be an explicit Runtime concept.

This change defines the KV Cache model.

## What Changes

This change introduces KV cache as a first-class inference Runtime resource.

A KV cache SHALL represent reusable key/value attention state associated with a
model execution context.

A KV cache MAY be associated with:

- an Inference Session
- a generation operation
- a model instance
- a prefix cache entry
- a batch slot
- a Provider-owned resource
- a Device-bound resource
- Memory Manager allocation records

The exact Rust type names are implementation-defined.

## KV Cache Is Runtime-Owned

KV cache identity, lifecycle, authorization, and cleanup SHALL be owned by the
Runtime.

Clients and Components SHALL NOT forge KV cache identity.

KV cache handles SHALL be opaque Runtime identifiers.

A KV cache identifier SHALL NOT expose raw pointers, Provider handles, Device
handles, memory addresses, or internal model buffers.

## KV Cache Is Not Session

An Inference Session may own or reference KV cache state.

The KV cache remains a distinct resource.

A session may exist without KV cache.

A KV cache may be released while the session remains alive if policy permits.

A KV cache may be retained in a Runtime cache after a session closes if policy
explicitly allows it.

## KV Cache Is Not Model Artifact

A Model Artifact is immutable model data.

A KV cache is mutable runtime state produced by prefill and decode.

KV cache state SHALL NOT be treated as model artifact content.

KV cache state SHALL NOT be trusted by manifest metadata.

## KV Cache Scope

KV cache scope SHALL be explicit.

Initial scopes SHOULD include:

```text
operation
session
model-instance
prefix-cache
batch-slot
runtime-cache
```

Semantics:

- `operation`: cache exists only for one generation operation
- `session`: cache is tied to an Inference Session
- `model-instance`: cache is tied to a loaded model context
- `prefix-cache`: cache is reusable for matching prompts or prefixes
- `batch-slot`: cache is assigned to a scheduling/batching slot
- `runtime-cache`: cache may outlive the immediate owner under policy

## KV Cache Lifecycle

KV cache lifecycle states SHOULD include:

```text
allocating
empty
prefilling
ready
active
sealed
evicting
evicted
invalid
released
failed
```

Semantics:

- `allocating`: memory is being reserved or allocated
- `empty`: cache exists but contains no usable token state
- `prefilling`: prompt tokens are populating the cache
- `ready`: cache can be used for decode
- `active`: cache is currently being read or mutated
- `sealed`: cache is read-only and may be reused as prefix
- `evicting`: cache is being removed under policy
- `evicted`: cache contents are no longer available
- `invalid`: cache is not safe to use
- `released`: resources have been released
- `failed`: creation or use failed

The exact serialized names are implementation-defined.

## KV Cache Identity

KV cache identity SHALL include an opaque Runtime-issued ID.

Identity MAY also reference:

- model instance identity
- tokenizer identity
- session identity
- cache scope
- prompt token hash
- prefix fingerprint
- Provider binding
- Device binding
- memory residency
- dtype/layout metadata

Only the opaque Runtime-issued ID is stable for external references.

Derived metadata SHALL not become authority.

## KV Cache Compatibility

A KV cache SHALL only be reused when compatible.

Compatibility checks SHOULD include:

- model identity
- model architecture
- model revision
- tokenizer identity
- tokenizer vocabulary compatibility
- prompt token prefix
- position encoding configuration
- attention implementation
- number of layers
- number of heads
- head dimension
- grouped-query or multi-query attention configuration
- dtype
- quantization mode where applicable
- Provider/Device residency
- Resource Affinity
- session policy
- Runtime policy

A cache mismatch SHALL reject reuse.

## Prefix Fingerprint

A prefix fingerprint MAY be used to identify reusable prompt prefixes.

The fingerprint SHALL be derived from validated token IDs and relevant model
configuration.

A prefix fingerprint SHALL NOT be based on raw prompt text by default.

The fingerprint SHALL not expose raw prompts.

## Prompt Privacy

KV cache metadata and observability SHALL not expose raw prompt text by default.

A KV cache may reveal information about prompt length, model identity, or memory
usage.

Such metadata SHALL be policy-controlled and redacted where needed.

## Memory Manager Ownership

KV cache memory SHALL be owned and tracked through the Memory Manager.

Memory Manager SHALL own:

- allocation class for KV cache
- residency
- size accounting
- memory pressure
- admission
- eviction pressure
- release
- pending allocation
- placement feasibility
- dtype placement
- Provider/Device binding metadata

Generation SHALL NOT allocate KV cache memory directly.

Scheduler SHALL NOT allocate KV cache memory directly.

Providers may allocate native resources, but Runtime Memory Manager SHALL track
their residency and ownership.

## KV Cache Layout Metadata

KV cache layout metadata SHOULD be explicit.

Metadata MAY include:

- layer count
- head count
- key head count
- value head count
- head dimension
- token capacity
- current token length
- batch dimension
- sequence dimension
- block size
- page size
- dtype
- layout format
- contiguous or paged layout
- quantized cache metadata
- position range

This change defines metadata, not a mandatory physical layout.

## Paged KV Cache

The KV cache model SHOULD allow paged or block-based cache layouts.

Paged KV cache MAY support:

- allocation by page
- sparse occupancy
- reuse of freed pages
- prefix sharing
- batch slot reassignment
- eviction by page
- memory pressure control

This change does not require immediate paged cache implementation.

It requires the model not to prevent it.

## Quantized KV Cache

KV cache MAY be stored in a dtype different from compute dtype.

Examples:

```text
cache storage dtype = fp16
compute dtype = bf16

cache storage dtype = int8
compute dtype = fp16
```

Memory Manager SHALL distinguish storage dtype and compute dtype for KV cache.

Provider compatibility SHALL be validated.

## Resource Affinity

KV cache residency implies Resource Affinity.

A Provider-owned or Device-bound KV cache SHALL constrain future operations.

Runtime SHALL not silently move KV cache between Providers or Devices.

Any movement, materialization, copy, transfer, or conversion SHALL be explicit.

If a cache is not compatible with the selected placement, Runtime SHALL either:

- reuse compatible placement
- perform explicit movement if supported and authorized
- reject reuse
- rebuild cache
- evict cache

Policy decides.

## Provider Relationship

Providers may create, read, update, and destroy Provider-owned KV cache
resources through Runtime-mediated execution.

Providers SHALL NOT expose raw KV cache handles to Components.

Provider KV cache failures SHALL map to stable Runtime errors.

Provider status, Device status, and memory pressure SHALL influence KV cache
admission and reuse.

## Device Relationship

A KV cache may be Device-bound.

Device-bound cache SHALL not be used on another Device without explicit Runtime
movement or rebuild.

Device unavailability may invalidate, evict, or block use of associated KV
cache according to policy.

## Session Relationship

An Inference Session may own or reference KV cache resources.

Session policy may define:

- whether KV cache is enabled
- maximum cache tokens
- maximum cache memory
- cache scope
- reuse policy
- eviction policy
- prefix reuse allowed
- cache persistence after session close
- cache sharing allowed
- cache privacy policy

Closing a session SHALL release or transfer KV cache resources according to
policy.

## Generation Relationship

Generation SHALL use KV cache through Runtime-managed references.

Prefill may create or populate KV cache.

Decode may read and append to KV cache.

Generation SHALL validate cache compatibility before reuse.

If the KV cache becomes invalid during generation, Runtime policy decides
whether generation fails, rebuilds, retries, or cancels.

## Prefix Cache Relationship

Prefix cache is a reuse policy and index around KV cache state.

This change defines the KV cache foundation.

A later prefix cache contract may define:

- prefix matching
- prefix index
- sharing policy
- eviction policy
- cache keys
- privacy and isolation policy

KV cache model SHALL not prevent prefix caching.

## Batching Relationship

Continuous batching may allocate or assign KV cache blocks to batch slots.

This change does not define batching.

It defines cache scopes and layout metadata sufficient to support future
batching.

Batching SHALL not own KV cache memory directly.

## Cache Sharing

KV cache sharing SHALL be explicit.

Sharing may be allowed only when policy permits.

Sharing policy SHOULD consider:

- tenant/user isolation
- session isolation
- prompt privacy
- model identity
- tokenizer identity
- prefix fingerprint
- cache sealing
- mutability
- Resource Affinity
- memory pressure

Mutable KV cache SHALL not be shared unsafely.

## Cache Sealing

A KV cache may be sealed.

A sealed cache is read-only and may be eligible for prefix reuse or sharing
according to policy.

A sealed cache SHALL not be mutated.

If generation must continue from a sealed cache, Runtime may fork, copy,
reference, or rebuild according to policy and capability.

## Cache Eviction

Runtime SHALL define cache eviction behavior.

Eviction MAY be triggered by:

- memory pressure
- session close
- idle TTL
- total TTL
- model unload
- Device unavailability
- Provider drain
- policy change
- explicit client close
- Runtime shutdown

Eviction SHALL produce stable diagnostics.

Eviction SHALL release Memory Manager resources.

## Cache Invalidation

KV cache invalidation SHALL be explicit.

Invalidation may occur due to:

- model mismatch
- tokenizer mismatch
- prompt mismatch
- position mismatch
- dtype/layout mismatch
- Provider/Device loss
- memory corruption detection
- cancellation policy
- generation error
- session policy change
- Runtime shutdown

Invalid cache SHALL not be reused.

## Cancellation

Cancellation of generation may affect KV cache.

Policy SHALL define whether partially populated KV cache is:

- released
- invalidated
- retained
- sealed up to valid prefix
- moved to prefix cache
- quarantined for diagnostics

Default policy SHOULD be conservative.

## Security And Privacy

KV cache may encode information derived from prompts.

Therefore:

- KV cache SHALL be treated as sensitive inference state
- raw cache content SHALL not be exposed to clients or Components
- cache sharing SHALL require explicit policy
- cache observability SHALL be redacted
- cache persistence SHALL be policy-controlled
- cache export SHALL not exist by default

## Browser Target

The KV cache model SHALL be platform-neutral.

Browser targets may support a reduced KV cache model depending on:

- browser memory limits
- WebAssembly linear memory
- WebGPU buffers
- browser-compatible Provider implementations
- session policy

Unsupported cache features SHALL return structured errors.

Browser KV cache support SHALL not require Wasmtime or native Provider loading.

## Error Model

KV cache errors SHALL be structured.

Error categories SHOULD include:

- cache allocation failed
- cache admission denied
- cache not found
- cache incompatible
- cache invalid
- cache evicted
- cache released
- cache capacity exceeded
- cache position mismatch
- cache prompt mismatch
- cache model mismatch
- cache tokenizer mismatch
- cache dtype mismatch
- cache layout mismatch
- cache provider mismatch
- cache device mismatch
- cache movement required
- cache movement unsupported
- cache sharing denied
- cache sealed
- cache mutation denied
- cache memory pressure
- cache provider failure
- cache device unavailable
- cache cancelled
- cache internal error

## Observability

Runtime SHOULD emit observations for:

- cache allocation requested
- cache allocation completed
- cache allocation failed
- cache prefill started
- cache prefill completed
- cache decode append
- cache hit
- cache miss
- cache compatibility failed
- cache sealed
- cache evicting
- cache evicted
- cache invalidated
- cache released
- cache memory pressure
- cache movement required
- cache sharing denied

Observability SHALL not expose raw prompt text or raw cache contents by default.

## Non-Goals

This change does not:

- define full prefix cache index
- define continuous batching
- define paged attention implementation
- require paged KV cache immediately
- define model loading lifecycle fully
- define Provider-specific KV cache ABI
- define remote cache storage
- define distributed cache sharing
- define cross-node KV cache movement
- expose raw KV cache content
- expose KV cache to Components
- define persistent cache across Runtime restarts
- define browser WebGPU implementation
- define sampling behavior
- define chat conversation storage

## Impact

Magnetar gains a stable KV cache ownership model.

Generation can now depend on Runtime-managed cache state instead of hiding it in
Provider or Scheduler internals.

The inference state model becomes:

```text
Inference Session
        |
        +-- Generation operation
        |       |
        |       +-- prefill
        |       +-- decode
        |
        +-- KV Cache
                |
                +-- Memory Manager allocation
                +-- Resource Affinity
                +-- Provider/Device residency
                +-- lifecycle
                +-- policy
```

This prepares future changes:

- model loading contract
- sampling and logits processing contract
- prefix cache model
- continuous batching contract