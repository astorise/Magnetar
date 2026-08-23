# KV Cache Model

The KV cache is Runtime-owned inference state for reusable transformer
attention key/value data. It is not an Inference Session, Model Artifact,
Provider handle, Device handle, Scheduler state, raw model buffer, prompt store,
or prefix-cache index.

Runtime owns KV cache identity, lifecycle, compatibility validation, memory
accounting, residency, Resource Affinity, policy, cleanup, and redacted
observability.

## Scope

KV cache scope is explicit:

- `operation`: cache exists for one generation operation.
- `session`: cache is tied to an Inference Session policy and lifecycle.
- `model-instance`: cache is tied to a loaded model context.
- `prefix-cache`: cache may be reused through a future prefix-cache index.
- `batch-slot`: cache is assigned to a future batching slot.
- `runtime-cache`: cache may outlive the immediate owner under Runtime policy.

## Lifecycle

KV cache lifecycle is represented by stable states:

- `allocating`: memory or provider resources are being reserved.
- `empty`: cache exists without usable token state.
- `prefilling`: prompt tokens are populating the cache.
- `ready`: cache can be reused for decode.
- `active`: cache is being read or appended.
- `sealed`: cache is read-only.
- `evicting`: cache cleanup has started.
- `evicted`: cache contents are unavailable.
- `invalid`: cache must not be reused.
- `released`: resources have been released.
- `failed`: creation or use failed.

Transitions are explicit. Terminal or invalid cache state cannot be silently
reused.

## Compatibility

Reuse requires compatibility across model identity, model architecture,
revision, tokenizer identity, tokenizer vocabulary, prefix fingerprint,
position encoding, attention implementation, layout, dtype, quantization,
Provider/Device residency, Resource Affinity, and policy.

Mismatches are reported through structured `KvCacheError` categories such as
`CacheModelMismatch`, `CacheTokenizerMismatch`, `CachePromptMismatch`,
`CacheDTypeMismatch`, `CacheProviderMismatch`, and `CacheDeviceMismatch`.

## Prefix Fingerprint

Prefix fingerprints are derived from validated token IDs plus relevant model
and tokenizer metadata. They do not include raw prompt text by default and do
not grant authority by themselves.

Prefix-cache indexing remains a future policy layer around KV cache state.

## Memory

KV cache allocations use `MemoryAllocationClass::KvCache` and are requested
through `MemoryManager`. Layout metadata records storage dtype, compute dtype,
capacity, current token length, batch and sequence dimensions, position range,
and contiguous, paged, block-based, or provider-opaque layout shape.

Runtime releases associated Memory Manager allocations when KV cache resources
are evicted, released, or cleaned up through session close or expiry.

## Resource Affinity

KV cache residency implies Resource Affinity. Provider-owned or Device-bound
cache constrains future reuse. Runtime rejects incompatible placement or
requires explicit movement, rebuild, or rejection according to policy.

Clients cannot forge cache affinity. Runtime derives affinity from owned cache
state and validates requested placement against it.

## Provider And Device

Providers may own native KV cache resources, but raw provider cache handles stay
inside Runtime/Provider internals. Portable APIs expose only opaque Runtime
KV cache identity.

Device-bound cache cannot be used on another Device without explicit Runtime
movement or rebuild. Provider and Device unavailability, drain, pressure, or
reset can invalidate, evict, or block cache use according to policy.

## Session And Generation

Sessions may reference KV cache resources through `SessionPolicy::kv_cache_policy`
and `SessionResources::kv_cache`. Session policy controls cache budget, reuse,
sharing, persistence, privacy, and cleanup on close.

Generation creates or populates cache during prefill, reads and appends cache
during decode, and validates compatibility before reuse through Runtime-managed
cache references.

## Sealing

A sealed KV cache is read-only. It may be eligible for prefix reuse or sharing
when policy allows it. Mutation of sealed cache is rejected unless Runtime forks,
copies, references, or rebuilds according to capability and policy.

## Eviction

Eviction may be triggered by memory pressure, session close, TTL expiry, model
unload, Provider drain, Device unavailability, explicit release, policy change,
or Runtime shutdown. Eviction records redacted observations and releases Memory
Manager resources.

## Invalidation

Invalidation prevents reuse. Causes include model mismatch, tokenizer mismatch,
prompt mismatch, position mismatch, dtype/layout mismatch, Provider or Device
loss, corruption detection, cancellation policy, generation error, session
policy change, and Runtime shutdown.

## Cancellation

Cancellation policy defines whether partially populated cache is released,
invalidated, retained, sealed up to a valid prefix, moved to future prefix-cache
policy, or quarantined for diagnostics. Conservative policy releases or
invalidates partial state by default.

## Privacy

KV cache is sensitive inference state because it is derived from prompts.
Runtime does not expose raw cache content, raw prompt text, or raw provider
handles by default. Observability is redacted and cache sharing requires
explicit policy.

Components do not receive raw KV cache contents or provider-native handles.

## Browser Compatibility

The KV cache model is platform-neutral. Browser targets may use reduced support
based on WebAssembly linear memory, browser memory limits, and future WebGPU
buffers. Unsupported features return structured errors and do not require
Wasmtime or native Provider loading.

## Non-Goals

This model does not define the full prefix-cache index, continuous batching,
paged-attention implementation, provider-specific KV cache ABI, remote or
distributed cache storage, cross-node movement, persistent cache across Runtime
restarts, sampling behavior, chat history storage, or raw cache export.
