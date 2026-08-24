# Define Prefix Cache Model

## Why

Magnetar now has a KV Cache model.

KV cache represents Runtime-owned reusable attention key/value state.

However, reusing KV cache efficiently across requests requires a separate
Prefix Cache model.

The Prefix Cache is not the KV cache itself.

It is the indexing, matching, compatibility, sharing, isolation, and eviction
policy that determines when a previously computed prefix can be reused.

Without a first-class Prefix Cache model, prefix reuse may become hidden inside:

- Generation
- KV cache internals
- Scheduler
- Session state
- Provider execution
- batching logic

That would make prompt privacy, cache sharing, tenant isolation, compatibility,
Resource Affinity, memory pressure, invalidation, and observability unsafe.

This change defines the Prefix Cache model.

## What Changes

This change introduces Prefix Cache as a first-class Runtime concept.

A Prefix Cache SHALL index reusable prompt-prefix state.

A Prefix Cache entry MAY reference one or more sealed KV cache resources.

Prefix Cache SHALL define:

- prefix fingerprinting
- prefix identity
- prefix matching
- cache entry lifecycle
- compatibility checks
- sharing policy
- privacy policy
- session policy
- memory policy
- Resource Affinity constraints
- Provider/Device residency constraints
- eviction
- invalidation
- observability

The exact Rust type names are implementation-defined.

## Prefix Cache Is Runtime-Owned

Prefix cache identity, lookup, insertion, sharing, invalidation, eviction, and
cleanup SHALL be owned by the Runtime.

Clients and Components SHALL NOT forge prefix cache entries.

A prefix cache identifier SHALL be opaque.

A prefix cache identifier SHALL NOT expose raw prompts, raw tokens, raw KV cache
contents, Provider handles, Device handles, or memory pointers.

## Prefix Cache Is Not KV Cache

A Prefix Cache entry may reference sealed KV cache state.

The Prefix Cache itself is an index and policy layer.

Conceptual relationship:

```text
Prefix Cache Entry
    ├── prefix fingerprint
    ├── compatibility metadata
    ├── sharing policy
    ├── privacy policy
    ├── Resource Affinity
    └── references sealed KV Cache
```

The KV cache stores attention state.

The Prefix Cache decides whether that state may be reused.

## Prefix Cache Is Not Client Conversation

The Prefix Cache SHALL NOT store client conversation history, messages,
workspace files, tools, Git state, network state, or secrets.

It may store redacted metadata derived from validated token prefixes.

Raw prompt text SHALL NOT be stored by default.

## Prefix Fingerprint

A prefix fingerprint SHALL be derived from validated token IDs and relevant
model/tokenizer metadata.

A fingerprint SHOULD include or bind to:

- model identity
- model revision
- architecture metadata
- tokenizer identity
- tokenizer revision
- tokenizer vocabulary compatibility
- token prefix
- position encoding metadata
- special token behavior
- template version where relevant
- sampling-independent context metadata where relevant

A fingerprint SHALL NOT be based only on raw prompt text.

A fingerprint SHALL NOT expose raw prompt text by default.

A fingerprint SHALL NOT be treated as authority by itself.

## Prefix Match

A prefix match occurs when a request token prefix is compatible with a cached
entry.

Matching SHALL validate:

- prefix fingerprint
- model compatibility
- tokenizer compatibility
- token prefix length
- position compatibility
- KV cache compatibility
- Resource Affinity
- Provider/Device residency
- session policy
- sharing policy
- privacy policy
- cache lifecycle
- Runtime policy

A fingerprint hit alone SHALL NOT be sufficient if compatibility checks fail.

## Match Kinds

Prefix cache lookup MAY return different match kinds.

Initial match kinds SHOULD include:

```text
miss
exact-prefix-hit
partial-prefix-hit
incompatible-hit
policy-denied-hit
stale-hit
evicted-hit
```

Semantics:

- `miss`: no candidate found
- `exact-prefix-hit`: full requested prefix is reusable
- `partial-prefix-hit`: a shorter compatible prefix is reusable
- `incompatible-hit`: fingerprint/index found but compatibility failed
- `policy-denied-hit`: candidate exists but sharing/reuse policy denies it
- `stale-hit`: candidate exists but is stale
- `evicted-hit`: index refers to an evicted entry

## Partial Prefix Reuse

Prefix Cache MAY support partial prefix reuse.

Partial prefix reuse SHALL be explicit.

If a partial prefix is reused, Runtime SHALL continue prefill from the reused
prefix boundary.

Partial reuse SHALL validate positional and attention compatibility.

## Entry Lifecycle

Prefix cache entries SHOULD have lifecycle states:

```text
creating
ready
sealed
active
stale
invalid
evicting
evicted
released
failed
```

Semantics:

- `creating`: entry metadata or backing KV cache is being prepared
- `ready`: entry can be considered for reuse
- `sealed`: entry is read-only and safe for reuse where policy allows
- `active`: entry is currently referenced by generation
- `stale`: entry exists but should be revalidated or not preferred
- `invalid`: entry must not be reused
- `evicting`: entry is being removed
- `evicted`: backing cache no longer available
- `released`: entry metadata and resources are released
- `failed`: creation or validation failed

## Sealed KV Cache Requirement

Reusable prefix cache entries SHOULD reference sealed KV cache state.

Mutable KV cache SHALL NOT be shared across unrelated operations unless explicit
policy allows safe mutation.

Default policy SHOULD require sealing before sharing.

## Prefix Cache Scope

Prefix cache scope SHALL be explicit.

Initial scopes SHOULD include:

```text
operation
session
model-instance
runtime
tenant
private
shared
```

Semantics:

- `operation`: only current operation may reuse
- `session`: reuse within one Inference Session
- `model-instance`: reuse for one loaded model context
- `runtime`: reuse within one Runtime process
- `tenant`: reuse within an authorized isolation group
- `private`: reuse only by the owner
- `shared`: reuse across authorized owners according to policy

## Sharing Policy

Prefix cache sharing SHALL be explicit and policy-controlled.

Policy SHOULD consider:

- session identity
- user or tenant identity where available
- model identity
- tokenizer identity
- prompt privacy
- cache scope
- sealed state
- mutability
- Resource Affinity
- memory pressure
- client authorization
- Runtime administrator policy

Default policy SHOULD be conservative.

Cross-session sharing SHALL be disabled unless explicitly enabled.

## Privacy

Prefix cache state may reveal information about prompts.

Therefore:

- raw prompt text SHALL NOT be stored by default
- raw prompt text SHALL NOT be logged by default
- raw token sequences SHOULD be protected or hashed where possible
- fingerprints SHALL be non-reversible where practical
- cache metadata SHALL be redacted
- sharing SHALL require explicit policy
- export SHALL not exist by default
- client-visible diagnostics SHALL avoid sensitive prefix content

## Prefix Entry Metadata

Prefix cache entry metadata MAY include:

- entry ID
- lifecycle state
- prefix fingerprint
- prefix token length
- model identity
- tokenizer identity
- model instance identity
- template identity where relevant
- backing KV cache reference
- Resource Affinity
- Provider binding
- Device binding
- storage dtype
- compute dtype
- position range
- creation timestamp
- last used timestamp
- hit count
- memory size estimate
- scope
- sharing policy
- privacy policy
- eviction priority

Metadata SHALL not expose raw prompts or raw KV cache content by default.

## Resource Affinity

Prefix cache entries inherit Resource Affinity from backing KV cache.

Runtime SHALL not silently reuse a prefix cache entry on incompatible Provider or
Device placement.

If reuse requires movement, Runtime SHALL either:

- perform explicit authorized movement
- rebuild the missing prefix
- fall back to full prefill
- reject reuse
- evict stale entry

Policy decides.

## Memory Manager Relationship

Prefix Cache SHALL integrate with Memory Manager.

Memory Manager SHALL account for:

- backing KV cache memory
- prefix entry metadata memory
- prefix index memory
- fingerprint storage
- lookup workspace
- eviction pressure
- cache hit/miss memory impact

Prefix Cache SHALL not allocate unbounded memory outside Runtime policy.

## Generation Relationship

Generation may query Prefix Cache before prefill.

Conceptual flow:

```text
GenerationRequest
    |
    v
tokenized prefix
    |
    v
Prefix Cache lookup
    |
    +-- hit  -> reuse sealed KV cache up to prefix length
    +-- miss -> prefill normally
```

Generation SHALL validate prefix cache results before reuse.

Generation remains responsible for decode loop and stop conditions.

## Session Relationship

Session policy may enable or disable Prefix Cache.

Session policy may define:

- session-local prefix reuse
- cross-session prefix reuse
- maximum prefix cache memory
- maximum prefix token length
- sharing scope
- privacy policy
- eviction policy
- TTL
- prefix cache persistence after session close

Closing a session SHALL release, retain, or transfer prefix cache entries
according to policy.

## Model Loading Relationship

Model unload or reload may invalidate prefix cache entries.

Prefix cache entries SHALL bind to model identity and loaded model context where
needed.

A prefix cache entry created under one incompatible model context SHALL not be
reused under another.

## Tokenizer Relationship

Prefix fingerprints and matching SHALL depend on tokenizer identity and token
semantics.

A tokenizer mismatch SHALL invalidate or reject reuse.

Template changes that affect tokenization SHALL also invalidate or reject reuse.

## KV Cache Relationship

Prefix Cache entries MAY reference sealed KV cache.

If backing KV cache is evicted, invalid, released, or incompatible, the Prefix
Cache entry SHALL not be reused.

The entry may be evicted, invalidated, or treated as stale.

## Batching Relationship

Continuous batching may use prefix cache entries to reduce prefill work.

This change does not define batching.

It requires the Prefix Cache model to preserve enough metadata for future
batching integration.

## Eviction

Prefix Cache SHALL define eviction behavior.

Eviction MAY be triggered by:

- memory pressure
- cache TTL
- idle TTL
- model unload
- tokenizer update
- template update
- Provider drain
- Device unavailable
- backing KV cache eviction
- policy change
- privacy policy change
- Runtime shutdown

Eviction SHALL release or dereference backing resources according to ownership
policy.

## Invalidation

Prefix Cache invalidation SHALL be explicit.

Invalidation may occur due to:

- model mismatch
- tokenizer mismatch
- template mismatch
- prompt prefix mismatch
- position mismatch
- KV cache invalidation
- Provider/Device incompatibility
- Resource Affinity conflict
- session policy change
- privacy policy change
- detected corruption
- Runtime shutdown

Invalid entries SHALL not be reused.

## Observability

Runtime SHOULD emit observations for:

- prefix cache lookup
- prefix cache hit
- prefix cache miss
- partial prefix hit
- policy denied hit
- incompatible hit
- entry created
- entry sealed
- entry reused
- entry invalidated
- entry evicted
- backing KV cache missing
- sharing denied
- privacy redaction applied
- memory pressure eviction

Observability SHALL not expose raw prompt text, raw token sequences, or raw KV
cache contents by default.

## Browser Target

Prefix Cache model SHALL be platform-neutral.

Browser targets may support reduced prefix caching depending on:

- browser memory limits
- WebAssembly linear memory
- WebGPU buffer availability
- Provider capability
- session policy

Unsupported features SHALL return structured errors.

Browser Prefix Cache SHALL not require Wasmtime or native Provider loading.

## Error Model

Prefix Cache errors SHALL be structured.

Error categories SHOULD include:

- prefix cache disabled
- prefix cache unavailable
- prefix entry not found
- prefix entry incompatible
- prefix fingerprint mismatch
- prefix model mismatch
- prefix tokenizer mismatch
- prefix template mismatch
- prefix position mismatch
- prefix policy denied
- prefix sharing denied
- prefix privacy denied
- prefix stale
- prefix invalid
- prefix evicted
- prefix backing cache missing
- prefix backing cache invalid
- prefix resource affinity conflict
- prefix movement required
- prefix movement unsupported
- prefix memory pressure
- prefix allocation failed
- prefix browser feature unsupported
- prefix internal error

## Non-Goals

This change does not:

- implement continuous batching
- define full scheduler batching policy
- define distributed prefix cache
- define cross-node cache sharing
- expose prompt text through cache keys
- expose raw token sequences by default
- expose raw KV cache contents
- define persistent cache across Runtime restarts
- define remote cache service
- define semantic prompt caching
- define embedding-based cache lookup
- define speculative decoding
- require prefix cache implementation on browser
- require GPU hardware

## Impact

Magnetar gains a safe reuse layer above KV cache.

Generation can reuse compatible prefixes without hiding policy inside Scheduler
or Provider internals.

The resulting model is:

```text
GenerationRequest
    |
    v
Tokenizer.encode
    |
    v
prefix fingerprint
    |
    v
Prefix Cache
    |
    +-- miss -> full prefill
    |
    +-- hit
          |
          v
      sealed KV Cache
          |
          v
      continue prefill/decode
```