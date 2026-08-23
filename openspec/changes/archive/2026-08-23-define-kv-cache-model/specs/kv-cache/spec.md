## ADDED Requirements

### Requirement: KV Cache

Magnetar SHALL define KV cache as Runtime-owned inference state for reusable
attention key/value data.

#### Scenario: Prefill creates cache

Given generation begins with input tokens

When prefill executes

Then Runtime may create or populate a KV cache.

---

### Requirement: KV Cache Is Runtime-Owned

KV cache identity, lifecycle, authorization, and cleanup SHALL be owned by the
Runtime.

Clients and Components SHALL NOT forge KV cache identity.

#### Scenario: Forged cache ID

Given a caller submits a fabricated KV cache ID

When Runtime resolves it

Then Runtime rejects it as cache-not-found or unauthorized.

---

### Requirement: KV Cache Is Not Session

An Inference Session MAY own or reference KV cache state, but the KV cache SHALL
remain a distinct resource.

#### Scenario: Session without cache

Given a session is created

When no generation has populated attention state

Then the session may exist without a KV cache.

---

### Requirement: KV Cache Is Not Model Artifact

A KV cache SHALL be mutable Runtime state and SHALL NOT be treated as immutable
Model Artifact content.

#### Scenario: Model artifact trusted

Given a Model Artifact is trusted

When no prefill has run

Then no KV cache exists merely because the artifact is trusted.

---

### Requirement: KV Cache Scope

A KV cache SHALL declare or be associated with an explicit scope.

Scopes SHOULD include operation, session, model-instance, prefix-cache,
batch-slot, and runtime-cache.

#### Scenario: Operation cache

Given a one-shot generation operation creates a cache

When the operation completes

Then operation-scoped cache is released unless policy says otherwise.

---

### Requirement: KV Cache Lifecycle

KV cache SHALL have lifecycle state.

Lifecycle states SHOULD include allocating, empty, prefilling, ready, active,
sealed, evicting, evicted, invalid, released, and failed.

#### Scenario: Cache ready

Given prefill has completed successfully

When the cache is usable for decode

Then its lifecycle is ready.

---

### Requirement: KV Cache Compatibility

Runtime SHALL validate KV cache compatibility before reuse.

Compatibility SHOULD include model, tokenizer, prompt prefix, position encoding,
attention configuration, dtype, layout, Provider, Device, Resource Affinity, and
policy.

#### Scenario: Model mismatch

Given a KV cache was created for model A

When generation for model B attempts reuse

Then Runtime rejects reuse with cache-model-mismatch.

---

### Requirement: Prefix Fingerprint

Runtime SHALL define a prefix fingerprint representation derived from validated
token IDs and relevant model configuration when prefix reuse is evaluated.

The fingerprint SHALL NOT expose raw prompt text by default.

#### Scenario: Fingerprint created

Given a tokenized prompt prefix

When Runtime computes a prefix fingerprint

Then the fingerprint is derived from tokens and model metadata, not raw text.

---

### Requirement: KV Cache Layout Metadata

KV cache SHALL expose or track layout metadata sufficient for Runtime planning.

Metadata MAY include layer count, head count, head dimension, token capacity,
current token length, block/page metadata, dtype, layout format, and position
range.

#### Scenario: Capacity exceeded

Given a cache has capacity for 1024 tokens

When decode would append token 1025

Then Runtime reports cache-capacity-exceeded or reallocates according to policy.

---

### Requirement: Paged Cache Ready Model

KV cache metadata SHALL be able to represent paged or block-based layouts.

This change SHALL NOT require immediate paged cache implementation.

#### Scenario: Paged metadata

Given a Provider supports paged attention

When Runtime records cache metadata

Then page size and block occupancy metadata can be represented.

---

### Requirement: Quantized KV Cache Metadata

KV cache SHALL distinguish storage dtype from compute dtype where applicable.

#### Scenario: INT8 cache storage

Given cache storage dtype is INT8

And compute dtype is FP16

When Runtime validates reuse

Then Provider and Memory Manager compatibility are checked.

---

### Requirement: KV Cache Memory Managed

KV cache memory SHALL be allocated, tracked, admitted, and released through the
Memory Manager.

#### Scenario: Cache allocation denied

Given Memory Manager denies KV cache allocation due to pressure

When generation requests cache creation

Then Runtime fails, queues, or retries according to policy.

---

### Requirement: KV Cache Resource Affinity

KV cache residency SHALL imply Resource Affinity.

Runtime SHALL not silently move Provider-owned or Device-bound KV cache state.

#### Scenario: Device-bound cache

Given a KV cache resides on Device A

When generation is planned on Device B

Then Runtime rejects reuse, performs explicit authorized movement, or rebuilds
according to policy.

---

### Requirement: Provider-Owned Cache Is Opaque

Provider-owned KV cache handles SHALL remain internal to Runtime and Provider.

They SHALL NOT be exposed to Components or public portable APIs.

#### Scenario: Provider cache created

Given Provider creates a native KV cache resource

When Runtime records it

Then Runtime exposes only an opaque Runtime KV cache identity.

---

### Requirement: Session KV Cache Policy

An Inference Session SHALL define or reference policy for KV cache usage.

#### Scenario: KV cache disabled

Given session policy disables KV cache

When generation runs

Then Runtime does not create reusable session KV cache.

---

### Requirement: Generation Uses Runtime Cache References

Generation SHALL use KV cache through Runtime-managed references.

#### Scenario: Decode with cache

Given prefill created a ready KV cache

When decode runs

Then Generation reads or appends cache through Runtime-managed cache state.

---

### Requirement: Sealed KV Cache

A KV cache SHALL define sealed state semantics.

A sealed cache SHALL be read-only.

#### Scenario: Mutate sealed cache

Given a KV cache is sealed

When generation attempts to append to it

Then Runtime rejects mutation or creates a new cache according to policy.

---

### Requirement: KV Cache Sharing Requires Policy

KV cache sharing SHALL be explicit and policy-controlled.

Mutable KV cache SHALL not be shared unsafely.

#### Scenario: Sharing denied

Given session A has a KV cache

When session B attempts reuse

And sharing policy denies it

Then Runtime returns cache-sharing-denied.

---

### Requirement: KV Cache Eviction

Runtime SHALL define KV cache eviction behavior.

Eviction SHALL release Memory Manager resources.

#### Scenario: Memory pressure eviction

Given memory pressure is high

When Runtime evicts a cache

Then the cache transitions to evicted and memory is released.

---

### Requirement: KV Cache Invalidation

Invalid KV cache SHALL not be reused.

#### Scenario: Tokenizer mismatch

Given a cache was created with tokenizer A

When a request using tokenizer B attempts reuse

Then Runtime rejects reuse with cache-tokenizer-mismatch.

---

### Requirement: Cancellation Cache Policy

Generation cancellation SHALL apply explicit policy to partially populated KV
cache.

#### Scenario: Cancel during prefill

Given generation is cancelled during prefill

When cache contains partial state

Then Runtime releases, invalidates, seals, or retains it according to policy.

---

### Requirement: KV Cache Privacy

KV cache SHALL be treated as sensitive inference state.

Raw KV cache contents SHALL not be exposed to clients or Components by default.

#### Scenario: Inspect cache

Given a caller inspects session status

When cache metadata is returned

Then raw cache content is not included.

---

### Requirement: Browser-Compatible KV Cache Model

The KV cache model SHALL be platform-neutral and SHALL not require Wasmtime or
native Provider loading.

#### Scenario: Browser target

Given Runtime runs on a browser target

When KV cache support is unavailable

Then Runtime reports a structured unsupported cache feature error.

---

### Requirement: KV Cache Error Categories

KV cache failures SHALL use structured error categories.

#### Scenario: Cache evicted

Given a generation request references an evicted cache

When Runtime validates it

Then Runtime returns cache-evicted or cache-not-found according to policy.

---

### Requirement: KV Cache Observability

Runtime SHALL define KV cache observation categories for allocation, prefill, decode append,
hit, miss, compatibility failure, sealing, eviction, invalidation, release,
memory pressure, movement requirement, and sharing denial.

Observability SHALL not expose raw prompt text or raw cache contents by default.

#### Scenario: Cache hit

Given a compatible cache exists

When generation reuses it

Then Runtime may emit a cache-hit observation with redacted metadata.
