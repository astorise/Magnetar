# prefix-cache Specification

## Purpose
TBD - created by archiving change define-prefix-cache-model. Update Purpose after archive.
## Requirements
### Requirement: Prefix Cache

Magnetar SHALL define Prefix Cache as a Runtime-owned prefix reuse index and
policy layer.

Prefix Cache MAY reference sealed KV cache state.

#### Scenario: Lookup prefix

Given a generation request has validated input tokens

When Runtime checks Prefix Cache

Then Runtime returns a hit, miss, or structured non-reusable result.

---

### Requirement: Prefix Cache Is Runtime-Owned

Runtime SHALL own Prefix Cache identity, lookup, insertion, sharing,
invalidation, eviction, and cleanup.

Clients and Components SHALL NOT forge Prefix Cache entries, and Runtime SHALL
reject fabricated entries.

#### Scenario: Forged entry

Given a caller submits a fabricated prefix entry ID

When Runtime resolves it

Then Runtime rejects it as not found or unauthorized.

---

### Requirement: Prefix Cache Is Not KV Cache

Prefix Cache SHALL be distinct from KV cache.

A Prefix Cache entry may reference sealed KV cache but SHALL not expose raw KV
cache content.

#### Scenario: Prefix hit

Given a prefix cache entry matches a request

When Runtime reuses it

Then Runtime uses the referenced sealed KV cache through Runtime-managed
references.

---

### Requirement: Prefix Cache Is Not Client Conversation

Prefix Cache SHALL NOT store client conversation, workspace, filesystem, Git,
network, tool, or secret state.

#### Scenario: Conversation messages

Given a client conversation contains multiple messages

When Prefix Cache records reuse metadata

Then it stores only inference-scoped prefix metadata, not the conversation
object.

---

### Requirement: Prefix Fingerprint

Prefix fingerprints SHALL be derived from validated token IDs and relevant
model/tokenizer/template metadata.

Raw prompt text SHALL NOT be used as the stored cache key by default.

#### Scenario: Compute fingerprint

Given tokenized prefix P

When Runtime computes a fingerprint

Then it derives a non-authoritative fingerprint from tokens and metadata.

---

### Requirement: Prefix Match Requires Compatibility

A prefix fingerprint hit SHALL not be sufficient for reuse.

Runtime SHALL validate compatibility before reuse.

#### Scenario: Fingerprint hit but model mismatch

Given an index candidate is found

But the model context differs

When Runtime validates reuse

Then Runtime returns incompatible hit or miss according to policy.

---

### Requirement: Match Kinds

Prefix cache lookup SHALL distinguish match kinds.

Match kinds SHOULD include miss, exact-prefix-hit, partial-prefix-hit,
incompatible-hit, policy-denied-hit, stale-hit, and evicted-hit.

#### Scenario: Policy denied candidate

Given a compatible entry exists

But sharing policy denies reuse

When lookup runs

Then Runtime returns policy-denied-hit or equivalent diagnostic.

---

### Requirement: Partial Prefix Reuse

Prefix Cache SHALL model partial prefix reuse when policy enables it.

Partial reuse SHALL validate prefix boundary, position compatibility, and
attention compatibility.

#### Scenario: Partial hit

Given the cache contains a reusable prefix of the request

When partial reuse is enabled

Then Runtime reuses the prefix and continues prefill from the boundary.

---

### Requirement: Prefix Entry Lifecycle

Prefix cache entries SHALL have lifecycle state.

States SHOULD include creating, ready, sealed, active, stale, invalid, evicting,
evicted, released, and failed.

#### Scenario: Entry evicted

Given an entry is evicted

When a request attempts reuse

Then Runtime does not use the entry.

---

### Requirement: Reusable Entries Reference Sealed KV Cache

Reusable shared prefix entries SHALL reference sealed KV cache state by default.

Mutable KV cache SHALL not be shared by default.

#### Scenario: Mutable cache

Given backing KV cache is active and mutable

When another session attempts reuse

Then Runtime denies reuse unless explicit policy allows it safely.

---

### Requirement: Prefix Cache Scope

Prefix cache scope SHALL be explicit.

Scopes SHOULD include operation, session, model-instance, runtime, tenant,
private, and shared.

#### Scenario: Session scope

Given an entry is session-scoped

When another session attempts reuse

Then Runtime denies reuse unless policy allows broader sharing.

---

### Requirement: Prefix Sharing Policy

Prefix sharing SHALL be explicit and policy-controlled.

Cross-session sharing SHALL be disabled by default.

#### Scenario: Cross-session request

Given session B attempts to reuse session A prefix

When sharing policy is default conservative

Then Runtime returns prefix-sharing-denied.

---

### Requirement: Prefix Privacy Policy

Prefix Cache SHALL protect prompt-derived data.

Raw prompt text, raw token sequences, and raw KV cache content SHALL not be
exposed by default.

#### Scenario: Inspect prefix entry

Given a prefix entry exists

When diagnostics are requested

Then Runtime returns redacted metadata only.

---

### Requirement: Prefix Entry Metadata

Prefix cache entries SHALL track metadata needed for safe reuse.

Metadata MAY include fingerprint, prefix length, model identity, tokenizer
identity, template identity, backing KV cache reference, Resource Affinity,
Provider/Device binding metadata, dtype metadata, position range, timestamps,
hit count, memory estimate, scope, sharing policy, privacy policy, and eviction
priority.

#### Scenario: Entry metadata

Given a prefix entry is ready

When Runtime evaluates reuse

Then it uses metadata without exposing raw prompt content.

---

### Requirement: Prefix Resource Affinity

Prefix cache entries SHALL inherit Resource Affinity from their backing KV
cache.

#### Scenario: Device mismatch

Given a prefix entry is backed by KV cache on Device A

When a request is planned on Device B

Then Runtime rejects reuse, explicitly moves if authorized, or falls back
according to policy.

---

### Requirement: Prefix Cache Memory Managed

Prefix Cache SHALL integrate with Memory Manager for metadata, index, backing KV
cache references, lookup workspace, and eviction pressure.

#### Scenario: Prefix index pressure

Given prefix metadata memory exceeds policy

When Runtime evaluates pressure

Then entries may be evicted according to policy.

---

### Requirement: Generation Uses Prefix Cache Before Prefill

Generation SHALL support querying Prefix Cache before prefill.

#### Scenario: Prefix hit

Given Prefix Cache returns exact-prefix-hit

When Generation starts

Then Runtime may skip or reduce prefill using the referenced sealed KV cache.

---

### Requirement: Session Controls Prefix Cache Use

Session policy SHALL control prefix cache enablement, scope, sharing, memory,
TTL, and persistence.

#### Scenario: Prefix cache disabled

Given session policy disables Prefix Cache

When generation runs

Then Runtime does not perform prefix lookup for that session.

---

### Requirement: Model Changes Invalidate Prefix Entries

Model unload, reload, or incompatible model change SHALL invalidate dependent
prefix entries according to policy.

#### Scenario: Model unload

Given a prefix entry depends on loaded model context M

When M is unloaded

Then Runtime invalidates or evicts the prefix entry.

---

### Requirement: Tokenizer Or Template Changes Invalidate Prefix Entries

Tokenizer or template mismatch SHALL prevent prefix reuse.

#### Scenario: Template changed

Given a prefix entry was created using chat template version A

When request uses template version B

Then Runtime rejects reuse.

---

### Requirement: Backing KV Cache Must Be Usable

Runtime SHALL prevent reuse if a backing KV cache is evicted, invalid, released, or incompatible.

#### Scenario: Backing cache evicted

Given a prefix entry references an evicted KV cache

When lookup runs

Then Runtime returns evicted-hit or miss according to policy.

---

### Requirement: Prefix Cache Eviction

Prefix Cache SHALL define eviction behavior.

Eviction SHALL release or dereference backing resources according to ownership
policy.

#### Scenario: Memory pressure eviction

Given Runtime is under memory pressure

When Prefix Cache evicts an entry

Then entry lifecycle becomes evicted and associated references are released or
dereferenced.

---

### Requirement: Prefix Cache Invalidation

Invalid Prefix Cache entries SHALL not be reused.

#### Scenario: Tokenizer mismatch

Given a prefix entry was created with tokenizer T1

When a request uses tokenizer T2

Then Runtime rejects reuse with prefix-tokenizer-mismatch.

---

### Requirement: Browser-Compatible Prefix Cache

Prefix Cache SHALL be platform-neutral and SHALL not require Wasmtime or native
Provider loading.

#### Scenario: Browser unsupported feature

Given browser target lacks backing cache capability

When Prefix Cache is requested

Then Runtime reports prefix-browser-feature-unsupported or disables prefix cache
according to policy.

---

### Requirement: Prefix Cache Error Categories

Prefix Cache failures SHALL use structured error categories.

#### Scenario: Sharing denied

Given a matching entry exists

But sharing is not authorized

When lookup completes

Then Runtime returns prefix-sharing-denied or equivalent structured result.

---

### Requirement: Prefix Cache Observability

Runtime SHALL define redacted Prefix Cache observation categories.

Observability SHALL not expose raw prompt text, raw token sequences, or raw KV
cache contents by default.

#### Scenario: Cache hit observation

Given prefix cache lookup returns exact-prefix-hit

When observability records the event

Then it may include redacted hit metadata and prefix length.

### Requirement: Prefix Cache May Reduce Prefill Work In Batching

Runtime SHALL allow Prefix Cache hits to reduce prefill work before batching where policy permits.

#### Scenario: Prefix hit

Given an operation has an exact Prefix Cache hit

When Scheduler plans prefill

Then it schedules only the remaining work according to Runtime plan.

---

### Requirement: Prefix Cache Policy Applies During Batching

Batching SHALL not bypass Prefix Cache privacy, sharing, or Resource Affinity
policy.

#### Scenario: Sharing denied

Given a prefix entry matches

But sharing policy denies reuse

When batching plans prefill

Then Runtime does not reuse the entry.

### Requirement: Prefix Cache Binds To Model Instance Compatibility

Prefix Cache entries SHALL include Model Instance identity or compatible model
context metadata where required for safe reuse.

#### Scenario: Instance mismatch

Given prefix entry was created for Model Instance A

When request uses incompatible Model Instance B

Then Runtime rejects reuse.

---

### Requirement: Model Instance Changes Invalidate Prefix Cache

Runtime SHALL invalidate dependent Prefix Cache entries according to policy on
Model Instance unload, invalidation, adapter mutation, or incompatible reload.

#### Scenario: Instance reload

Given a prefix entry depends on old instance state

When instance reload changes compute dtype or adapter state

Then Runtime invalidates or rejects reuse.

---

### Requirement: Prefix Cache Policy Is Exposed Through Inference API

Runtime Inference API SHALL expose Prefix Cache policy inputs without exposing raw prompt text or raw KV cache contents.

#### Scenario: Prefix cache enabled

Given Prefix Cache policy is enabled

When generation prepares prefill

Then Runtime may use Prefix Cache internally and report redacted hit/miss
metadata.

---

### Requirement: Inference API Does Not Expose Prefix Fingerprint Inputs By Default

Raw prompt text and raw token sequences used for Prefix Cache fingerprinting SHALL not be exposed by default.

#### Scenario: Diagnostics requested

Given Prefix Cache miss occurs

When diagnostics are returned

Then raw prompt fingerprint inputs are redacted by default.

