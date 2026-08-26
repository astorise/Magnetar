# model-source-cache-roadmap Specification

## Purpose
TBD - created by archiving change define-post-baseline-model-source-and-cache-roadmap. Update Purpose after archive.
## Requirements
### Requirement: Post-Baseline Model Source And Cache Roadmap

Magnetar SHALL define a post-baseline roadmap for model source resolution and
artifact cache.

#### Scenario: Roadmap available

Given model format roadmap is defined

When source/cache work begins

Then source kinds, cache rules, trust, integrity, and boundaries are defined.

---

### Requirement: Source Provides Artifact Candidate

A model source SHALL provide artifact bytes and metadata as an artifact
candidate.

Source kind SHALL not imply trust.

#### Scenario: Source candidate

Given local cache source returns artifact candidate

When Model Loading evaluates it

Then trust and integrity are still checked.

---

### Requirement: Cache Stores Validated Bytes And Metadata

Artifact cache SHALL store artifact bytes and metadata under policy.

Cache presence SHALL not imply trust or readiness for loading.

#### Scenario: Cache hit

Given artifact is present in cache

When Runtime loads it

Then Runtime still validates trust, integrity, compatibility, and policy.

---

### Requirement: Runtime Has No Arbitrary Download Authority

Runtime SHALL not perform arbitrary network downloads during inference.

#### Scenario: Remote model reference

Given inference request contains remote model reference

When Runtime validates it

Then Runtime uses authorized source contracts or rejects arbitrary network
access.

---

### Requirement: Runtime Has No Arbitrary Directory Scan Authority

Runtime SHALL not scan arbitrary local directories during inference.

#### Scenario: Directory path

Given model reference is a local directory path

When Runtime validates it

Then Runtime requires an authorized local directory source or rejects it.

---

### Requirement: CLI Owns User-Facing Source UX

`magnetar-cli` MAY own user-facing download, import, alias, cache, prune, and inspect operations, and CLI SHALL not bypass Runtime artifact validation.

#### Scenario: CLI model pull

Given CLI downloads model artifacts

When Runtime later loads them

Then Runtime still validates normalized artifacts.

---

### Requirement: Source Kinds Are Explicit

Source kinds SHALL be explicit.

#### Scenario: Development fixture

Given source kind is development-fixture

When production policy is active

Then policy may reject it.

---

### Requirement: ModelRef Resolution Is Explicit

ModelRef resolution SHALL produce a Model Instance, Model Artifact, source
candidate, or structured failure.

#### Scenario: Ambiguous reference

Given ModelRef matches both alias and cache entry ambiguously

When resolution runs

Then Runtime or CLI returns model-source-ambiguous or model-alias-ambiguous.

---

### Requirement: Aliases Do Not Bypass Validation

Model aliases SHALL not bypass artifact validation, trust, integrity, loading,
or policy.

#### Scenario: Alias to revoked model

Given alias points to revoked cached artifact

When loading runs

Then Runtime rejects it.

---

### Requirement: Artifact Identity Is Digest-Based

Artifact identity SHALL be digest-based where possible.

Human-readable names SHALL not be authoritative identity.

#### Scenario: Same name different digest

Given two artifacts share same name but different digest

When cache resolves them

Then they remain distinct identities.

---

### Requirement: Cache Addressing Is Controlled

Cache entries SHALL be addressed by digest or normalized identity.

Public APIs SHALL not expose raw cache paths by default.

#### Scenario: Inspect cache

Given caller inspects cache metadata

When Runtime or CLI returns metadata

Then raw cache path is redacted unless policy allows disclosure.

---

### Requirement: Cache Entry Metadata

Cache entries SHOULD preserve identity, manifest, source, trust, integrity, format, size, parts, shards, tokenizer, adapter, timestamps, eviction, and validation metadata, and cache metadata SHALL not include raw cache paths by default.

#### Scenario: Cache entry inspected

Given cache entry exists

When diagnostics are requested

Then stable redacted metadata can be returned.

---

### Requirement: Cache Trust Is Policy-Evaluated

Cached trust metadata SHALL be evaluated against current policy.

#### Scenario: Revoked trust

Given artifact was previously trusted

And revocation status now invalidates it

When loading runs

Then Runtime rejects cached artifact.

---

### Requirement: Cache Integrity Is Validated

Cache integrity SHALL be validated before loading.

#### Scenario: Corrupt shard

Given cached shard digest does not match metadata

When loading validates it

Then corrupt cache entry is rejected.

---

### Requirement: Cache Mutation Is Policy-Controlled

Cache mutation SHALL be explicit and policy-controlled.

#### Scenario: Evict active model

Given cache entry is used by active Model Instance

When eviction is requested

Then eviction is denied or deferred.

---

### Requirement: Partial Cache Entries Are Not Loadable By Default

Partial cache entries SHALL not be used for Model Loading unless explicitly
supported and validated.

#### Scenario: Partial download

Given artifact download is incomplete

When loading runs

Then Runtime returns model-cache-partial-entry.

---

### Requirement: Offline Mode Uses Local Sources Only

Offline mode SHALL use local cache, client-provided artifacts, or development
fixtures only.

#### Scenario: Offline remote

Given offline mode is active

When remote source is requested

Then Runtime or CLI returns model-source-offline-unavailable.

---

### Requirement: Authentication Is Outside Core Runtime Inference

Authentication for remote model sources SHALL not be owned by core Runtime
inference.

#### Scenario: Registry token

Given CLI has registry token

When Runtime cache metadata is written

Then token is not stored by default.

---

### Requirement: Source Policy Controls Allowed Sources

Source policy SHALL control allowed source kinds and restrictions.

#### Scenario: Remote denied

Given source policy denies remote sources

When model hub source is requested

Then request is rejected.

---

### Requirement: License And Provenance Preserved

Source and cache metadata SHOULD preserve license and provenance metadata, and license metadata SHALL not be treated as verified unless validated by policy.

#### Scenario: License restricted

Given license policy rejects artifact license

When loading runs

Then artifact is denied by policy.

---

### Requirement: Cache Does Not Imply Memory Residency

Cache presence SHALL not imply model is loaded or memory-resident.

#### Scenario: Cached artifact

Given model artifact is cached

When Memory Manager is inspected

Then no Tensor Resource is resident unless Model Loading materialized it.

---

### Requirement: Source Cache Error Categories

Source and cache failures SHALL use structured error categories.

#### Scenario: Cache corrupt

Given cache entry fails digest validation

When loading runs

Then Runtime returns model-cache-entry-corrupt or model-cache-integrity-failed.

---

### Requirement: Source Cache Observability

Runtime and CLI SHOULD emit source/cache observations with default redaction, and observations SHALL not expose credentials or raw cache paths by default.

#### Scenario: Cache miss

Given cache lookup misses

When observability records it

Then raw cache paths and credentials are absent by default.

### Requirement: Server Uses Authorized Source Cache Contracts

Server API SHALL use authorized source/cache contracts for model references.

#### Scenario: Cached model

Given generation references cached model

When server resolves it

Then cache validation and Model Loading still run.

---

### Requirement: Server Does Not Download During Generation

Server generation SHALL not perform arbitrary model downloads.

#### Scenario: Remote URL

Given generation request includes remote model URL

When server validates it

Then request is rejected or routed through authorized source policy outside
generation path.

### Requirement: Source Cache Release Security Boundary

Source/cache release security SHALL preserve explicit source trust, cache trust,
integrity, and policy validation.

#### Scenario: Alias to untrusted cache

Given alias points to untrusted cache entry

When Runtime loads it

Then release security validation rejects it.

---

### Requirement: Cache Metadata Does Not Store Secrets By Default

Cache metadata SHALL not store credentials or secrets by default.

#### Scenario: Registry token

Given CLI used token to fetch artifact

When cache metadata is written

Then token is absent.

