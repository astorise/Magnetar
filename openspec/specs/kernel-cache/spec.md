# kernel-cache Specification

## Purpose
This specification defines the content-addressed Kernel Artifact cache: its separation from Model Artifact cache, Prefix Cache, KV Cache, and Runtime Tensor residency; cache identity and qualification-evidence compatibility keying; immutability of cache entries; integrity validation; eviction behavior relative to active Prepared Kernel state; and explicit pinning policy.
## Requirements
### Requirement: Kernel Artifact Cache

The Kernel Artifact cache SHALL remain distinct from Model Artifact cache, Prefix Cache, and KV Cache.

Magnetar SHOULD support a content-addressed Kernel Artifact cache separate from
model, prefix and KV caches.

#### Scenario: Compiled kernel reused

Given compatible compiled artifact exists in cache

When model is loaded offline

Then Runtime may reuse it without recompilation.

---

### Requirement: Cache Hit Does Not Imply Eligibility

Cache hit SHALL NOT imply trust, qualification, compatibility or active status.

#### Scenario: Cached revoked kernel

Given cached kernel is revoked

When Registry evaluates it

Then it is rejected.

---

### Requirement: Cache Content Is Immutable

A mutation SHALL produce a new digest-addressed entry rather than alter an existing entry in place.

Content-addressed artifacts SHOULD be immutable.

#### Scenario: Binary changes

Given cached binary bytes change

When stored

Then they produce a new digest identity.

---

### Requirement: Qualification Evidence Cache Compatibility

Qualification evidence SHALL be keyed by qualification-relevant context.

#### Scenario: Suite version changes

Given qualification suite version changes incompatibly

When cached evidence is read

Then old evidence is not silently accepted as current.

---

### Requirement: Cache Integrity

Cache entries SHALL be integrity validated according to policy.

#### Scenario: Corrupt binary

Given cached compiled artifact digest mismatches

When read

Then entry is rejected.

---

### Requirement: Cache Eviction Does Not Destroy Active Prepared Kernel

Persistent cache lifecycle SHALL remain distinct from Prepared Kernel lifetime.

#### Scenario: Artifact evicted

Given Prepared Kernel remains active

When persistent compiled artifact is evicted according to policy

Then active native prepared state is not implicitly destroyed.

---

### Requirement: Cache Pinning

Pinning policy SHALL be explicit.

Kernel cache MAY pin artifacts required for active, rollback, offline or
reproducibility policy.

#### Scenario: Rollback candidate

Given previous active kernel is retained for rollback

When cache eviction runs

Then pinned artifact is preserved.

---

### Requirement: Bundle Blobs May Populate Kernel Cache

Validated content-addressed bundle blobs MAY be inserted into Kernel Cache, and insertion SHALL preserve the blob's declared content-addressed digest identity.

#### Scenario: CUBIN import

Given CUBIN digest validates

When import policy allows caching

Then bytes may be stored under same content identity.

---

### Requirement: Bundle Import Does Not Grant Cache Trust

Importing bundle into cache SHALL NOT mark artifact trusted solely from import.

#### Scenario: Local file import

Given user imports unknown generated Kernel

When cache entry is created

Then trust remains separately evaluated.

---

### Requirement: Cache Can Deduplicate Bundle Content

Identical digest blobs SHOULD be reusable across multiple manifests, and deduplication SHALL NOT merge blobs whose digests differ.

#### Scenario: Shared source artifact

Given two manifests reference same source digest

When cached

Then one content-addressed blob may satisfy both references.

---

### Requirement: Cache Corruption Detected Against Manifest

Cached blob SHALL be revalidated where policy requires.

#### Scenario: Cached bytes corrupted

Given manifest expects digest D

When cached content no longer hashes to D

Then content is rejected.
