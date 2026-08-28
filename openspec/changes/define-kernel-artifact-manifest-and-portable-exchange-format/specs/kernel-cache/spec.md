## ADDED Requirements

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