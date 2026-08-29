## ADDED Requirements

### Requirement: Staging Is Separate From Accepted Cache

Uncommitted ingestion bytes SHALL not be visible as accepted Kernel Cache
entries.

#### Scenario: Validation in progress

Given blob is staged

When Runtime searches accepted Kernel Cache

Then staged blob is not returned as accepted candidate.

---

### Requirement: Quarantine Is Separate From Accepted Cache

Quarantined artifact SHALL not appear in normal accepted candidate lookups.

#### Scenario: Quarantined CUBIN

Given digest is retained for review

When Kernel selection runs

Then quarantined entry cannot be selected.

---

### Requirement: Atomic Logical Artifact Publication

Kernel Cache SHALL support transactionally publishing complete logical artifact
metadata.

#### Scenario: Required blobs already deduplicated

Given blobs exist but manifest metadata has not committed

When lookup occurs

Then new logical Kernel is not visible until transaction commits.

---

### Requirement: Deduplication Does Not Grant Trust

Reuse of an existing digest blob SHALL NOT bypass current trust evaluation.

Existing digest blob MAY be reused without making a new manifest trusted.

#### Scenario: Trusted artifact and untrusted manifest share source blob

Given same source bytes are referenced

When second manifest imports

Then second artifact trust is evaluated independently where required.

---

### Requirement: Revocation Metadata Is Independent From Cache Deletion

Eviction/deletion of cached bytes SHALL not by itself erase revocation record.

#### Scenario: Revoked artifact evicted

Given bytes later reappear

When imported

Then revocation can still block it.