## ADDED Requirements

### Requirement: Model Loading Enforces Release Trust

Release baseline Model Loading SHALL enforce artifact trust and integrity.

#### Scenario: Integrity failure

Given fixture artifact digest mismatches

When release E2E runs

Then Model Loading fails and release gate fails.

---

### Requirement: Model Loading Rejects Cache Trust Shortcut

Model Loading SHALL not load cached artifact merely because cache entry exists.

#### Scenario: Cached artifact

Given cached artifact lacks valid trust status under current policy

When loading runs

Then loading is denied.