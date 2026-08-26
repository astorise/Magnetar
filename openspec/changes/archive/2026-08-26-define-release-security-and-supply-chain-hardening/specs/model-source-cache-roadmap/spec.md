## ADDED Requirements

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