## ADDED Requirements

### Requirement: Source Cache Observability Is Redacted

Source/cache observations SHALL be redacted by default.

#### Scenario: Cache lookup

Given cache lookup occurs

When observation is emitted

Then raw cache paths, credentials, raw file contents, and raw weights are absent.

---

### Requirement: Source Cache Observability Preserves Correlation

Source/cache observations SHOULD include correlation IDs linking model resolution, cache lookup, normalization, validation, and loading, and correlation identifiers SHALL not themselves expose redacted metadata.

#### Scenario: Cache corrupt

Given cache entry is corrupt

When loading fails

Then observations can correlate source resolution and integrity failure.