## ADDED Requirements

### Requirement: Security Documentation Matches Component Runtime Controls
Security documentation SHALL accurately describe implemented Component Runtime controls and remaining gaps.

#### Scenario: Wasmtime controls documented
- **WHEN** Wasmtime fuel, deadline/epoch interruption, resource limits, no ambient WASI authority, and Component Artifact trust checks are implemented
- **THEN** `SECURITY.md` describes them as implemented controls rather than known gaps.

#### Scenario: Remaining gaps precise
- **WHEN** a security limitation remains open
- **THEN** `SECURITY.md` documents the limitation precisely without listing already implemented controls as absent.
