## ADDED Requirements

### Requirement: Release Security Conformance

Conformance SHALL include security release gates for dependency audit status,
license audit status, secret scanning, redaction, native handle exposure, trust
boundaries, and artifact integrity.

#### Scenario: Security conformance

Given release candidate is tested

When security conformance runs

Then release-blocking security checks pass.

---

### Requirement: Redaction Conformance Blocks Release

Redaction conformance failure SHALL block stable release.

#### Scenario: Raw KV cache leak

Given diagnostics expose raw KV cache content

When release conformance runs

Then stable release is blocked.

---

### Requirement: Trust Boundary Conformance Blocks Release

Trust boundary conformance failure SHALL block stable release.

#### Scenario: Cache trust bypass

Given cached artifact loads without trust validation

When release conformance runs

Then stable release is blocked.