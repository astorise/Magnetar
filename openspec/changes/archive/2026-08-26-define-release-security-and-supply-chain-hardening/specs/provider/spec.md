## ADDED Requirements

### Requirement: Provider Trust Model Documented

Release SHALL document that Providers are trusted native code.

#### Scenario: Provider release docs

Given release docs describe Provider system

When security notes are inspected

Then trusted native Provider model is explicit.

---

### Requirement: Dynamic Provider Loading Status Documented

Dynamic Provider loading SHALL be disabled, experimental, or explicitly marked
unstable unless security reviewed.

#### Scenario: Dynamic Provider feature

Given dynamic Provider loading exists

When release metadata is generated

Then support status and security limitations are documented.

---

### Requirement: Provider Native Handles Hidden

Provider native handles SHALL remain hidden in release APIs, diagnostics, and
reports.

#### Scenario: Provider diagnostic

Given Provider diagnostic is emitted

When redaction gate runs

Then no native handle is present.