## ADDED Requirements

### Requirement: Manifest Import Observability

Runtime MAY observe Kernel Manifest parsing and validation lifecycle, and when observability is enabled, emitted records SHALL comply with default redaction rules.

#### Scenario: Bundle imported

Given valid bundle

When imported

Then schema version, manifest digest and artifact counts may be observed.

---

### Requirement: Artifact Payloads Redacted

Observability SHALL not emit raw Kernel source or compiled blobs by default.

#### Scenario: Digest mismatch

Given binary integrity fails

When error is logged

Then expected/actual digest may appear while binary bytes do not.

---

### Requirement: Sensitive Locators Redacted

External artifact location diagnostics SHALL redact credentials and sensitive
path components.

#### Scenario: Credential-bearing URI rejected

Given malformed locator includes secret

When diagnostic is emitted

Then credential value is absent.

---

### Requirement: Trust Claims And Decisions Distinguished

Observability SHOULD distinguish asserted provenance from evaluated trust, and diagnostics SHALL NOT label an unauthenticated claim as an authenticated trust decision.

#### Scenario: Publisher claim

Given manifest says publisher A

When trust is denied

Then diagnostics do not represent publisher claim as authenticated fact.