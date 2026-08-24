## ADDED Requirements

### Requirement: Provider Selection Remains Runtime-Owned Through Inference API

Runtime Inference API callers SHALL not directly select Providers or Devices. Provider and Device preferences are non-authoritative policy inputs only.

#### Scenario: Provider preference

Given request prefers Reference CPU

When Runtime selects execution path

Then Runtime validates readiness, compatibility, memory, and policy before use.

---

### Requirement: Inference API Does Not Expose Provider Internals

Runtime Inference API SHALL not expose Provider handles, Device handles, Kernel handles, or native Provider diagnostics beyond redacted stable metadata.

#### Scenario: Provider unavailable diagnostic

Given Provider is unavailable

When diagnostics are returned

Then they include redacted Provider status summary only.