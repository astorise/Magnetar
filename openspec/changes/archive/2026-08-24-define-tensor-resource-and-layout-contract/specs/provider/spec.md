## ADDED Requirements

### Requirement: Providers May Own Opaque Tensor Storage

Providers MAY own opaque Tensor Resource storage, and provider-owned storage SHALL remain opaque to Components and clients.

#### Scenario: Provider-owned tensor

Given Provider creates opaque output

When Runtime records Tensor Resource

Then metadata is tracked without exposing Provider handle.

---

### Requirement: Provider Tensor Access Occurs Through Runtime Invocation

Providers SHALL access tensor storage only through Runtime-created invocations and Provider-owned internals.

#### Scenario: Component tries Provider tensor access

Given Component requests Provider-owned tensor storage

When Runtime authorizes access

Then access is denied.

---

### Requirement: Provider Reports Tensor Metadata Effects

Providers SHALL report output readiness, residency, and metadata effects after Kernel execution.

#### Scenario: Kernel output

Given Provider executes Kernel

When it returns result

Then Runtime can update Tensor Resource metadata.