## ADDED Requirements

### Requirement: Component Artifact Package

A Component Artifact Package SHALL be the distribution unit for a
Magnetar-compatible Inference Component.

It SHALL contain or reference:

- executable Component bytes
- Component manifest
- declared digest
- source identity
- optional publisher identity
- optional signature metadata
- optional provenance metadata

#### Scenario: Receive package

Given a Component Distribution Source provides a package

When Magnetar receives it

Then the Runtime treats it as untrusted input until local validation completes.

---

### Requirement: Component Distribution Source

A Component Distribution Source SHALL provide Component Artifact Packages or
resolvable package metadata.

Source identity SHALL be metadata and SHALL NOT imply trust.

#### Scenario: Source is known

Given a package comes from a known source

When trust is evaluated

Then source identity may be considered by policy

But does not automatically mark the package trusted.

---

### Requirement: Source-Declared Digest Is A Claim

A digest provided by a Component Distribution Source SHALL be verified locally.

#### Scenario: Source digest mismatch

Given a source declares digest A

And Magnetar computes digest B from received bytes

When A and B differ

Then the package is rejected.

---

### Requirement: Manifest Validation For Distributed Packages

A distributed Component Package SHALL pass Component manifest validation before
preparation.

#### Scenario: Invalid distributed manifest

Given a package contains malformed manifest data

When Magnetar validates the package

Then the package is rejected before ComponentEngine preparation.

---

### Requirement: Distributed Package WIT Consistency

A distributed package SHALL pass WIT consistency validation between manifest and
actual executable Component.

#### Scenario: Manifest hides import

Given the executable Component imports interface X

And the distributed manifest omits X

When validation runs

Then the package is rejected.

---

### Requirement: Distributed Package Authority Is Inference Scoped

A distributed package SHALL request only Magnetar inference-scoped Component
authority.

#### Scenario: Distributed Component requests Git

Given a distributed package manifest requests Git authority

When Magnetar validates the package

Then the package is rejected as outside Runtime scope.

---

### Requirement: Package Does Not Grant Authority

A Component Artifact Package SHALL NOT grant authority even when it declares
requested authority.

It SHALL NOT grant that authority.

#### Scenario: Package declares compute

Given a package declares `compute-capability`

When Magnetar validates it

Then Runtime policy still decides whether the Component receives a Compute
endpoint in its Link Plan.

---

### Requirement: Package Does Not Imply Trust

A Component Artifact Package SHALL NOT be trusted merely because it exists,
came from a source, or contains a manifest.

#### Scenario: Package from cache

Given a package exists in the local cache

When Magnetar loads it

Then the package still requires integrity and trust validation.

---

### Requirement: Distributed Package Revocation

A distributed package SHALL be rejected if its artifact digest is revoked.

#### Scenario: Revoked package received

Given a source provides a package whose digest is revoked

When Magnetar validates it

Then validation fails before preparation.

---

### Requirement: Distributed Package Compatibility

A distributed package SHALL be checked for Runtime, Capability, Component
Engine, WIT, and inference authority compatibility.

#### Scenario: Package requires unsupported Compute major version

Given a package requires an unsupported Compute major version

When compatibility validation runs

Then the package is rejected.

---

### Requirement: Optional Provenance Metadata

A distributed package SHALL treat provenance metadata as optional,
non-authoritative metadata when present.

Provenance metadata SHALL NOT imply trust by itself.

#### Scenario: Provenance present

Given a package includes source repository and build commit metadata

When trust is evaluated

Then the metadata may be recorded

But trust still depends on Runtime policy.

---

### Requirement: Optional Signature Metadata

A distributed package SHALL treat signature metadata as optional,
non-authoritative metadata when present.

Unsupported or unverified signatures SHALL NOT imply trust by themselves.

#### Scenario: Signed package without trust root

Given a package includes signature metadata

And Runtime has no configured verifier or trust root

When trust is evaluated

Then the package is not trusted solely because a signature is present.

---

### Requirement: Cache Integrity

If Magnetar caches Component packages, cached content SHALL be verified before
use.

#### Scenario: Corrupted cache entry

Given cached bytes no longer match the expected digest

When Magnetar loads the cache entry

Then the cache entry is rejected.

---

### Requirement: Offline Distribution

The distribution contract SHALL support local/offline Component packages.

#### Scenario: Offline local package

Given a trusted local Component package is available

And no external network is available

When Magnetar validates it

Then validation can succeed without contacting a remote service.

---

### Requirement: Distribution Is Not Instantiation

Receiving or resolving a package SHALL NOT instantiate a Component.

#### Scenario: Package fetched

Given Magnetar fetches a Component package

When no instantiation is requested

Then no Component Instance is created.

---

### Requirement: Distribution Is Not Trust

Resolving a logical Component identity to an artifact digest SHALL NOT mark the
artifact trusted.

#### Scenario: Version resolved

Given a source resolves a version request to a digest

When Magnetar receives the candidate

Then Magnetar still validates and applies trust policy before preparation.
