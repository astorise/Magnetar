## ADDED Requirements

### Requirement: Runtime Validates Distributed Components Locally

The Runtime SHALL validate every distributed Component Artifact Package locally
before preparation.

#### Scenario: External source package

Given an external source provides a package

When Runtime receives it

Then Runtime performs digest, manifest, WIT, compatibility, authority, and trust
validation locally.

---

### Requirement: Runtime Treats Distribution Sources As Untrusted Input

Distribution sources SHALL be treated as untrusted inputs unless Runtime policy
explicitly grants trust to specific digests or source properties.

#### Scenario: Known source provides package

Given a known source provides a Component package

When Runtime evaluates it

Then the package is not executable until local validation and trust evaluation
succeed.

---

### Requirement: Runtime Supports Push Delivery

Runtime SHALL validate Component Artifact Packages pushed by a local or
external source when push delivery is supported.

Push delivery SHALL not bypass validation.

#### Scenario: Client pushes package

Given a client provides package bytes to Runtime

When Runtime receives them

Then Runtime validates the package before preparation.

---

### Requirement: Runtime Supports Pull Resolution

Runtime SHALL validate Component Artifact Packages resolved and fetched from
configured sources when pull resolution is supported.

Pulled packages SHALL not bypass validation.

#### Scenario: Runtime pulls package

Given Runtime is configured with a local source

When Runtime resolves a Component identity

Then it fetches candidate package data

And validates the resulting bytes locally.

---

### Requirement: Runtime Verifies Source Claims

Runtime SHALL verify source-provided digest and manifest claims against received
bytes.

#### Scenario: Source lies about digest

Given a source declares digest A

And the executable bytes hash to digest B

When Runtime validates the package

Then validation fails.

---

### Requirement: Runtime Enforces Inference Scope On Distributed Packages

Runtime SHALL reject distributed Component packages requesting authority outside
Magnetar inference scope.

#### Scenario: Distributed filesystem tool

Given a distributed package requests filesystem authority

When Runtime validates it

Then Runtime rejects it before ComponentEngine preparation.

---

### Requirement: Runtime Cache Does Not Imply Trust

Runtime cache presence SHALL not imply that a Component package is trusted or
executable.

#### Scenario: Cached package exists

Given a package is found in cache

When Runtime loads it

Then Runtime verifies digest and policy before preparation.

---

### Requirement: Runtime Rejects Revoked Distributed Artifacts

Runtime SHALL reject distributed packages whose artifact digest is revoked.

#### Scenario: Revoked digest from trusted source

Given a package comes from a trusted source

But its digest is revoked

When Runtime validates it

Then revocation wins and the package is rejected.

---

### Requirement: Runtime Does Not Require Tachyon

Runtime SHALL not require Tachyon to resolve, validate, trust, or execute a
Component Artifact Package.

#### Scenario: No Tachyon configured

Given no Tachyon source is configured

When Runtime loads a trusted local package

Then Runtime can proceed without Tachyon.

---

### Requirement: Runtime Validates Tachyon-Provided Packages

If Tachyon provides a Component package, Runtime SHALL apply the same validation
as for any other source.

#### Scenario: Tachyon package

Given Tachyon supplies a package

When Runtime receives it

Then Runtime verifies digest, manifest, WIT, compatibility, inference authority,
trust, and revocation locally.

---

### Requirement: Runtime Does Not Transfer Client Authority

Runtime SHALL not inherit authority from the client or source that supplied a
package.

#### Scenario: CLI has Git access

Given `magnetar-cli` has Git access

And it supplies a Component package

When Runtime validates and executes the Component

Then the Component does not gain Git authority.

---

### Requirement: Runtime Emits Distribution Observability

Runtime SHALL keep Component distribution observations structured and
non-authoritative when those observations are emitted.

Observability SHALL not alter validation or trust results.

#### Scenario: Fetch fails

Given a configured source cannot provide a package

When Runtime records the failure

Then the observation reports a stable distribution failure category

And execution does not proceed with missing or unvalidated bytes.
