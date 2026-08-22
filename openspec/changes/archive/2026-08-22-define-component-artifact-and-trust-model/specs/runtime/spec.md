## ADDED Requirements

### Requirement: Runtime Validates Component Artifacts Before Execution

The Runtime SHALL validate a Component Artifact before preparing, instantiating,
or invoking it.

Validation SHALL include digest, manifest, WIT consistency, compatibility, and
trust policy.

#### Scenario: Unvalidated local WASM

Given a local `.wasm` file exists

When Runtime execution is requested

Then the Runtime does not prepare it until artifact validation succeeds.

---

### Requirement: Runtime Computes Artifact Digest

The Runtime SHALL compute the digest of executable Component Artifact content.

The computed digest SHALL be compared to the declared manifest digest.

#### Scenario: Modified bytes

Given Component bytes are modified after the manifest was created

When Runtime computes the digest

Then digest comparison fails

And the artifact is rejected.

---

### Requirement: Runtime Trust Policy Is External to Artifact

Runtime trust policy SHALL be separate from the Component Artifact manifest.

A Component Artifact SHALL NOT grant trust to itself.

#### Scenario: Artifact manifest says trusted

Given the manifest contains text claiming trust

When Runtime evaluates trust

Then the Runtime ignores that claim as authority

And uses configured trust policy.

---

### Requirement: Runtime Trust Store

The Runtime SHALL consume a trust store or equivalent policy source for
Component Artifact trust decisions.

The initial implementation MAY use a local file-based trust store.

#### Scenario: Trusted digest configured

Given a digest is listed as trusted in the Runtime trust store

When a matching artifact validates successfully

Then the Runtime may mark it trusted.

---

### Requirement: Runtime Denies Unknown Artifacts by Default

The Runtime SHALL deny unknown Component Artifacts by default.

Unless explicit development or permissive policy is configured, unknown
Component Artifacts SHALL NOT be prepared for execution.

#### Scenario: Valid but unknown artifact

Given a Component artifact has a valid manifest and digest

But no trust policy permits it

When preparation is requested

Then the Runtime denies preparation.

---

### Requirement: Runtime Revocation Enforcement

The Runtime SHALL prevent new preparation or instantiation of revoked Component
Artifacts.

#### Scenario: Revoked artifact requested

Given an artifact digest is revoked

When preparation is requested

Then the Runtime rejects the artifact before ComponentEngine receives it.

---

### Requirement: Runtime Quarantine Enforcement

Quarantined Component Artifacts SHALL remain non-executable.

#### Scenario: Quarantined artifact requested

Given an artifact is quarantined

When instantiation is requested

Then the Runtime denies execution.

---

### Requirement: Runtime Development Mode Is Explicit

Development mode SHALL be explicit configuration.

Development mode SHALL not disable digest, manifest, WIT, or compatibility
validation.

#### Scenario: Developer runs local fixture

Given development mode is enabled

When a local unsigned Component is loaded

Then the Runtime may accept it according to development policy

But still validates digest, manifest, WIT, and compatibility.

---

### Requirement: Runtime Separates Artifact Validation from Engine Preparation

Artifact validation SHALL complete before ComponentEngine preparation.

ComponentEngine SHALL not be used as the sole artifact validation mechanism.

#### Scenario: Engine could compile untrusted bytes

Given ComponentEngine can compile a local `.wasm`

But trust policy rejects the artifact

When Runtime handles the artifact

Then preparation is denied before engine compilation.

---

### Requirement: Runtime Attaches Artifact Identity to Component Definitions

When Runtime creates a Component Definition from a trusted artifact, it SHALL
attach artifact identity metadata.

#### Scenario: Prepared Component observed

Given a Component Definition is created from digest D

When observability records preparation

Then digest D can be associated with the definition.

---

### Requirement: Runtime Does Not Confuse Component and Model Artifacts

The Runtime SHALL keep Component Artifact identity separate from Model Artifact
identity.

#### Scenario: Model weights referenced by Component

Given a Component declares that it needs model weights

When Runtime evaluates the Component Artifact

Then the Component executable digest and model artifact identity remain
separate.

---

### Requirement: Runtime Does Not Require Tachyon

The Component Artifact trust model SHALL work for local artifacts without
Tachyon.

#### Scenario: Standalone Magnetar

Given Tachyon is unavailable

When a local trusted Component Artifact is provided

Then Magnetar can validate and prepare it according to local policy.

---

### Requirement: Runtime Validates Tachyon-Provided Artifacts Locally

Runtime SHALL validate Tachyon-provided Component Artifacts locally.

If an external system such as Tachyon provides a Component Artifact, Runtime
SHALL still validate it locally.

#### Scenario: External source artifact

Given an external source provides artifact bytes and metadata

When Runtime receives them

Then Runtime computes digest and evaluates local trust policy before execution.

---

### Requirement: Runtime Emits Trust Observability

Runtime SHALL keep trust observability non-authoritative.

Runtime SHOULD emit observations for Component Artifact validation and trust
decisions.

Observability SHALL not alter the trust result.

#### Scenario: Observability sink fails

Given a trust decision is made

And observability delivery fails

When Runtime continues

Then the trust decision remains unchanged.
