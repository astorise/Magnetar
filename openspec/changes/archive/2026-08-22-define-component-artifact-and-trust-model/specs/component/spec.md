## ADDED Requirements

### Requirement: Component Artifact

A Component Artifact SHALL represent executable WebAssembly Component code.

A Component Artifact SHALL be distinct from a Model Artifact, Provider binary,
Runtime module, and trust policy.

#### Scenario: Classify artifacts

Given a `.wasm` Component and a model weights file

When Magnetar classifies artifacts

Then the `.wasm` file is a Component Artifact

And the weights file is a Model Artifact.

---

### Requirement: Component Artifact Identity

Every Component Artifact SHALL have a Runtime-recognized identity.

The identity SHALL include at least:

- artifact kind
- digest algorithm
- content digest
- logical Component name
- Component version
- manifest version

#### Scenario: Identify Component bytes

Given a local Component artifact

When Magnetar evaluates it

Then its executable bytes are identified by computed digest

And not merely by filename.

---

### Requirement: Content Digest

A Component Artifact SHALL be identified by a content digest.

The digest algorithm SHALL be explicit.

The initial digest algorithm SHOULD be `sha256`.

#### Scenario: Same name different bytes

Given two artifacts have the same logical name and version

But different executable bytes

When Magnetar computes their digests

Then they are different Component Artifact identities.

---

### Requirement: Component Manifest

A Component Artifact SHALL have a manifest describing its declared metadata.

The manifest SHALL include required fields for identity, compatibility, WIT
contracts, and authority declarations.

The initial manifest SHALL use `schema: magnetar-component-artifact` and
`schema_version: 1`.

The initial manifest SHALL include `artifact.kind: component`, a canonical
`artifact.digest`, `component` metadata, `runtime.magnetar.min_version`, `wit`
imports and exports, `capabilities.requires`, `authority.requires`, `source`,
and `signatures`.

#### Scenario: Load manifest

Given a Component artifact has a sidecar manifest

When Magnetar loads the artifact

Then the manifest is parsed and validated before execution preparation.

#### Scenario: Manifest schema version

Given a Component artifact manifest uses schema `magnetar-component-artifact`

And schema version `1`

When Magnetar validates the manifest

Then the manifest is accepted as the initial Component Artifact manifest format.

---

### Requirement: Manifest Is Not Trust

A Component manifest SHALL NOT be treated as proof of trust.

A Component Artifact SHALL NOT be trusted merely because its manifest says it is
trusted.

#### Scenario: Manifest claims trust

Given a manifest contains a field or text claiming the artifact is trusted

When Magnetar evaluates the artifact

Then trust is determined by Runtime trust policy

And not by the artifact's own claim.

---

### Requirement: Manifest Digest Match

A manifest SHALL identify the executable artifact digest.

The Runtime SHALL reject a Component Artifact when the manifest digest does not
match the computed digest.

#### Scenario: Digest mismatch

Given a Component artifact's manifest declares digest A

And the Runtime computes digest B

When A and B differ

Then the artifact is rejected before preparation.

---

### Requirement: WIT Manifest Consistency

Declared WIT imports and exports in the manifest SHALL be consistent with the
actual executable Component.

#### Scenario: Manifest omits import

Given a Component executable imports interface X

And its manifest does not declare X

When artifact validation runs

Then validation fails.

#### Scenario: Manifest claims nonexistent export

Given a manifest declares export Y

And the executable Component does not export Y

When artifact validation runs

Then validation fails.

---

### Requirement: Runtime Compatibility Declaration

A Component manifest SHALL declare Runtime compatibility where required.

The Runtime SHALL reject artifacts that declare incompatible Runtime
requirements.

#### Scenario: Future Runtime required

Given a Component requires a future Magnetar Runtime version

When the current Runtime cannot satisfy that requirement

Then the artifact is rejected before preparation.

---

### Requirement: Capability Compatibility Declaration

A Component manifest SHALL declare required Magnetar Capabilities.

Capability compatibility SHALL be evaluated before instantiation.

#### Scenario: Unsupported Capability major version

Given a Component requires a Capability major version unsupported by the Runtime

When validation runs

Then the artifact is rejected.

---

### Requirement: Authority Requirement Declaration

A Component manifest SHALL declare requested authority.

Authority declarations MAY include:

- filesystem
- network
- environment
- process execution
- secrets
- clock
- randomness
- source-control access
- tool access
- external services

This change defines declaration and validation only.

Authority granting and scoping require a dedicated authority-scoping model.

#### Scenario: Filesystem authority requested

Given a Component declares filesystem authority

When artifact validation runs

Then the declaration is recorded and syntax-validated

But filesystem access is not granted merely by declaration.

---

### Requirement: Unsupported Authority Fails Closed

Unsupported or unknown authority declarations SHALL fail closed unless Runtime
policy explicitly allows unknown declarations as non-executable metadata.

#### Scenario: Unknown authority

Given a Component manifest declares an unknown authority kind

When validation runs

Then the artifact is rejected or marked not executable according to policy.

---

# Trust

### Requirement: Component Trust Status

The Runtime SHALL represent Component Artifact trust status explicitly.

Trust status SHALL include at least:

- unknown
- trusted
- rejected
- quarantined
- revoked

Only trusted Component Artifacts MAY be prepared for execution.

#### Scenario: Unknown artifact

Given a valid Component artifact with no matching trust policy

When artifact validation completes

Then the artifact remains unknown or rejected according to policy

And is not prepared as trusted executable code.

---

### Requirement: Trust Policy Determines Executability

A Component Artifact SHALL be executable only when Runtime trust policy permits
it.

Trust policy MAY evaluate:

- digest
- source
- publisher
- signature metadata
- revocation
- local administrator decision

#### Scenario: Digest allowlist

Given a Component artifact digest is present in the trust allowlist

And no rejection or revocation rule applies

When validation succeeds

Then the artifact may be marked trusted.

---

### Requirement: Rejection Overrides Trust

Rejected or revoked artifact status SHALL override allowlist or publisher trust
unless policy explicitly defines a stronger administrative override.

#### Scenario: Digest both allowed and revoked

Given an artifact digest appears in both allowlist and revoked list

When trust is evaluated

Then revoked status wins

And the artifact is not executable.

---

### Requirement: Publisher Metadata Is Not Sufficient Trust

Publisher identity SHALL be metadata.

It SHALL NOT imply trust unless Runtime policy explicitly trusts that publisher
and all other validation succeeds.

#### Scenario: Known publisher

Given a manifest declares a known publisher

But Runtime policy does not trust that publisher

When trust is evaluated

Then the artifact is not trusted solely because of the publisher field.

---

### Requirement: Source Metadata Is Not Sufficient Trust

Source identity SHALL describe where the artifact came from.

It SHALL NOT imply trust unless Runtime policy explicitly trusts that source and
all other validation succeeds.

#### Scenario: Local file source

Given a Component artifact is loaded from a local directory

When trust is evaluated

Then local presence alone does not make it trusted.

---

### Requirement: Signature Metadata Is Optional and Non-Authoritative

Signature metadata SHALL be optional and non-authoritative.

Signature metadata MAY be present.

An unsupported or unverified signature SHALL NOT make an artifact trusted.

#### Scenario: Signature present but unsupported

Given a manifest contains signature metadata

And the Runtime has no configured verifier for that signature

When trust is evaluated

Then the signature is recorded as unverified

And does not by itself grant trust.

---

### Requirement: Revocation

The Runtime SHALL support revoking Component Artifacts by digest.

A revoked artifact SHALL NOT be prepared or instantiated.

#### Scenario: Previously trusted artifact revoked

Given an artifact digest was previously trusted

And the digest is later revoked

When new preparation is requested

Then preparation is denied.

---

### Requirement: Quarantine

The Runtime SHALL treat quarantined Component Artifacts as non-executable.

The Runtime MAY quarantine invalid or suspicious Component Artifacts.

A quarantined artifact SHALL NOT be prepared or instantiated.

#### Scenario: Suspicious artifact

Given an artifact fails a trust or integrity check

When policy chooses quarantine

Then diagnostic metadata may be retained

But executable preparation is prohibited.

---

### Requirement: Development Mode

Development mode SHALL be explicit when enabled.

Development mode MAY allow unsigned local Component Artifacts.

Development mode SHALL be explicit.

Development mode SHALL still validate digest, manifest structure, WIT
consistency, and compatibility.

#### Scenario: Local development Component

Given development mode is enabled

And a local unsigned Component has a valid manifest and digest

When trust policy evaluates it

Then it may be accepted according to development policy.

---

### Requirement: File-Based Trust Store

The initial Runtime trust store SHALL support YAML with
`schema: magnetar-component-trust` and `schema_version: 1`.

The trust store SHALL support `trusted_digests`, `rejected_digests`,
`revoked_digests`, `trusted_publishers`, `trusted_sources`, and
`development.allow_unsigned_local`.

The trust store SHALL remain separate from the Component Artifact manifest.

#### Scenario: Minimal trust store

Given a trust store lists a digest under `trusted_digests`

And the same digest is not rejected or revoked

When a matching artifact passes manifest, digest, WIT, compatibility, and
authority validation

Then policy may mark the artifact trusted.

#### Scenario: Manifest cannot self-trust

Given a manifest includes publisher metadata or trust-like text

When the trust store has no matching trust rule

Then the artifact is not trusted.

---

# Lifecycle

### Requirement: Artifact Validation Before Preparation

A Component Artifact SHALL be validated and trusted before ComponentEngine
preparation.

Validation SHALL follow this order: compute digest, load manifest, validate
manifest schema, compare digest, inspect actual WIT imports and exports, compare
manifest WIT to actual WIT, check Runtime compatibility, check Capability
compatibility, validate authority declarations, evaluate trust policy, and only
then produce a trusted Component Artifact for `ComponentEngine::prepare`.

#### Scenario: Prepare Component

Given a local `.wasm` file has not been validated

When preparation is requested

Then Runtime first performs artifact validation and trust evaluation.

---

### Requirement: Artifact State Is Distinct from Prepared State

A trusted Component Artifact SHALL NOT automatically be a Prepared Component.

#### Scenario: Trust succeeds but compilation fails

Given artifact validation and trust evaluation succeed

But the ComponentEngine cannot prepare the artifact

When preparation runs

Then the artifact remains trusted

And preparation fails with a separate Component preparation error.

---

### Requirement: Artifact State Is Distinct from Instance State

Trusting or preparing a Component Artifact SHALL NOT automatically instantiate a
Component.

#### Scenario: Trust artifact

Given a Component Artifact is marked trusted

When no instantiation is requested

Then no Component Instance is created.

---

### Requirement: Artifact Digest Attached to Component Definition

A Component Definition created from a Component Artifact SHALL retain the
artifact digest.

#### Scenario: Inspect Component definition

Given a Component Definition was created from a trusted artifact

When Runtime observability or diagnostics refer to it

Then the artifact digest can be included as identity metadata.

---

# Tachyon Boundary

### Requirement: Vendor-Neutral Component Source

The Component Artifact model SHALL be independent from any one distribution
source.

Tachyon MAY be a future source of Component Artifacts, but Magnetar SHALL NOT
require Tachyon to validate or execute local Components.

#### Scenario: Local Component install

Given a Component artifact is installed locally without Tachyon

When Magnetar validates and trusts it

Then it can be prepared according to the same artifact model.

---

### Requirement: Tachyon Distribution Does Not Imply Trust

If Tachyon supplies a Component Artifact, Magnetar SHALL still validate the
artifact locally.

#### Scenario: Tachyon-provided artifact

Given Tachyon provides a Component Artifact

When Magnetar receives it

Then Magnetar computes digest, validates manifest consistency, evaluates
compatibility, and applies trust policy before execution.

---

# Observability

### Requirement: Component Artifact Observability

The Runtime SHALL keep Component Artifact observations non-authoritative.

The Runtime SHOULD emit structured observations for Component Artifact
validation and trust decisions.

Observations MAY include:

- artifact source
- digest algorithm
- digest
- manifest validation result
- WIT consistency result
- compatibility result
- trust decision
- revocation
- quarantine

#### Scenario: Artifact rejected

Given a Component Artifact fails digest validation

When Runtime observability records the event

Then the observation identifies the stable failure category

And does not expose secrets or private signature material.
