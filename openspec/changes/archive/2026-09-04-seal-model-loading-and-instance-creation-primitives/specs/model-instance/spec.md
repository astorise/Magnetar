## MODIFIED Requirements

### Requirement: Model Instance References Architecture Implementation

A Model Instance SHALL be able to reference the Model Component or Runtime-native
architecture implementation used to create it, and the referenced architecture and Resource Affinity SHALL be consistent with the values the loading phase already resolved for that artifact wherever the loading phase resolved a value.

Model Instance creation SHALL reject an architecture implementation whose architecture identity disagrees with the loaded artifact's resolved architecture, and SHALL reject a Resource Affinity whose provider or device disagrees with a provider or device the loading phase already resolved.

An architecture implementation kind or required capability the loading phase did not resolve is a legitimate choice for the caller creating the Model Instance, since the loading phase never resolves a value for either. A Resource Affinity's provider or device left unresolved by the loading phase is a known, documented limitation, not an intentional grant of caller authority: today's implementation applies the caller-supplied value directly as the instance's effective placement, with no Runtime-side arbitration step at instance-creation time, which does not yet fully satisfy `inference-api`'s "Provider Preferences Are Non-Authoritative" requirement for this specific case.

#### Scenario: Instance compatibility

Given a Model Instance was created with Model Component C

When cache compatibility is evaluated

Then C's identity and version may be considered.

---

#### Scenario: Architecture disagreeing with the resolved load is rejected

Given a Model Artifact was loaded and its loading phase resolved a specific architecture identity

When Model Instance creation is attempted with a different architecture identity

Then Runtime rejects the creation rather than silently accepting the caller's value

---

#### Scenario: Affinity disagreeing with a resolved provider or device binding is rejected

Given a Model Artifact's loading phase resolved a specific provider or device binding

When Model Instance creation is attempted with a Resource Affinity naming a different provider or device

Then Runtime rejects the creation rather than silently accepting the caller's value

---

#### Scenario: Unresolved provider or device becomes the caller's value directly

Given a Model Artifact's loading phase did not resolve a provider or device binding

When Model Instance creation supplies a Resource Affinity naming a provider or device

Then Runtime accepts it and that value becomes the instance's effective placement directly, without Runtime-side arbitration

---

## ADDED Requirements

### Requirement: Model Instance Creation Is Reachable Only Through Runtime

`ModelInstanceDefinition::from_loaded_context` and the operation that registers a definition as a new Model Instance SHALL NOT be reachable by a caller outside the Runtime-owned Model Instance creation entrypoint, so that the architecture and Resource Affinity cross-checks that entrypoint performs cannot be bypassed by constructing and registering a definition directly.

#### Scenario: No external construct-and-register path

Given a caller outside the Runtime-owned Model Instance creation entrypoint

When that caller attempts to construct a Model Instance Definition from a loaded context and register it as a new Model Instance

Then no such path is reachable outside the crate implementing Runtime
