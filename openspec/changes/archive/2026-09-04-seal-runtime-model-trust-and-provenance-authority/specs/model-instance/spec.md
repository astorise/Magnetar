## MODIFIED Requirements

### Requirement: Model Instance References Architecture Implementation

A Model Instance SHALL be able to reference the Model Component or Runtime-native
architecture implementation used to create it, and the referenced architecture and Resource Affinity SHALL be consistent with the values the loading phase already resolved for that artifact wherever the loading phase resolved a value.

Model Instance creation SHALL reject an architecture implementation whose architecture identity disagrees with the loaded artifact's resolved architecture, and SHALL reject a Resource Affinity whose provider or device disagrees with a provider or device the loading phase already resolved.

An architecture implementation kind, required capability, or Resource Affinity field the loading phase did not resolve is a legitimate choice for the caller creating the Model Instance and SHALL NOT be constrained by this requirement.

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

#### Scenario: Unresolved plan fields impose no constraint

Given a Model Artifact's loading phase did not resolve a provider or device binding

When Model Instance creation supplies a Resource Affinity naming a provider or device

Then Runtime accepts it, since no resolved value exists to disagree with

---

## ADDED Requirements

### Requirement: Materialized Weight Content Matches Its Declared Shape And Storage Dtype

Materialization evidence for a weight tensor SHALL only be produced for content whose shape matches the artifact's declared shape for that tensor and whose declared storage dtype is one the Runtime can legitimately materialize as the supplied content's representation.

This applies independently of, and precedes, digest verification: a tensor whose inventory entry declares no content digest -- for example because its declared storage dtype cannot yet be digested -- is not thereby exempt from shape and dtype agreement. A caller cannot bypass a format parser's correct refusal to materialize a non-materializable tensor by supplying self-constructed content directly to the materialization transaction under that tensor's name.

#### Scenario: Matching shape and dtype is evidenced normally

Given a Model Instance's loaded artifact declares a tensor's shape and a materializable storage dtype

When the authorized materialization transaction stages content with that shape

Then materialization evidence is minted for that tensor as usual

#### Scenario: Shape mismatch is rejected regardless of digest presence

Given a Model Instance's loaded artifact declares a specific shape for a tensor

When a caller attempts to materialize content with a different shape under that tensor's name

Then no materialization evidence is minted for that tensor, whether or not the tensor's inventory entry declares a content digest

#### Scenario: Declared non-materializable dtype is rejected even with well-formed content

Given a Model Instance's loaded artifact declares a tensor's storage dtype as one the Runtime cannot materialize into the supplied content's representation

When a caller attempts to materialize well-formed content under that tensor's name

Then no materialization evidence is minted for that tensor
