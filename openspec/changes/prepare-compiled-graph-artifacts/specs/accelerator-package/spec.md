## ADDED Requirements

### Requirement: Accelerator Package Envelope

The Runtime SHALL accept `magnetar:accelerator-package@1` envelopes as canonical JSON, with the duplicate-key, nesting and size limits of manifest v1 and a digest over the canonical bytes. An envelope SHALL group variants of one semantic binding, each with a kind (`operator-kernel`, `fused-kernel`, `compiled-graph`), a Provider compatibility, a format, requirements and an artifact reference.

#### Scenario: Duplicate keys

- **WHEN** an envelope contains a duplicate object key
- **THEN** it is rejected before any reference is resolved

### Requirement: The Envelope Grants Nothing

The Runtime SHALL resolve an operator or fused kernel variant to a Kernel Exchange Bundle by manifest digest and take it through lots 0 to 2 unchanged. An envelope's signature or requirements SHALL NOT grant trust, execution permission, qualification or selection to any variant.

#### Scenario: A signed envelope referencing an unsigned bundle

- **WHEN** a trusted envelope references a bundle without a kernel signature, under the production policy
- **THEN** that variant is untrusted, and the other variants are evaluated independently

### Requirement: Kinds Are Not Interchangeable

The Runtime SHALL never register a `compiled-graph` variant as an operator kernel candidate, and SHALL never substitute an operator kernel variant for a graph segment.

#### Scenario: A compiled graph offered for an operator

- **WHEN** a `compiled-graph` variant names an operator semantic binding
- **THEN** it is considered only for segment or instance substitution, never by kernel selection
