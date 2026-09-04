## ADDED Requirements

### Requirement: Tensor Content Digest Binding

When a tensor's inventory entry declares a content digest, that digest SHALL identify the specific bytes that count as that tensor's content for the artifact it belongs to.

A Model Artifact's tensor inventory MAY declare a per-tensor content digest; declaring one is optional per tensor. A tensor entry that declares no digest is not covered by this requirement; whether its content is verified is left to whatever mechanism does declare a digest for it, if any.

#### Scenario: Declared digest identifies real content

Given a tensor inventory entry declares a content digest

When the artifact's real bytes for that tensor are hashed with the digest's algorithm

Then the computed digest matches the declared one

#### Scenario: Tensor without a declared digest is unconstrained by this requirement

Given a tensor inventory entry declares no content digest

When that tensor's content is later supplied for materialization

Then this requirement imposes no constraint on it
