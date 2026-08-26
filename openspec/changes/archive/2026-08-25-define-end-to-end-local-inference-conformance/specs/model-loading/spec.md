## ADDED Requirements

### Requirement: E2E Uses Model Loading Contract

E2E conformance SHALL load models through Model Loading Contract.

#### Scenario: Load fixture model

Given E2E fixture model reference is valid

When loading begins

Then Runtime validates artifact, component, memory, provider, and policy before
ready instance publication.

---

### Requirement: E2E Validates Loading Failure

E2E conformance SHALL validate structured loading failure paths.

#### Scenario: Untrusted fixture

Given fixture artifact trust state is invalid

When loading runs

Then Runtime returns structured model loading failure.