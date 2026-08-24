## ADDED Requirements

### Requirement: Provider Participates In Model Materialization

A Provider SHALL participate in model materialization only through Runtime-controlled
loading phases.

Provider participation SHALL remain behind Runtime policy and Provider
Capability validation.

#### Scenario: Provider-specific transform

Given a Provider supports a specific quantized layout

When Runtime loads compatible weights

Then Provider may materialize Provider-owned model resources through the Runtime
loading path.

---

### Requirement: Provider Model Resources Are Opaque

Provider-owned model resources SHALL remain opaque to Components and public
portable APIs.

#### Scenario: Provider-loaded weights

Given Provider materializes weights into native memory

When Runtime exposes model status

Then it exposes stable Runtime residency metadata

And not raw Provider handles.

---

### Requirement: Provider Loading Failure Maps To Model Loading Error

Provider failures during model materialization SHALL map to stable Model Loading
errors.

#### Scenario: Provider initialization fails

Given Provider fails to initialize model resources

When loading reports failure

Then Runtime returns provider-initialization-failed or materialization-failed.
