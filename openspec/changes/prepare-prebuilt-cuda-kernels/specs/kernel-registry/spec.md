## ADDED Requirements

### Requirement: Candidates From Provider-Prepared Artifacts

The Runtime SHALL register the advertisement of a prepared external kernel under Provider authority, bound to its bundle identity and blob digest as a prepared generation, beside the Provider's built-in kernels for the same operator. It SHALL rank such candidates by the normal selection policy after eligibility filtering.

#### Scenario: Two candidates for one operator

- **WHEN** a prepared `silu-mul` and the built-in `silu-mul` are both eligible
- **THEN** selection ranks them by policy, and the selection explanation names the bundle identity of the external one

### Requirement: Explicit Pin For Experiments

The Runtime SHALL accept, from its policy configuration only, an explicit pin of an operator to an eligible prepared candidate. It SHALL record the pin in dispatch observations and SHALL refuse a pin to an ineligible candidate rather than fall back silently.

#### Scenario: A pin to a revoked artifact

- **WHEN** the configured pin names a revoked bundle
- **THEN** the pin is refused with a structured error, and selection proceeds without it
