## ADDED Requirements

### Requirement: Provider Resources May Back Model Instance

Runtime SHALL allow a Provider to own opaque resources backing a Model Instance.

Provider-owned resources SHALL remain internal to Runtime and Provider.

#### Scenario: Provider initialized model

Given Provider initializes native model resources

When Runtime exposes Model Instance status

Then it exposes stable metadata and not raw Provider handles.

---

### Requirement: Provider Status Affects Model Instance Readiness

Provider health, readiness, pressure, admission, and failure SHALL affect Model
Instance readiness and lifecycle.

#### Scenario: Provider not ready

Given Provider backing a Model Instance is not ready

When Runtime evaluates readiness

Then Model Instance is not ready or is draining/suspended according to policy.

---

### Requirement: Provider Failure Maps To Instance State

Provider failure during model use SHALL map to stable Model Instance lifecycle
or error state.

#### Scenario: Provider execution fails

Given Provider fails during generation

When Runtime processes the error

Then related Model Instance state is updated according to policy.
