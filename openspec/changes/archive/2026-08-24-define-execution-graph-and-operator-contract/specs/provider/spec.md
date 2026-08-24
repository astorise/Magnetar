## ADDED Requirements
### Requirement: Providers Advertise Operator Capabilities

Providers SHALL advertise supported operator families, dtypes, layouts, shape
constraints, memory behavior, and execution constraints.

#### Scenario: Provider supports attention

Given a Provider supports attention

When Runtime reads Provider advertisements

Then supported attention metadata is available for planning.

---

### Requirement: Provider Executes Operators Through Runtime Dispatch

Providers SHALL execute operators or graph fragments through Runtime-managed
dispatch.

#### Scenario: Execute matmul

Given Runtime dispatches a matmul operator

When Provider executes it

Then Provider receives Runtime-validated operator metadata and resources.

---

### Requirement: Provider Operator Failures Are Structured

Provider failures during operator execution SHALL map to stable Runtime operator
or graph errors.

#### Scenario: Provider dtype unsupported

Given Provider rejects an operator dtype

When Runtime reports the failure

Then it maps to dtype-unsupported or Provider-capability-unavailable.
