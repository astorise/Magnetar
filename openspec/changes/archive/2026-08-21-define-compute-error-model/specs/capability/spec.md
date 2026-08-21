## ADDED Requirements

### Requirement: Stable Compute Errors

`magnetar:compute/run` SHALL expose stable structured compute errors.

Compute errors SHALL use stable error categories.

Backend-specific error strings SHALL NOT define the stable contract.

#### Scenario: Provider returns native error

Given a Provider returns a backend-specific error

When the Runtime reports the failure

Then the Runtime maps it to a stable Compute Error category

And backend-specific details are attached only as diagnostics.

---

### Requirement: Error Phase

Every Compute Error SHALL identify the phase in which the error occurred.

Error phases SHALL include:

- validation
- resolution
- affinity-validation
- submission
- execution
- cancellation
- completion
- interruption

#### Scenario: Report validation error

Given graph validation fails

When the Runtime returns an error

Then the error phase is `validation`.

---

### Requirement: Validation Errors

The Compute Error Model SHALL include validation errors.

Validation errors SHALL include:

- invalid tensor descriptor
- invalid shape
- invalid dtype
- invalid layout
- size overflow
- invalid graph
- cyclic graph
- missing input
- missing output

#### Scenario: Invalid graph

Given a Compute Graph is malformed

When the Runtime validates it

Then the Runtime rejects it with a structured validation error before Provider
execution begins.

---

### Requirement: Unsupported Feature Errors

The Compute Error Model SHALL include unsupported feature errors.

Unsupported feature errors SHALL include:

- unsupported operation
- unsupported operation family
- unsupported dtype
- unsupported layout
- unsupported data movement
- unsupported transfer
- unsupported materialization
- unsupported conversion

#### Scenario: Unsupported dtype

Given a Compute Graph requires a dtype unsupported by all compatible Providers

When the Runtime validates Provider compatibility

Then the Runtime returns a structured unsupported dtype error.

---

### Requirement: Resolution Errors

The Compute Error Model SHALL include resolution errors.

Resolution errors SHALL include:

- no compatible Provider
- policy rejected Provider
- Provider unavailable
- Device unavailable
- Capability version mismatch

#### Scenario: No compatible Provider

Given no Provider can satisfy a compute request

When the Runtime resolves the request

Then the Runtime returns a structured no-compatible-provider error.

---

### Requirement: Resource Affinity Errors

The Compute Error Model SHALL include Resource Affinity errors.

Resource Affinity errors SHALL include:

- incompatible resource affinity
- Provider-pinned resource
- Device-bound resource
- artifact fingerprint mismatch
- affinity group mismatch

#### Scenario: Incompatible tensor resource

Given a Tensor Resource is bound to one Provider

And a Compute Graph is resolved to another incompatible Provider

When the Runtime validates the graph

Then the Runtime rejects the submission with a structured resource affinity
error.

---

### Requirement: Execution Errors

The Compute Error Model SHALL include execution errors.

Execution errors SHALL include:

- execution failed
- execution interrupted
- execution cancelled
- operation timeout
- out of memory
- resource exhausted

#### Scenario: Provider execution failure

Given a Provider fails during compute execution

When the Runtime reports the failure

Then the Runtime returns a stable execution error category.

---

### Requirement: Cancellation Errors

Cancellation SHALL be represented as a terminal execution outcome.

Cancellation SHALL NOT be reported as an unknown execution failure.

#### Scenario: Operation cancelled

Given a Compute Submission is running

When cancellation succeeds

Then the submission reaches a cancelled terminal state.

---

### Requirement: Interruption Errors

Interruption SHALL be distinct from cancellation.

Interruption means execution could not continue because of Provider, Device,
resource or runtime failure.

#### Scenario: Provider interrupted

Given a Provider-pinned operation is running

When the owning Provider becomes unavailable

Then the Runtime reports an interruption instead of silently resolving another
Provider.

---

### Requirement: Recovery Hints

The Compute Error Model SHALL define stable Recovery Hint categories.

Compute Errors MAY include Recovery Hints.

Recovery Hints SHALL be advisory.

Recovery Hints SHALL NOT imply that the Runtime has already retried, migrated or
replayed execution.

Supported Recovery Hints SHALL include:

- not-retryable
- retry-before-state
- restartable-with-replay
- explicit-transfer-required
- explicit-materialization-required
- Provider-pinned

When present, Recovery Hints SHALL use the stable hint categories defined by
this model.

#### Scenario: Provider-pinned failure

Given a Provider-pinned generation or compute session fails after state creation

When the Runtime reports the error

Then the error may include a Provider-pinned recovery hint.

---

### Requirement: No Automatic Migration Claim

The Compute Error Model SHALL NOT claim automatic live state migration.

Errors MAY describe whether work is transparent, restartable or Provider-pinned.

The Runtime SHALL NOT report a migrated result unless a future migration
contract explicitly defines it.

#### Scenario: Live state failure

Given a Provider-pinned resource has observable state

When the owning Provider fails

Then the Runtime reports interruption or failure

And does not claim successful migration.

---

### Requirement: Diagnostic Payload

The Compute Error Model SHALL define stable diagnostic payload rules.

Compute Errors MAY include diagnostic payloads.

When present, diagnostic payloads SHALL use stable identifiers and redacted
debug strings.

Diagnostics MAY include:

- Provider identifier
- Device identifier
- Capability identifier
- operation family
- rejected candidate identifiers
- backend diagnostic message
- debug trace identifier

Diagnostics SHALL NOT expose:

- raw pointers
- GPU pointers
- backend storage objects
- queues
- streams
- locks
- native handles
- credentials
- ambient filesystem paths

#### Scenario: Inspect diagnostic error

Given a Compute Error contains diagnostics

When a Component inspects the error

Then it observes stable identifiers and redacted diagnostic strings only.

---

### Requirement: Stable Error Serialization

Compute Errors SHALL be serializable across WIT boundaries.

Error fields SHALL use portable value types.

Error fields SHALL NOT contain Rust trait objects, associated types, callbacks,
channels or platform-specific handles.

#### Scenario: Return error through WIT

Given a compute request fails

When the error crosses the WIT boundary

Then it is represented using stable portable values.

---

### Requirement: Error Compatibility

Future versions of the Compute Error Model SHALL preserve compatibility for
existing stable error categories.

New error categories MAY be added in compatible versions when callers can treat
them as a general compute error.

#### Scenario: Add new error category

Given a future Compute Error category is introduced

When an older caller receives it

Then the caller can still handle it through a stable generic error fallback.

