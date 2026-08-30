## ADDED Requirements
### Requirement: Resource Residency Is Observable

Runtime SHALL expose redacted Resource residency diagnostics when diagnostics
are requested.

#### Scenario: KV on GPU

Given KV page is Device-resident

When diagnostics are requested

Then ResourceId, memory-domain class and Device stable identity may be shown.

### Requirement: Zero-Copy Decision Is Observable

Runtime SHALL record whether Resource binding used zero-copy or explicit
movement when residency diagnostics are enabled.

#### Scenario: Shared-memory input

Given no copy is required

When Plan executes

Then zero-copy-selected event may be emitted.

### Requirement: Transfer Elision Is Observable

Runtime SHALL report redundant movement elimination when residency diagnostics
are enabled.

#### Scenario: Tensor already on target Device

Given Plan initially considered transfer

When Runtime detects compatible residency

Then transfer-elided reason may be observed.

### Requirement: Mapping Lifecycle Is Observable

Logical mapping creation/release SHALL be traced when residency diagnostics are
enabled.

#### Scenario: Final output mapping

Given host mapping occurs

When diagnostics enabled

Then mapping lifetime can be observed without revealing pointer.

### Requirement: Native Memory State Is Redacted

Observability SHALL NOT expose native addresses or external-memory handles.

#### Scenario: CUDA allocation

Given Tensor has Device pointer

When trace is emitted

Then pointer value is absent.
