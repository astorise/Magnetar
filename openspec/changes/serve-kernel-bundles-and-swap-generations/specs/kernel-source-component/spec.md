## ADDED Requirements

### Requirement: Kernel Package World

The Runtime SHALL host Components of the `magnetar:kernel-source@0.1.0` world `kernel-package` as Kernel Artifact Sources. Such a Component SHALL export `source` (`list-bundles`, `read-chunk`) and SHALL import nothing. A Component whose linking requires any import SHALL be refused.

#### Scenario: A kernel-package Component that imports a Capability

- **WHEN** a kernel-package Component imports `graph-builder` or any WASI interface
- **THEN** instantiation fails at linking, and no read happens

### Requirement: Bounded Component Execution

The Runtime SHALL instantiate a kernel-package Component only after Component trust verification, with a linear-memory limit, a fuel budget per call and an epoch deadline per call. A trap, exhaustion or deadline SHALL fail only the current ingestion transaction.

#### Scenario: A Component that loops

- **WHEN** `read-chunk` never returns
- **THEN** the epoch deadline interrupts it, the transaction fails, and inference is unaffected

### Requirement: Embedded And Resolvable Bundles

A descriptor marked `embedded` SHALL be read through `read-chunk` only. A descriptor marked `resolvable` SHALL be resolved by the Runtime, by manifest digest, from an operator-configured source through its source authority, and only if such a source is configured. The Component SHALL supply no location or credential, and its descriptors SHALL remain claims bounded by the gateway.

#### Scenario: A resolvable bundle without a configured source

- **WHEN** a Component lists a `resolvable` bundle and no OCI source is configured
- **THEN** the bundle stays discovered and unavailable, and the Runtime makes no network request

#### Scenario: The same bundle embedded and from OCI

- **WHEN** one manifest digest arrives embedded from a Component and from an authorized OCI source
- **THEN** both resolve to one bundle identity with distinct observed sources

### Requirement: Component-Served Bytes Are Ordinary Bundles

Bytes from a kernel-package Component SHALL be a Kernel Exchange Bundle archive, validated by the existing extractor, path safety and manifest validation, and subject to lot 0 trust and execution permission like any other source.

#### Scenario: A corrupt embedded bundle

- **WHEN** a Component serves an archive whose blob does not match its digest
- **THEN** the bundle is rejected or quarantined by policy and never reaches a Provider
