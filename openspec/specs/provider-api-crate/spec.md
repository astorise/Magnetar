# provider-api-crate Specification

## Purpose
This specification defines the Provider API crate: the small crate that holds the types a Provider implements or calls, so that Providers depend on a stable contract instead of on Runtime orchestration.
## Requirements
### Requirement: Provider API Crate

The Rust types that a Provider implements or calls SHALL be defined in a Provider API crate, separate from Runtime orchestration. The crate SHALL contain the Provider and Provider execution traits, Provider metadata and registration, the execution request, handle, result and status types, the tensor, packed weight, tensor-parallel and layer placement types, and the operator semantics and quantization tables that Providers reuse.

#### Scenario: Provider builds against the API crate

- **WHEN** a Provider is built with the Provider API crate as its only Magnetar dependency outside tests
- **THEN** it compiles, and every Runtime item its non-test code names resolves through that crate or a lower-layer crate

#### Scenario: Conformance tests

- **WHEN** a Provider runs the Provider conformance suite or compares against the Reference CPU Provider
- **THEN** it depends on the Runtime crate for tests only

### Requirement: Provider API Crate Dependencies

The Provider API crate SHALL depend only on lower-layer Runtime crates. It SHALL NOT depend on Runtime orchestration, on the Component engine, or on a Component, Provider, Format or Loader crate. It SHALL NOT access the filesystem, the network, secrets or processes.

#### Scenario: No orchestration dependency

- **WHEN** the dependency graph of the Provider API crate is resolved
- **THEN** it contains neither the Runtime orchestration crate nor the Component engine crate

#### Scenario: Provider loading stays in the Runtime

- **WHEN** the items that load Provider libraries from the filesystem are located
- **THEN** they are defined in the Runtime orchestration crate, not in the Provider API crate

### Requirement: Runtime Changes Do Not Reach Providers

A change confined to Runtime orchestration, the Component engine or the generation engine SHALL NOT require a change to a Provider repository that depends on the Provider API crate, including its lockfile.

#### Scenario: Generation engine change

- **WHEN** a dependency is added to the Runtime orchestration crate, or a module of the generation engine changes
- **THEN** the lockfile of a Provider that depends on the Provider API crate is unchanged, and the Provider is not rebuilt for it

### Requirement: Provider Contract Unchanged

Moving the Provider contract to the Provider API crate SHALL NOT change it. Trait methods, signatures, default implementations, the Provider API version and the Provider ABI version SHALL be identical before and after. Providers SHALL keep implementing the same Capabilities and exposing the same Devices, with the same Resource Affinity behaviour.

#### Scenario: Provider built before the move

- **WHEN** a Provider that depends on the Runtime crate, unchanged, is built against the Runtime after the move
- **THEN** it compiles and its tests pass

#### Scenario: Provider and Runtime agree on the contract

- **WHEN** a Provider built against the Provider API crate is registered with a Runtime
- **THEN** the Runtime accepts it through the same Provider trait, because the trait is defined once and re-exported

### Requirement: Runtime Paths Preserved

Every public path of the Runtime crate that resolved before the move SHALL resolve after it to the same item. No public item SHALL be removed or renamed.

#### Scenario: Existing consumers

- **WHEN** the CLI, the integration test crates, and the Format and Loader repositories are built against the Runtime after the move
- **THEN** they compile and their tests pass without a source change

#### Scenario: Sealed items

- **WHEN** a crate-private item is used across the new crate boundary
- **THEN** it becomes public only if it carries no authority, and is hidden from the generated documentation
