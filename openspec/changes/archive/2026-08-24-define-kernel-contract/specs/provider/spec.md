## ADDED Requirements
### Requirement: Provider Advertises Kernels

Providers SHALL advertise supported Kernels through Runtime-readable metadata.

#### Scenario: Provider startup

Given a Provider registers with Runtime

When Runtime reads its capabilities

Then Kernel advertisements are available where supported.

---

### Requirement: Provider Owns Kernel Implementations

Provider SHALL own native Kernel implementations and execution boundary.

Kernels SHALL not be independently registered as Providers.

#### Scenario: Kernel registration

Given a Provider exposes ten Kernels

When Runtime registers them

Then one Provider remains registered with ten Kernel advertisements.

---

### Requirement: Provider Executes Runtime-Created Kernel Invocations

Providers SHALL execute only Runtime-created Kernel Invocations.

#### Scenario: Execute kernel

Given Runtime dispatches a validated Kernel Invocation

When Provider executes it

Then Provider uses validated Runtime resource references and metadata.

---

### Requirement: Provider Kernel Failures Are Structured

Provider failures during Kernel execution SHALL map to stable Kernel errors.

#### Scenario: Provider native failure

Given native Kernel execution fails

When Runtime receives the failure

Then it maps to kernel-execution-failed or a more specific Kernel error.

---

### Requirement: Provider Kernel Advertisement Participates In Conformance

Advertised Kernels SHALL be eligible for Kernel conformance profiles.

#### Scenario: Advertised matmul kernel

Given Provider advertises a matmul Kernel

When conformance runs

Then the Kernel is tested against matmul Operator semantics and metadata.