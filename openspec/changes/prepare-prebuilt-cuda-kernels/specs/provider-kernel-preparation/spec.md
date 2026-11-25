## ADDED Requirements

### Requirement: Optional Provider Kernel Preparation Capability

A Provider MAY expose a kernel preparation capability that advertises the compiled artifact formats and targets it can load. A Provider that exposes none SHALL be treated as unable to prepare external artifacts. Preparation SHALL be separate from kernel compilation: it SHALL NOT compile source, and a Provider SHALL NOT be assumed able to prepare because it can compile or execute kernels.

#### Scenario: A Provider without the capability

- **WHEN** the Runtime holds an eligible `nvidia:cubin` bundle and the Provider returns no preparation capability
- **THEN** the bundle is not prepared, and the built-in kernels keep serving the operator

### Requirement: Preparation Input Is Admitted Artifact Data

The Runtime SHALL request preparation only for an artifact that the lot 0 rules make eligible (trusted, permitted, qualified, target-compatible, not revoked). The request SHALL carry the staged, verified bytes, the parsed launch descriptor, the semantic binding, the specialization and the Runtime-selected Device. The Provider SHALL NOT read any path, URL or source named by the manifest.

#### Scenario: An ineligible artifact

- **WHEN** an artifact is trusted but not permitted for the Provider
- **THEN** no preparation request is made

### Requirement: Preparation Result

A successful preparation SHALL return an opaque prepared kernel identity and a kernel advertisement whose kernel identity derives from the bundle identity and entry symbol and differs from every built-in kernel. A failed preparation SHALL leave no prepared state and SHALL report a structured error (incompatible target, missing symbol, invalid launch descriptor, resource limit, or Provider failure).

#### Scenario: A missing entry symbol

- **WHEN** the image does not export the descriptor's entry symbol
- **THEN** preparation fails with a missing-symbol error, and no candidate is registered

### Requirement: Preparation Is Cold Path

Preparation and release SHALL never run inside token dispatch or a step replay. A prepared kernel SHALL be ready before any selection can choose it.

#### Scenario: Selection before readiness

- **WHEN** a decode step runs while a preparation is in progress
- **THEN** the step uses the currently active kernel, and the preparation completes independently

### Requirement: Release Respects References

A Provider SHALL refuse to release a prepared kernel while dispatched work or a step capture references it, and SHALL report it in use.

#### Scenario: A capture references the kernel

- **WHEN** a decode capture recorded a launch of the prepared kernel and release is requested
- **THEN** release reports the kernel in use, and the module stays loaded
