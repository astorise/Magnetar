## ADDED Requirements

### Requirement: Compiled Graph Binding

A compiled graph variant SHALL declare its granularity (segment or instance), its segment signature for a segment, its model binding (the Model Artifact digest or the digests of the weights it embeds), its exact toolchain versions (SDK, compiler, driver, runtime), its accelerator generation and its declared memory. A variant missing any of them SHALL be ineligible.

#### Scenario: No model binding

- **WHEN** a compiled graph declares no model or weight digests
- **THEN** it is ineligible for every Model Instance

### Requirement: Exact Substitution

The Runtime SHALL substitute a compiled graph only for a contiguous segment of a prepared execution graph whose operators, attributes and boundary tensors equal the signature and whose weights match the model binding, or for a whole Model Instance of the bound Model Artifact. The Provider's reported toolchain and accelerator SHALL equal the declared ones. Any mismatch SHALL be a structured incompatibility, never a partial or approximate substitution.

#### Scenario: A different weight digest

- **WHEN** the Model Instance's weights differ from the compiled graph's bound digests
- **THEN** the compiled graph is not used, and the plan keeps its operator kernels

#### Scenario: A newer driver

- **WHEN** the Provider reports a driver version other than the declared one
- **THEN** the variant is ineligible on that Device

### Requirement: Compiled Graph Memory Admission

The Memory Manager SHALL admit a compiled graph's declared memory before the Provider prepares it. The Runtime SHALL reconcile the declaration with the Provider's reported usage after preparation and fail preparation when the report exceeds the declaration.

#### Scenario: An under-declared graph

- **WHEN** a prepared compiled graph reports more device memory than it declared
- **THEN** preparation fails, its instance is released, and the plan keeps its operator kernels

### Requirement: Compiled Graph Swap Through New Instances

The Runtime SHALL swap a compiled graph by preparing a new instance and a new plan revision, with lot 2 generation leases. The old instance SHALL drain before release, and a running plan SHALL never be patched.

#### Scenario: A new compiled graph version

- **WHEN** version 2 of a compiled graph is promoted while requests run on version 1
- **THEN** those requests finish on version 1, and version 1 is released after it drains
