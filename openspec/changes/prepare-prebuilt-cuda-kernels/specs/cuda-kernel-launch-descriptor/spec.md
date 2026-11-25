## ADDED Requirements

### Requirement: CUDA Launch Descriptor Extension

A compiled CUDA operator kernel artifact SHALL carry a required manifest extension `magnetar:cuda-launch@1` declaring its entry symbol, its ordered parameters, its block size, its grid and its dynamic shared memory. A compiled CUDA artifact without it SHALL be unsupported for preparation.

#### Scenario: No descriptor

- **WHEN** an `nvidia:cubin` artifact has no `magnetar:cuda-launch@1` extension
- **THEN** the manifest validates as a bundle, but the artifact is not offered for preparation

### Requirement: Declarative Parameters And Geometry

Launch parameters SHALL be drawn from a closed vocabulary: an input or output tensor pointer, an integer or float attribute of the invocation, a tensor dimension, a tensor element count, or a literal constant. The block SHALL be constant. Each grid axis SHALL be a constant or the ceil-division of one such derived value by a constant. The descriptor SHALL NOT carry expressions, pointer arithmetic or code.

#### Scenario: An unknown parameter kind

- **WHEN** a descriptor declares a parameter kind outside the vocabulary
- **THEN** the descriptor is invalid and the artifact is not prepared

### Requirement: Descriptor Validation Before Launch

The Runtime and the Provider SHALL validate the descriptor at preparation against the semantic binding, the specialization and the Device limits: input and output indices that exist, attribute types, a block within limits, dynamic shared memory within limits, and integer parameters that cannot overflow for the specialization's shape range. A descriptor that fails SHALL fail preparation, never a launch.

#### Scenario: A block over the Device limit

- **WHEN** the block size exceeds the Device's maximum threads per block
- **THEN** preparation fails with an invalid launch descriptor error

#### Scenario: An index past the binding

- **WHEN** a parameter names input 2 of an operator with two inputs
- **THEN** preparation fails before any module is loaded
