## ADDED Requirements

### Requirement: Production Qwen Component Is Configuration-Driven

The production Qwen Model Component SHALL derive all load-bearing architecture dimensions and graph structure from Runtime-authorized normalized model configuration rather than compile-time fixture constants.

This SHALL include hidden size, intermediate size, layer count, attention head count, KV head count, head dimension, vocabulary size, RMSNorm epsilon, RoPE parameters, and tied-embedding behavior where applicable.

#### Scenario: Two different compatible Qwen configs
- **WHEN** Model Instance A and Model Instance B declare different valid layer/head/intermediate dimensions
- **THEN** the same production Qwen Component implementation can produce configuration-correct graphs for each
- **AND** it does not require rebuilding the Component with different Rust constants.

---

### Requirement: Production Qwen Graph Shape Follows Loaded Artifact Configuration

The Qwen Component SHALL derive logical tensor names and expected tensor shapes from normalized configuration and SHALL validate them against the Runtime-recognized weight inventory.

#### Scenario: Layer tensor missing
- **WHEN** `num_hidden_layers` requires layer N
- **AND** a required layer-N tensor is absent from the validated Model Artifact
- **THEN** the model cannot become a ready executable Qwen Model Instance.

#### Scenario: Tensor shape mismatches config
- **WHEN** a required Qwen weight's parsed shape disagrees with the shape implied by normalized architecture config
- **THEN** compatibility/loading fails before Kernel dispatch.

---

### Requirement: Production Qwen Component Remains Provider And Source Agnostic

Making the Qwen Component configurable SHALL NOT grant it model-source, filesystem/network, Provider, or Device authority.

#### Scenario: Production graph on CUDA
- **WHEN** Runtime selects a CUDA Provider for a loaded Qwen Model Instance
- **THEN** the Qwen Component produces the same portable architecture graph semantics without receiving CUDA or Device identity.
