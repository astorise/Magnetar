## ADDED Requirements

### Requirement: The graph-builder offers every tensor of the loaded artifact

For a loaded Model Artifact, the graph-builder SHALL offer each tensor of the artifact's inventory as a weight a Component may request:
- a tensor the architecture configuration derives SHALL keep its derived shape;
- any other tensor SHALL have the shape the inventory declares.

#### Scenario: Per-head norm weights
- **WHEN** a Component requests `layers.0.self_attn.q_norm` and the loaded artifact carries that tensor
- **THEN** the graph-builder binds it with the artifact's shape, and the graph is built
