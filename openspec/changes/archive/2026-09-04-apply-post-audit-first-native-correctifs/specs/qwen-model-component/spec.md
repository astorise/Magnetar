## ADDED Requirements

### Requirement: Qwen WASM Component Is First-Native Authority
The first-native Qwen E2E path SHALL instantiate an executable Qwen WASM Component Artifact and use its validated output as the source of graph semantics.

#### Scenario: Component supplies graph
- **WHEN** first-native Qwen E2E builds its execution graph
- **THEN** the graph originates from the instantiated Qwen WASM Component and passes Runtime validation before planning.

#### Scenario: Missing component fails
- **WHEN** the Qwen Component Artifact is unavailable, invalid, untrusted, or cannot be instantiated
- **THEN** first-native Qwen E2E fails with a structured component or trust error and does not fall back to a Rust fixture graph.

### Requirement: Qwen Component Has No Provider Authority
The Qwen Component SHALL NOT receive Provider, Device, Kernel, stream, queue, memory pointer, or native resource handles.

#### Scenario: Provider selection stays runtime-owned
- **WHEN** Qwen Component describes model operations
- **THEN** Runtime remains responsible for Provider, Device, Kernel, memory, and KV resource decisions.
