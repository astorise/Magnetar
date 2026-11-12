## ADDED Requirements

### Requirement: Provider-facing API in its own crate

The Runtime SHALL expose everything a Provider needs to implement `ProviderExecutionApi` and `Provider` (kernel invocations and results, operators, affinity, devices, compute descriptors, packed weights, memory admission, execution errors, tensor parallel and layer placement plans) from a crate that does not depend on the Runtime's generation engine, Component host or kernel optimization pipeline. `magnetar-runtime` SHALL depend on that crate and SHALL NOT be a dependency of it.

#### Scenario: A Provider builds without the generation engine

- **WHEN** a Provider depends only on the provider API crate
- **THEN** it compiles and implements `ProviderExecutionApi` without compiling `first_native_runtime`, the Wasm Component host or the kernel optimization pipeline

#### Scenario: Dependency direction

- **WHEN** the workspace dependency graph is inspected
- **THEN** the provider API crate depends on no other Magnetar crate, the kernel crate depends at most on the provider API crate, and `magnetar-runtime` depends on both; no extracted crate depends on `magnetar-runtime`

### Requirement: Stable public paths during the move

Every public item that moves to another module or crate SHALL remain reachable at its previous `magnetar_runtime::...` path through a re-export until a separate change removes the old path. Moving an item SHALL NOT change its behavior, signature or trait implementations.

#### Scenario: Existing dependents keep compiling

- **WHEN** a stage of this change lands
- **THEN** `providers/*`, `loaders/*`, `formats/*`, `components/*`, `inference-components`, `magnetar-cli` and the integration tests compile without source changes

#### Scenario: No behavior change

- **WHEN** a stage of this change lands
- **THEN** the Runtime's unit and contract tests pass unchanged in number and outcome, and the GPU workflow's token checks and qualification results are unchanged

### Requirement: Generation engine split by responsibility

The first-native generation engine SHALL be organized in modules each owning one responsibility (fixture and conformance report, component graph construction and cache, per-node dispatch, graph execution and KV state, step capture and replay, weight materialization, placement and tensor parallelism, the production loaded model), with each module's tests next to it. No module of the engine SHALL exceed a documented size budget without a recorded reason.

#### Scenario: Changing step replay

- **WHEN** a change touches only step capture and replay
- **THEN** it edits the step-replay module and its tests, and no other engine module's file

### Requirement: Roadmap scaffolding outside the compiled Runtime

Roadmap contracts that describe future work (phases, rejection rules for unimplemented formats or Providers) SHALL NOT be compiled into `magnetar-runtime`; they SHALL live in `roadmap-contracts`. Types and validators that current code uses SHALL live in a module named for what they do, not for the roadmap.

#### Scenario: A Loader using model format metadata

- **WHEN** `loaders/huggingface` needs `HfConfigMetadata` or `detect_duplicate_tensor_names`
- **THEN** it imports them from the Runtime's model format module, and `magnetar-runtime` contains no module named for a roadmap
