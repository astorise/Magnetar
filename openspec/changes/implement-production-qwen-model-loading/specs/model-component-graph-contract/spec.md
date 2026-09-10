## ADDED Requirements

### Requirement: Graph Contract Exposes Runtime-Authorized Model Configuration

The versioned Model Component graph contract SHALL provide a Runtime-owned Capability through which a Component can obtain normalized model-architecture configuration required to produce portable graphs.

The Capability SHALL expose only Runtime-authorized logical configuration and SHALL NOT expose raw config files, arbitrary filesystem paths, raw model-weight bytes, Provider/Device identities, Runtime native handles, or unrestricted process-local state.

#### Scenario: Qwen Component requests hidden size
- **WHEN** the active Model Instance was loaded from a validated Qwen artifact
- **AND** the Qwen Component requests its normalized hidden size through the configuration Capability
- **THEN** Runtime returns the value bound to that loaded Model Artifact/Instance.

#### Scenario: Component requests Provider identity through config
- **WHEN** a Component attempts to obtain Provider or Device selection through the model-configuration Capability
- **THEN** the capability does not expose such a value and Provider/Device resolution remains Runtime-owned.

---

### Requirement: Model Configuration Is Bound To The Active Validated Artifact

Configuration values exposed to a Model Component SHALL be derived from the Runtime-authorized normalized metadata of the active loaded artifact/Model Instance.

#### Scenario: Config from another model
- **WHEN** a Component instance is producing a graph for Model Instance A
- **THEN** it cannot obtain architecture configuration from Model Instance B or caller-invented metadata.

---

### Requirement: Production Tensor Descriptors Are Not Fixed To Fixture Float32 Assumptions

The production-capable graph contract SHALL represent the portable tensor descriptor information required to validate graphs for supported production model configurations rather than hard-coding the fixture's contiguous-Float32 assumption.

#### Scenario: Production model metadata differs from fixture dimensions
- **WHEN** a compatible loaded Qwen artifact has different hidden size, layer count, vocabulary size, or head configuration from the baseline fixture
- **THEN** the Component can produce a valid graph whose tensor shapes are derived from that model configuration
- **AND** the Runtime validates those descriptors through normal graph validation.
