## ADDED Requirements

### Requirement: Production Model Loading Consumes Registered Artifact Ingestors

Runtime SHALL support loading from a normalized artifact produced by a registered, format-neutral Model Artifact ingestion contract.

Concrete ingestion implementations SHALL NOT gain Runtime trust authority, Provider/Device selection authority, or Model Instance readiness authority.

#### Scenario: Hugging Face ingestor returns normalized artifact
- **WHEN** a registered external Hugging Face ingestor normalizes an authorized model bundle
- **THEN** Runtime validates the resulting `ModelManifest`, trust, integrity, component compatibility, memory/residency policy, and Provider capabilities through the standard Model Loading path before any ready Model Instance is published.

#### Scenario: Parser success but artifact untrusted
- **WHEN** a concrete ingestor successfully parses all files but Runtime trust policy rejects the normalized artifact
- **THEN** Model Loading fails before weight materialization and no ready Model Instance is published.

---

### Requirement: Production Model Loading Supports Bounded Tensor Payload Access

Production Model Loading SHALL be able to materialize required model tensors from bounded payload access associated with the validated artifact without requiring the entire model to first exist as a whole-model host tensor map.

#### Scenario: Multi-gigabyte model
- **WHEN** a model has many tensors across one or more weight files
- **THEN** Runtime may read, validate, convert, and stage tensors incrementally according to explicit buffering policy
- **AND** peak transient host staging SHALL be bounded independently of total model size except where policy explicitly requests full residency.

---

### Requirement: Production Weight Materialization Is Transactional

Production model weight materialization SHALL use one transaction whose commit is the authority for publishing required weight bindings/readiness.

#### Scenario: Middle tensor fails
- **WHEN** a required tensor fails payload validation, conversion, allocation, Provider write, or completion after earlier tensors were staged
- **THEN** Runtime releases all staged Provider-side tensors and Memory Manager allocations/residency created by the transaction
- **AND** no incomplete resource binding/evidence is published
- **AND** the Model Instance does not become Ready.

---

### Requirement: Production Loading Supports F16 And BF16 Storage

The first production Qwen profile SHALL support model weights stored as F32, F16, or BF16.

Storage dtype and compute dtype SHALL remain distinct. Any conversion SHALL be explicit in the loading/residency plan and SHALL NOT be inferred silently from source annotations such as `torch_dtype`.

#### Scenario: BF16 checkpoint on Float32-only Provider
- **WHEN** a validated checkpoint stores a required tensor as BF16
- **AND** the selected Provider supports Float32 compute but not native BF16 compute for the required Kernel path
- **THEN** Model Loading MAY convert BF16 storage values to Float32 staging values explicitly
- **AND** tensor content integrity is verified against the original storage bytes before conversion.

#### Scenario: Unsupported storage dtype
- **WHEN** the checkpoint declares a storage dtype the production loader does not support
- **THEN** loading fails with a structured storage-dtype error rather than reinterpreting or silently dequantizing bytes.

---

### Requirement: Production Model Loading Supports Sharded Safetensors Artifacts

The first production Qwen profile SHALL support single-file Safetensors and Hugging Face-style indexed sharded Safetensors bundles.

#### Scenario: Sharded model is valid
- **WHEN** `model.safetensors.index.json` maps required tensors across multiple present, valid shards
- **THEN** Model Loading normalizes every tensor/shard mapping into the canonical Model Artifact contracts and may materialize the model.

#### Scenario: Required shard missing
- **WHEN** the index references a required shard that is unavailable
- **THEN** loading fails before ready Model Instance publication.

---

### Requirement: Production Loading Produces A Ready Executable Model Instance

For inference target usage, the supported production loading surface SHALL sequence artifact ingestion, Runtime validation, Component compatibility, residency planning, weight materialization, Model Instance warm/readiness validation, and publication into one supported caller workflow.

#### Scenario: Authorized Qwen bundle loads
- **WHEN** a caller supplies an authorized, compatible production Qwen bundle and required Runtime resources are available
- **THEN** the supported loading workflow returns a Runtime-owned ready `ModelInstanceId`
- **AND** the caller does not need to construct trust decisions, tensor resources, weight bindings, or Provider allocations.

---

### Requirement: Production Model Loading Has No Fixture Identity Requirement

No production Model Loading success path SHALL require a specific fixture model reference such as `qwen-test`.

#### Scenario: Non-fixture Qwen identity
- **WHEN** a validated Qwen-compatible Model Artifact has an identity other than `qwen-test`
- **THEN** it can follow the same supported production loading workflow based on its normalized metadata and compatibility
- **AND** no fixture manifest or fixture tensor inventory is synthesized.
