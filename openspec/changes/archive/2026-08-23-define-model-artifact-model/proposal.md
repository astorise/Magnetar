# Define Model Artifact Model

## Why

Magnetar is scoped to local AI inference execution.

The Runtime now has contracts for:

- Components
- Component Artifacts
- inference-scoped Component authority
- Component distribution
- Providers
- Devices
- Runtime Memory Manager
- platform-specific Component Engines

The next missing foundation is the Model Artifact model.

Model data must not be represented as executable Component code.

A model is not a Provider.

A model architecture may be implemented by a Component or native Runtime code,
but the model weights, tokenizer files, configuration, quantization metadata,
and templates are data artifacts.

Magnetar needs a stable model for identifying, validating, loading, placing,
and trusting model data.

Without this model, future inference features would blur together:

```text
model weights
tokenizer files
chat templates
quantization metadata
adapter weights
Component code
Provider execution
Device placement
```

That would make model loading, caching, memory residency, trust, and
compatibility unsafe.

This change defines the Model Artifact model.

## What Changes

This change introduces Model Artifacts as first-class inference data artifacts.

A Model Artifact SHALL represent model-related data required for inference.

A Model Artifact MAY include or reference:

- model weights
- model configuration
- tokenizer data
- tokenizer configuration
- chat template
- prompt template
- generation defaults
- architecture metadata
- quantization metadata
- dtype metadata
- sharding metadata
- vocabulary metadata
- special token metadata
- license metadata
- provenance metadata
- optional signature metadata
- optional adapter compatibility metadata

A Model Artifact SHALL be distinct from:

- Component Artifact
- Provider binary
- Runtime configuration
- Device metadata
- execution plan
- inference session state
- KV cache
- prefix cache

## Artifact Kinds

The Model Artifact model SHALL distinguish artifact kinds.

Initial kinds SHOULD include:

```text
model-bundle
model-weights
model-config
tokenizer
tokenizer-config
chat-template
prompt-template
generation-config
quantization-config
adapter
vocabulary
special-tokens
```

A `model-bundle` may reference multiple artifact parts.

For example:

```text
model-bundle
    ├── model-weights
    ├── model-config
    ├── tokenizer
    ├── tokenizer-config
    ├── chat-template
    ├── generation-config
    └── quantization-config
```

## Model Artifact Identity

A Model Artifact SHALL have immutable content identity.

The identity SHALL include:

- artifact kind
- digest algorithm
- content digest
- logical model name
- model version or revision
- optional variant
- optional shard identity
- optional source identity

The digest SHALL be authoritative for content identity.

Logical names, aliases, paths, registry tags, or user-friendly names SHALL NOT
be sufficient identity.

## Digest

Model Artifacts SHALL be content-addressable.

The Runtime SHALL support at least SHA-256 digest identity.

A source-provided digest SHALL be treated as a claim until the Runtime verifies
artifact bytes locally.

A digest mismatch SHALL reject the artifact before loading or residency.

## Manifest

A Model Artifact SHALL include or resolve to a manifest.

The manifest SHALL describe:

- schema version
- artifact kind
- logical model identity
- revision
- architecture family
- architecture identifier
- artifact parts
- digests
- dtypes
- layouts
- tensor metadata where available
- tokenizer references
- template references
- generation defaults
- quantization metadata
- sharding metadata
- required Runtime features
- required Memory Manager features
- required Provider Capabilities
- optional Component requirements
- optional license metadata
- optional provenance metadata
- optional signature metadata

The manifest SHALL NOT be treated as proof of trust.

Trust is decided by Runtime policy.

## Bundle Manifest

A model bundle manifest SHALL describe the relationship between model parts.

Example conceptual structure:

```yaml
schema: magnetar-model-artifact
schema_version: 1

model:
  name: qwen.example
  revision: 2026-08-23
  architecture: qwen
  variant: instruct

artifacts:
  weights:
    kind: model-weights
    digest: sha256:...
    format: safetensors
  config:
    kind: model-config
    digest: sha256:...
  tokenizer:
    kind: tokenizer
    digest: sha256:...
  tokenizer_config:
    kind: tokenizer-config
    digest: sha256:...
  chat_template:
    kind: chat-template
    digest: sha256:...
  generation_config:
    kind: generation-config
    digest: sha256:...
```

The exact serialized schema is implementation-defined.

The semantic requirements are defined here.

## Model Architecture

Model architecture metadata SHALL identify the model family or architecture.

Examples:

```text
llama
qwen
gemma
mistral
phi
custom
```

Architecture metadata SHALL NOT create a Provider.

The following is invalid architecture thinking:

```text
LlamaProvider
QwenProvider
GemmaProvider
```

Providers are execution implementations such as CPU, CUDA, Metal, OpenVINO, QNN,
or temporary Candle Provider.

The model architecture determines how model data is interpreted.

The Provider determines how compatible operations execute.

## Model Component Relationship

A model architecture implementation MAY be provided by a Magnetar Component.

For example:

```text
Model Component + Model Artifact + Provider + Device = Model Instance
```

But the Component Artifact and Model Artifact remain separate.

The Model Component is executable code.

The Model Artifact is model data.

They have separate identity, trust, validation, loading, and caching behavior.

## Provider Relationship

A Model Artifact SHALL NOT select a Provider directly.

A Model Artifact may declare required Capabilities or supported execution
constraints.

The Runtime uses Resolution Policy, Resource Affinity, Memory Manager, and
Provider advertisements to determine placement and execution.

A model manifest SHALL NOT contain authoritative Provider or Device selection.

## Device Relationship

A Model Artifact SHALL NOT select a Device directly.

It may declare memory requirements, supported dtypes, expected layouts, or
sharding metadata.

The Runtime determines feasible Device placement during loading and execution.

## Memory Relationship

Model Artifact loading SHALL be integrated with the Runtime Memory Manager.

The Memory Manager SHALL own:

- model residency
- weights materialization
- storage dtype implications
- compute dtype implications
- temporary dequantization workspace
- sharded placement feasibility
- adapter residency
- transfer staging
- memory pressure
- loading admission

A Model Artifact manifest may declare memory-relevant metadata.

The Memory Manager determines whether loading is feasible.

## Storage DType And Compute DType

A Model Artifact SHALL distinguish stored representation from compute
representation.

For example:

```text
storage_dtype = int8
compute_dtype = bf16

storage_dtype = q4_k
compute_dtype = fp16

storage_dtype = fp16
compute_dtype = fp32
```

The artifact manifest SHALL declare storage dtype where known.

Runtime loading policy may choose or require compute dtype.

The Memory Manager SHALL account for both.

## Quantization

Quantization metadata SHALL be represented explicitly.

Quantization metadata MAY include:

- quantization format
- group size
- block size
- scale dtype
- zero-point dtype
- per-channel/per-tensor behavior
- calibration metadata
- supported compute dtype
- required dequantization workspace
- Provider capability requirements

Quantized storage SHALL not imply that every Provider can execute it directly.

Runtime must validate Provider and Memory Manager compatibility.

## Sharding

Model Artifacts MAY be sharded.

The manifest SHALL describe:

- shard identities
- shard digests
- shard ordering or mapping
- tensor-to-shard mapping where available
- shard sizes
- required shard count
- optional parallel loading behavior

All required shards SHALL be validated before the model is considered complete.

## Tensor Metadata

Where available, Model Artifacts SHOULD expose tensor metadata.

Tensor metadata MAY include:

- tensor name
- shape
- storage dtype
- layout
- shard location
- offset
- size
- quantization metadata
- expected compute dtype
- optional semantic role

Tensor metadata supports Memory Manager planning.

It SHALL not expose raw memory handles.

## Tokenizer Relationship

A tokenizer may be represented as a Model Artifact part.

This change defines tokenizer artifact identity and association only.

The tokenizer execution contract is defined later by:

```text
define-tokenizer-contract
```

A model bundle may reference a tokenizer artifact.

## Chat And Prompt Templates

Chat templates and prompt templates may be represented as Model Artifact parts.

This change defines their identity and association only.

Template evaluation behavior is defined later.

## Generation Defaults

Generation defaults may be represented as Model Artifact metadata.

Examples:

- temperature
- top-p
- top-k
- max tokens
- stop tokens
- repetition penalty
- default chat template

These are defaults, not mandatory Runtime policy.

Runtime or client policy may override them.

## Adapter Compatibility

Model Artifacts MAY declare adapter compatibility metadata.

Adapters themselves MAY be Model Artifact parts or separate artifact kinds.

This change defines identity and compatibility placeholders only.

Full adapter loading behavior is deferred.

## Trust

Model Artifacts SHALL be validated and trusted before loading.

Trust policy may consider:

- digest
- source
- publisher
- signature metadata
- provenance metadata
- license metadata
- revocation
- local administrator decision

A Model Artifact SHALL NOT declare itself trusted.

Trust is separate from manifest content.

## License Metadata

Model Artifacts MAY include license metadata.

License metadata SHALL be recorded.

This change does not implement license enforcement.

Runtime or client policy may later decide how to enforce license constraints.

## Provenance

Model Artifacts MAY include provenance metadata.

Provenance may include:

- source repository
- registry
- original model identifier
- conversion tool
- conversion timestamp
- builder identity
- commit digest
- dataset metadata where provided
- publisher

Provenance SHALL NOT imply trust by itself.

## Source Neutrality

Model Artifacts may come from multiple sources.

Possible sources include:

- local directory
- local cache
- client-provided source
- registry
- Hugging Face style source
- OCI artifact source
- Tachyon source
- development fixture

Source identity SHALL not imply trust.

The concrete model distribution protocol is not defined by this change.

## Loading Pipeline

The conceptual Model Artifact loading pipeline is:

```text
source / local path / cache
        |
        v
Model Artifact bytes + manifest
        |
        v
compute digest
        |
        v
validate manifest
        |
        v
validate all parts / shards
        |
        v
validate architecture metadata
        |
        v
validate dtype / quantization metadata
        |
        v
validate tokenizer/template associations
        |
        v
evaluate trust policy
        |
        v
memory feasibility via Memory Manager
        |
        v
create Model Artifact record
        |
        v
create Model Residency plan
        |
        v
later: create Model Instance
```

This change defines artifacts and validation.

It does not fully define Model Instance execution.

## Model Instance Boundary

A Model Artifact is not a Model Instance.

A Model Instance is a loaded executable inference entity.

Conceptually:

```text
Model Instance =
    Model Artifact
  + model architecture implementation
  + Runtime loading plan
  + Memory residency
  + Provider resolution
  + Device placement
```

This change prepares for Model Instance but does not fully define it.

## Error Model

Model Artifact errors SHALL be structured.

Error categories SHOULD include:

- manifest missing
- manifest invalid
- unsupported manifest version
- artifact digest mismatch
- missing required part
- shard digest mismatch
- incomplete shard set
- unsupported model architecture
- unsupported artifact format
- unsupported storage dtype
- unsupported compute dtype
- unsupported quantization format
- invalid tensor metadata
- tokenizer reference missing
- template reference missing
- trust rejected
- revoked artifact
- license policy denied
- memory feasibility failed
- model source unavailable

## Observability

Runtime SHOULD emit observations for:

- model artifact discovered
- manifest loaded
- manifest validation failed
- digest computed
- digest mismatch
- shard validated
- artifact trusted
- artifact rejected
- memory feasibility checked
- model residency planned
- model artifact cached
- model artifact evicted
- model source failure

Observability SHALL not expose secrets or raw file content.

## Non-Goals

This change does not:

- implement model inference
- define generation API
- define tokenizer execution contract
- define sampling contract
- define KV cache semantics
- define Model Instance lifecycle fully
- define model registry protocol
- define model download protocol
- define Hugging Face integration
- define Tachyon model distribution
- define license enforcement
- define adapter loading behavior fully
- define LoRA merge behavior
- define distributed model sharding
- define cross-node model placement
- define Provider-specific model loading ABI
- allow Model Artifacts to select Providers directly
- allow Model Artifacts to select Devices directly

## Impact

Magnetar gains a clear model data boundary.

The Runtime can now distinguish:

```text
executable Component code
model data
native Provider execution
Device placement
Runtime memory residency
```

This prepares future changes:

- tokenizer contract
- generation contract
- inference session model
- KV cache model
- model loading contract
- sampling and logits processing contract