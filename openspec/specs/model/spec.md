# model Specification

## Purpose
TBD - created by archiving change define-model-artifact-model. Update Purpose after archive.
## Requirements
### Requirement: Model Artifact

A Model Artifact SHALL represent model-related inference data.

A Model Artifact SHALL be distinct from executable Component code, Provider
binaries, Runtime configuration, Device metadata, and Model Instances.

#### Scenario: Classify model weights

Given a file contains model weights

When Magnetar classifies the file

Then the file is a Model Artifact

And not a Component Artifact.

---

### Requirement: Model Artifact Is Not Component Artifact

A Model Artifact SHALL NOT be treated as executable WASM Component code.

#### Scenario: Model bundle with Component

Given a model bundle references a Model Component

When Magnetar validates the bundle

Then the Component Artifact and Model Artifact keep separate identities and
trust decisions.

---

### Requirement: Model Artifact Is Not Provider

A Model Artifact SHALL NOT be a Provider and SHALL NOT define Provider
execution.

#### Scenario: Qwen model loaded

Given a Qwen Model Artifact is loaded

When Runtime selects execution implementation

Then it resolves a Provider such as CPU, CUDA, Metal, OpenVINO, QNN, or Candle

And does not create a `QwenProvider`.

---

### Requirement: Model Artifact Identity

A Model Artifact SHALL have immutable content identity based on digest.

Logical model names, paths, aliases, registry tags, or user-friendly names SHALL
not be sufficient identity.

#### Scenario: Same name different digest

Given two model artifacts share the same logical model name

But have different content digests

When Magnetar records them

Then they are distinct artifacts.

---

### Requirement: Model Artifact Manifest

A Model Artifact SHALL include or resolve to a manifest describing model
identity, architecture, parts, digests, dtype metadata, tokenizer/template
association, quantization metadata, sharding metadata, compatibility, and trust
metadata.

#### Scenario: Missing manifest

Given model artifact bytes have no manifest

When Runtime validation runs

Then validation fails unless explicit development policy allows a limited
fallback.

---

### Requirement: Model Bundle

A model bundle SHALL be allowed to reference multiple Model Artifact parts.

A bundle SHALL validate all required parts before it is considered complete.

#### Scenario: Missing tokenizer part

Given a text-generation model bundle requires a tokenizer

And the tokenizer artifact is missing

When validation runs

Then the bundle is rejected.

---

### Requirement: Artifact Kind

Model Artifact kinds SHALL be explicit.

Initial kinds SHOULD include model-bundle, model-weights, model-config,
tokenizer, tokenizer-config, chat-template, prompt-template, generation-config,
quantization-config, adapter, vocabulary, and special-tokens.

#### Scenario: Unknown artifact kind

Given a manifest declares an unknown artifact kind

When validation runs

Then Runtime rejects it or applies explicit compatibility policy.

---

### Requirement: Model Architecture Metadata

A Model Artifact SHALL declare architecture metadata sufficient for Runtime to
select compatible model implementation.

Architecture metadata SHALL NOT select Provider directly.

#### Scenario: Architecture declared

Given a manifest declares architecture `llama`

When Runtime validates it

Then the architecture is used to find a compatible model implementation

And not to select `LlamaProvider`.

---

### Requirement: No Provider Selection

A Model Artifact manifest SHALL NOT provide authoritative Provider selection.

#### Scenario: Manifest pins CUDA

Given a Model Artifact manifest attempts to require Provider `cuda`

When validation runs

Then the field is rejected or treated as non-authoritative policy metadata

And Runtime Resolution remains authoritative.

---

### Requirement: No Device Selection

A Model Artifact manifest SHALL NOT provide authoritative Device selection.

#### Scenario: Manifest pins GPU 0

Given a Model Artifact manifest attempts to select Device `gpu:0`

When validation runs

Then the field is rejected or treated as non-authoritative policy metadata

And Runtime placement remains authoritative.

---

### Requirement: Storage DType

A Model Artifact SHALL declare storage dtype where known.

Storage dtype describes how model data is stored.

#### Scenario: INT8 weights

Given model weights are stored as INT8

When Memory Manager plans loading

Then allocation size uses INT8 storage metadata.

---

### Requirement: Compute DType Compatibility

A Model Artifact SHALL be allowed to declare supported or preferred compute dtypes.

Compute dtype describes how execution may run.

#### Scenario: BF16 compute

Given a model supports BF16 compute

When Runtime plans loading

Then Memory Manager accounts for required compute workspace and Provider
compatibility.

---

### Requirement: Quantization Metadata

Quantized Model Artifacts SHALL declare quantization metadata.

#### Scenario: Unsupported quantization

Given a model uses a quantization format unsupported by Runtime or Provider
policy

When validation runs

Then the artifact is rejected or marked unloadable.

---

### Requirement: Sharded Model Artifact

A Model Artifact SHALL be allowed to be sharded.

All required shards SHALL be validated by digest before the artifact is
considered complete.

#### Scenario: Shard digest mismatch

Given a model bundle references shard A with digest X

And Runtime computes digest Y

When validation runs

Then the model artifact is rejected.

---

### Requirement: Tensor Metadata

Model Artifacts SHALL be allowed to expose tensor metadata where available.

Tensor metadata SHALL not expose raw memory handles.

#### Scenario: Tensor metadata used

Given tensor metadata includes shape and storage dtype

When Memory Manager estimates loading feasibility

Then it may use that metadata without exposing raw storage handles.

---

### Requirement: Tokenizer Association

A Model Artifact SHALL be allowed to reference tokenizer-related artifacts.

Tokenizer execution behavior is defined by a later tokenizer contract.

#### Scenario: Tokenizer reference

Given a text-generation model bundle references tokenizer artifact T

When validation runs

Then T is validated as a Model Artifact part or associated artifact.

---

### Requirement: Template Association

A Model Artifact SHALL be allowed to reference chat or prompt templates.

Template rendering behavior is defined later.

#### Scenario: Chat template reference

Given an instruct model bundle references a chat template

When validation runs

Then the template artifact identity is validated.

---

### Requirement: Generation Defaults

A Model Artifact SHALL be allowed to declare generation defaults.

Generation defaults SHALL be overridable by Runtime or client policy.

#### Scenario: Default temperature

Given a model declares default temperature

When a client requests another temperature allowed by policy

Then the client or Runtime value may override the default.

---

### Requirement: Model Artifact Trust

A Model Artifact SHALL be validated and trusted before loading.

A Model Artifact SHALL NOT declare itself trusted.

#### Scenario: Manifest claims trusted

Given a Model Artifact manifest claims the model is trusted

When Runtime evaluates trust

Then Runtime uses trust policy instead.

---

### Requirement: Model Artifact Provenance

A Model Artifact SHALL be allowed to include provenance metadata.

Provenance SHALL be separate from trust.

#### Scenario: Provenance present

Given a model manifest includes source repository and conversion tool metadata

When validation completes

Then Runtime records provenance

But does not infer trust from it alone.

---

### Requirement: Model Artifact License Metadata

A Model Artifact SHALL be allowed to include license metadata.

This change records license metadata but does not require license enforcement.

#### Scenario: License metadata

Given a model manifest includes a license identifier

When validation completes

Then Runtime records the license metadata.

---

### Requirement: Model Artifact Source Is Metadata

Model Artifact source identity SHALL be metadata and SHALL NOT imply trust.

#### Scenario: Registry source

Given a model comes from a known registry

When Runtime evaluates trust

Then the source may inform policy

But does not automatically trust the model.

---

### Requirement: Model Artifact Loading Is Memory-Managed

Model Artifact loading SHALL use the Runtime Memory Manager for feasibility,
residency, and pressure.

#### Scenario: Model too large

Given a model requires more memory than policy permits

When loading feasibility is evaluated

Then Memory Manager rejects the load with a structured memory error.

---

### Requirement: Model Artifact Is Not Model Instance

A Model Artifact SHALL not be treated as a loaded Model Instance.

#### Scenario: Artifact validated

Given a Model Artifact validates and is trusted

When no loading request is made

Then no Model Instance is created.

---

### Requirement: Model Artifact Error Categories

Model Artifact validation and loading SHALL use structured error categories.

#### Scenario: Unsupported architecture

Given a manifest declares an unsupported architecture

When validation runs

Then Runtime returns an unsupported-model-architecture style error.

