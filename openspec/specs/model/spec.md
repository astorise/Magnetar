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

### Requirement: Model References Tokenizer Artifact

A text-generation Model Artifact SHALL validate tokenizer references when tokenizer metadata is required.

#### Scenario: Model bundle tokenizer

Given a model bundle includes tokenizer metadata

When Runtime validates the model bundle

Then the referenced tokenizer artifact identity is validated.

---

### Requirement: Tokenizer Compatibility Is Part Of Model Validation

Model validation SHALL include tokenizer compatibility when a tokenizer is
required.

#### Scenario: Wrong tokenizer

Given a model expects tokenizer digest A

And a tokenizer artifact with digest B is selected

When validation runs

Then Runtime rejects the pairing unless explicit policy permits override.

---

### Requirement: Tokenizer Metadata Does Not Define Generation

Tokenizer metadata SHALL NOT define generation behavior beyond tokenization
defaults and special token metadata.

#### Scenario: EOS metadata

Given tokenizer metadata defines EOS token ID

When generation later uses EOS as a stop condition

Then generation behavior is defined by Generation Contract

And tokenizer only supplies token metadata.

### Requirement: Model Supports Generation Metadata

A Model Artifact or future Model Instance SHALL expose metadata required for
generation validation.

Metadata MAY include context length, supported generation modes, supported
dtypes, supported tokenizer, EOS token metadata, and architecture generation
support.

#### Scenario: Model has context length

Given a model declares context length

When generation validates prompt and max new tokens

Then the model context length is used.

---

### Requirement: Model Artifact Alone Is Not Executable Generation

A Model Artifact SHALL not be treated as an executable generation context by
itself.

#### Scenario: Artifact trusted

Given a Model Artifact is trusted

When generation is requested

Then Runtime still requires a loaded model context or future Model Instance.

### Requirement: Session References Model Context

An Inference Session SHALL reference a validated model context or future Model
Instance.

A Model Artifact alone SHALL not be sufficient to run session generation unless
Runtime policy performs loading first.

#### Scenario: Session with artifact only

Given a session creation request references only an unloaded Model Artifact

When Runtime does not allow implicit loading

Then session creation fails with model-unavailable.

---

### Requirement: Model Residency May Outlive Session

Model residency SHALL be able to outlive a session according to Runtime cache policy.

#### Scenario: Session closes

Given a session references resident model memory

When the session closes

Then model residency may remain cached if Runtime policy allows it.

### Requirement: Model Artifact Must Be Loaded Before Execution

A Model Artifact SHALL be loaded or materialized into Runtime-owned model
residency before it is used for inference execution.

#### Scenario: Generate from unloaded artifact

Given a Model Artifact is valid and trusted

But not loaded

When generation requires a loaded context

Then Runtime rejects the request or performs explicit policy-controlled loading.

---

### Requirement: Model Residency Is Distinct From Artifact Identity

Model Residency SHALL be distinct from Model Artifact identity.

#### Scenario: Same artifact multiple placements

Given the same artifact is loaded on CPU and GPU

When Runtime reports residency

Then the artifact identity is the same

But residency and Resource Affinity differ.

---

### Requirement: Model Loading Does Not Trust Artifacts

Loading a Model Artifact SHALL NOT make untrusted content trusted.

#### Scenario: Untrusted artifact loading

Given a Model Artifact is untrusted

When loading is requested

Then loading fails before materialization.

### Requirement: Model Metadata May Constrain Sampling

Runtime SHALL treat Model Artifact or loaded model metadata declarations for
supported or default sampling parameters as policy input during Sampling
validation.

#### Scenario: Unsupported sampling mode

Given a model metadata declares it does not support a requested sampling mode

When generation validates parameters

Then Runtime rejects the request or applies policy fallback.

---

### Requirement: Model Vocabulary Must Match Sampling

Loaded model output logits SHALL be compatible with tokenizer vocabulary
metadata used by Sampling.

#### Scenario: Logits vocabulary mismatch

Given loaded model produces logits for vocabulary size X

And tokenizer reports vocabulary size Y

When Sampling validates logits

Then Runtime reports vocabulary-mismatch.

### Requirement: Model Artifact May Reference Adapter Compatibility

Model Artifact metadata SHALL preserve adapter compatibility metadata when declared.

#### Scenario: Adapter compatible

Given a base model declares compatible adapter target modules

When adapter validation runs

Then Runtime may use that metadata to validate adapter targets.

---

### Requirement: Adapter Artifact Is Distinct From Base Model Artifact

Adapter Artifact identity SHALL remain distinct from base Model Artifact
identity.

#### Scenario: Adapter and base model

Given adapter A targets base model M

When Runtime records them

Then A and M remain separate artifacts with separate trust decisions.

### Requirement: Model Artifact May Have Multiple Instances

Runtime SHALL allow a single Model Artifact to have multiple Model Instances.

#### Scenario: Same artifact CPU and GPU

Given one Model Artifact is loaded for CPU and GPU execution

When Runtime records instances

Then each instance has distinct lifecycle, readiness, and residency.

---

### Requirement: Model Artifact Alone Is Not Executable

A valid and trusted Model Artifact SHALL not be treated as executable inference
state.

#### Scenario: Generate from artifact

Given a Model Artifact is valid

But no ready Model Instance exists

When generation is requested

Then Runtime rejects or performs explicit policy-controlled loading.

### Requirement: Model Artifact May Require Model Component

A Model Artifact SHALL be allowed to declare compatible Model Component requirements.

#### Scenario: Required component missing

Given a Model Artifact requires architecture family `qwen`

And no compatible Model Component or native implementation exists

When Model Loading validates it

Then loading fails with model-component-not-found or architecture-unsupported.

---

### Requirement: Model Artifact Remains Data

Model Artifact SHALL remain data and SHALL NOT embed Provider or Kernel
selection authority.

#### Scenario: Artifact requests CUDA kernel

Given Model Artifact metadata names a CUDA Kernel

When Runtime validates it

Then the metadata is rejected or treated as non-authoritative.

---

### Requirement: Qwen-Compatible Model Artifact

Model Artifact metadata SHALL be permitted to declare a Qwen-compatible
architecture family for Qwen baseline validation.

#### Scenario: Qwen artifact

Given artifact metadata declares Qwen-compatible architecture

When Model Loading validates it

Then Runtime resolves Qwen Model Component compatibility.

---

### Requirement: Model Artifact Does Not Select Qwen Provider

Qwen-compatible Model Artifact metadata SHALL NOT select a Qwen Provider.

#### Scenario: Artifact requests QwenProvider

Given artifact metadata references `QwenProvider`

When Runtime validates metadata

Then Runtime rejects it or treats it as non-authoritative invalid metadata.

---

### Requirement: Model References Are Inference API Inputs

Runtime Inference API SHALL support model references as inputs to model resolution, loading, session creation, or one-shot inference.

#### Scenario: Model reference

Given caller submits model reference `qwen-test`

When Runtime resolves it

Then Runtime maps it to validated Model Artifact metadata or reports resolution
failure.

---

### Requirement: Model Reference Does Not Grant File Access

A model reference SHALL not grant arbitrary filesystem access.

#### Scenario: Path-like reference

Given model reference looks like a filesystem path

When Runtime validates it

Then Runtime uses only authorized model source contracts or rejects it.

---

### Requirement: E2E Fixture Model Artifact

E2E conformance SHALL use a fixture Model Artifact that still passes normal
Model Artifact validation.

#### Scenario: Fixture artifact

Given fixture model artifact is loaded

When validation runs

Then normal artifact identity, manifest, config, tensor inventory, and trust
checks are applied.

---

### Requirement: E2E Fixture Does Not Bypass Model Artifact Contract

Fixture models SHALL not bypass Model Artifact validation.

#### Scenario: Invalid fixture manifest

Given fixture manifest is invalid

When E2E loading runs

Then Model Loading fails before Model Instance creation.

### Requirement: Normalized Model Artifact Manifest

Model Artifact SHALL support a normalized manifest that can be produced from
external formats.

#### Scenario: External format normalized

Given safetensors and config metadata are parsed

When normalization completes

Then Model Artifact manifest contains normalized identity, architecture,
weights, tokenizer, generation, trust, and integrity metadata.

---

### Requirement: Model Format Does Not Grant Trust

A model format SHALL not be trusted merely because it is recognized.

#### Scenario: Recognized untrusted format

Given safetensors file is parseable

But trust policy rejects its source

When Model Loading runs

Then loading fails.

---

### Requirement: Source Metadata Is Not Automatically Runtime Policy

Source metadata such as `torch_dtype` SHALL not silently become Runtime compute
policy.

#### Scenario: torch_dtype bf16

Given config declares `torch_dtype` as bf16

When Runtime loads model

Then compute dtype follows validated Runtime policy, not silent source metadata.

### Requirement: Model Source Candidate

Model Artifact workflow SHALL support source candidates that normalize into
Model Artifact metadata.

#### Scenario: Client-provided artifact

Given client provides artifact source

When normalization succeeds

Then Model Artifact metadata is available for loading.

---

### Requirement: Model Artifact Identity Uses Digest

Model Artifact identity SHALL use digest-based identity where possible.

#### Scenario: Name collision

Given two artifacts named `qwen-local`

When their digests differ

Then Runtime treats them as distinct artifacts.

---

### Requirement: Cached Model Still Validates

Cached Model Artifacts SHALL still pass validation before loading.

#### Scenario: Cached untrusted artifact

Given cached artifact is untrusted

When Model Loading runs

Then Runtime rejects it.

### Requirement: Model Artifact Trust Required For Release

Model Artifacts SHALL pass trust and integrity validation before release
baseline loading.

#### Scenario: Fixture model

Given fixture Model Artifact is used in E2E release gate

When Model Loading runs

Then fixture trust policy is explicit and validation passes.

---

### Requirement: Recognized Model Format Is Not Trust

Recognized model format SHALL not imply trust.

#### Scenario: Recognized safetensors

Given safetensors is parseable

When source trust is denied

Then loading is denied.

