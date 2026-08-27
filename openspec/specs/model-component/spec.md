# model-component Specification

## Purpose
This specification defines model Component identity, authority, metadata, graph production, cache compatibility, and Provider-neutral execution constraints.
## Requirements
### Requirement: Model Component

Magnetar SHALL define Model Component as a portable architecture implementation
role for inference.

#### Scenario: Qwen architecture

Given a Qwen Model Artifact is loaded

When Runtime needs architecture behavior

Then it uses a compatible Model Component or Runtime-native architecture
implementation.

---

### Requirement: Model Component Is Not Provider

A Model Component SHALL NOT be treated as a Provider.

#### Scenario: Qwen model execution

Given a Qwen Model Component produces an Execution Graph

When Runtime executes the graph

Then execution uses Runtime-selected Providers and Kernels, not a Qwen Provider.

---

### Requirement: Model Component Is Not Kernel

A Model Component SHALL NOT be treated as a Kernel.

#### Scenario: Attention graph

Given a Model Component emits an attention Operator

When Runtime plans execution

Then Kernel Registry selects a compatible attention Kernel later.

---

### Requirement: Model Component Is Not Model Artifact

A Model Component SHALL be distinct from Model Artifact data.

#### Scenario: Same component multiple artifacts

Given one Llama Model Component supports multiple compatible artifacts

When Runtime loads those artifacts

Then each Model Artifact remains distinct from the Component.

---

### Requirement: Model Component Identity

Model Component identity SHALL include stable versioned compatibility metadata.

#### Scenario: Unsupported component version

Given a Model Component requires unsupported Runtime Capability version

When Runtime validates it

Then Runtime rejects it with model-component-unsupported-version.

---

### Requirement: Architecture Compatibility Validation

Model Component SHALL validate architecture metadata for supported model
families.

#### Scenario: Unsupported architecture

Given a Model Artifact declares unsupported architecture metadata

When Model Component validation runs

Then validation fails with architecture-unsupported or
architecture-metadata-invalid.

---

### Requirement: Config Validation

Model Component SHALL validate model config through Runtime-authorized config data when config validation is supported.

It SHALL NOT read arbitrary filesystem paths.

#### Scenario: Filesystem denied

Given a Model Component attempts to open arbitrary path `/tmp/model.json`

When authority is checked

Then Runtime denies filesystem authority.

---

### Requirement: Target Module Metadata

Model Component SHALL expose target module metadata where adapters are
supported.

#### Scenario: Adapter validates q_proj

Given a LoRA adapter targets `q_proj`

When Adapter Loading validates it

Then Runtime uses Model Component target module metadata.

---

### Requirement: Graph Production

Model Component SHALL define graph production support for model-load, warmup, prefill,
decode, adapter, sampling-helper, or test phases.

#### Scenario: Decode graph

Given Generation requests decode graph

When Model Component produces it

Then Runtime validates it before planning or execution.

---

### Requirement: Operator Requirements

Model Component SHALL declare required portable Operators or Operator families.

It SHALL NOT require Provider-specific Kernel names as authoritative execution
requirements.

#### Scenario: Invalid kernel requirement

Given a Model Component declares `cuda.flash_attention_v2` as required portable
operator

When Runtime validates requirements

Then Runtime rejects it or treats it as non-authoritative invalid metadata.

---

### Requirement: Capability Requirements

Model Component SHALL declare inference-scoped Runtime Capability requirements.

#### Scenario: Missing graph Capability

Given a Model Component requires graph-production Capability

And Runtime does not provide it

When validation runs

Then Runtime rejects or disables that Component path.

---

### Requirement: Inference-Scoped Authority

Model Component authority SHALL be limited to inference-scoped permissions.

Filesystem, network, process, shell, secrets, workspace, Git, source-control,
tool-execution, and external-service authority SHALL be denied.

#### Scenario: Network denied

Given a Model Component attempts network access

When Runtime authorizes it

Then access is denied.

---

### Requirement: Provider Boundary

Model Component SHALL NOT receive raw Provider, Device, Kernel, memory, or
Provider-owned resource handles.

#### Scenario: Provider handle requested

Given a Model Component requests a Provider handle

When Runtime validates authority

Then Runtime denies access.

---

### Requirement: Component-Produced Graphs Are Untrusted Until Validated

Graphs produced by Model Components SHALL be validated by Runtime before use.

#### Scenario: Invalid graph

Given a Model Component emits a graph with invalid tensor edges

When Runtime receives it

Then Runtime rejects the graph.

---

### Requirement: Model Loading Uses Model Component Safely

Model Loading SHALL be allowed to use Model Component for architecture validation and graph
metadata, but SHALL NOT allow it to bypass artifact trust or memory admission.

#### Scenario: Untrusted artifact

Given Model Artifact is untrusted

When Model Component is compatible

Then loading still fails before materialization.

---

### Requirement: Model Instance References Model Component

Model Instance metadata SHALL be able to reference the Model Component or architecture
implementation used for the instance.

#### Scenario: Instance metadata

Given a Model Instance is ready

When Runtime reports redacted metadata

Then compatible Model Component identity may be included.

---

### Requirement: Generation Uses Model Component Graphs Safely

Generation SHALL be able to request prefill and decode graphs from a Model Component.

Generation semantics SHALL remain owned by Generation Contract.

#### Scenario: Stop condition

Given decode graph produces logits

When Generation receives logits

Then Generation still owns Sampling, stop conditions, and streaming semantics.

---

### Requirement: Adapter Compatibility Metadata

Model Component SHALL expose metadata needed for adapter validation where
adapters are supported.

#### Scenario: Adapter incompatible

Given adapter target module does not exist

When Runtime validates against Model Component metadata

Then Adapter Loading fails.

---

### Requirement: KV Cache Metadata

Model Component SHALL declare KV cache metadata where relevant.

#### Scenario: KV head count

Given architecture uses grouped-query attention

When KV cache metadata is requested

Then Model Component exposes KV head count and compatibility metadata.

---

### Requirement: Tokenizer Compatibility Metadata

Model Component SHALL declare tokenizer compatibility requirements.

#### Scenario: Vocabulary mismatch

Given Model Artifact expects vocabulary size X

And tokenizer reports Y

When compatibility validation runs

Then Runtime rejects or reports tokenizer-incompatible.

---

### Requirement: Quantization Compatibility Metadata

Model Component SHALL declare quantization compatibility where relevant.

#### Scenario: Unsupported quantization

Given Model Artifact uses unsupported quantization method

When Model Component validates metadata

Then validation fails with quantization-unsupported.

---

### Requirement: Browser-Compatible Model Component

Model Component Contract SHALL be platform-neutral and SHALL not require
Wasmtime or native Provider loading.

#### Scenario: Browser target

Given Runtime runs on browser target

When Wasmtime-only Model Component path is requested

Then Runtime returns browser-feature-unsupported or selects a browser-compatible
path.

---

### Requirement: Model Component Versioning

Model Component compatibility SHALL be versioned and negotiated.

#### Scenario: Graph contract mismatch

Given Model Component emits graph contract version 9

And Runtime supports version 1

When validation runs

Then Runtime rejects it with graph-contract-incompatible.

---

### Requirement: Model Component Conformance

Model Components SHALL be subject to conformance testing.

#### Scenario: Conformance failure

Given Model Component emits invalid graph for supported architecture

When conformance runs

Then Model Component fails the relevant conformance profile.

---

### Requirement: Model Component Error Categories

Model Component failures SHALL use structured error categories.

#### Scenario: Target module unavailable

Given target module metadata is missing

When Adapter Loading requires it

Then Runtime reports target-module-unavailable.

---

### Requirement: Model Component Observability

Runtime SHALL define Model Component observations.

Observability SHALL not expose raw model weights, prompts, adapter tensors, KV
cache contents, Provider handles, Device handles, Kernel handles, or memory
pointers by default.

#### Scenario: Graph produced

Given Model Component produces a graph

When Runtime emits observability

Then it records redacted graph production metadata.

### Requirement: First Baseline Model Component Uses Scoped Operators

A Model Component used for the first baseline SHALL declare only operators that
are implemented or explicitly allowed by first scope policy.

#### Scenario: Unsupported operator requirement

Given Model Component declares flash-attention as mandatory

When first baseline validates the Component

Then Runtime rejects it or requires a non-flash attention graph alternative.

---

### Requirement: Model Component May Provide Unfused Graph

For first baseline execution, Model Component SHALL be able to provide unfused
graphs using required-now operators.

#### Scenario: Unfused MLP

Given fused MLP is unavailable

When graph production is requested

Then Model Component may emit matmul, SiLU, mul, and matmul sequence.

---

### Requirement: Qwen May Be First Model Component Baseline

Model Component contract SHALL allow a Qwen-like decoder-only baseline as the
first concrete architecture implementation.

#### Scenario: First component

Given Runtime supports first baseline Components

When Qwen-compatible metadata is loaded

Then Qwen Model Component may satisfy architecture implementation.

---

### Requirement: Model Component Baseline Uses Portable Operators

First baseline Model Components SHALL use portable Operator identities.

#### Scenario: Provider-specific operator rejected

Given Qwen Component graph references `cuda.qwen_attention`

When Runtime validates graph

Then validation fails.

---

### Requirement: Model Component Baseline May Be Runtime-Native First

The first Model Component baseline SHALL be permitted to be Runtime-native
before a WASM Component implementation exists.

#### Scenario: Native Qwen implementation

Given WASM Component path is not implemented

When Runtime resolves Qwen support

Then Runtime may use native architecture implementation if policy allows.

---

### Requirement: Inference API Uses Model Components Through Runtime

Runtime Inference API SHALL resolve and use Model Components through Runtime validation.

#### Scenario: Qwen request

Given model reference resolves to Qwen-compatible artifact

When inference begins

Then Runtime resolves compatible Model Component before graph production.

---

### Requirement: Inference API Does Not Grant Component Extra Authority

Calling inference through Runtime API SHALL not grant Model Components additional authority.

#### Scenario: Component asks filesystem

Given Model Component attempts filesystem access during API request

When Runtime authorizes it

Then access is denied.

### Requirement: Model Components Interpret Normalized Architecture Metadata

Model Components SHALL interpret normalized architecture metadata rather than raw
external format files.

#### Scenario: Qwen config normalized

Given Hugging Face-style config is normalized

When Qwen Model Component validates architecture

Then it reads normalized metadata.

---

### Requirement: Format Parsers Do Not Produce Execution Graphs

Format parsers SHALL not produce authoritative execution graphs.

#### Scenario: Parser emits graph

Given format parser attempts to emit Provider-specific graph

When Runtime validates it

Then Runtime rejects or ignores graph behavior outside Model Component contract.

