# e2e-conformance Specification

## Purpose
Define the End-to-End Local Inference Conformance suite that proves the
complete Magnetar local inference path — model resolution, Model Loading,
Model Instance lifecycle, session creation, tokenization, generation
(prefill/decode), Sampling, streaming, Execution Graph production and
validation, Kernel Registry selection, Reference CPU Provider dispatch, Tensor
Resource handling, Memory Manager participation, CLI boundary enforcement,
diagnostics/observability redaction, and cleanup — works together through
normal Runtime contracts, without GPU hardware, network access, Tachyon,
external model downloads, or shell/process/tool/Git/workspace execution, so
that this suite becomes the correctness gate for the first executable
Magnetar baseline.
## Requirements
### Requirement: End-to-End Local Inference Conformance

Magnetar SHALL define an End-to-End Local Inference Conformance suite that
validates the complete local inference path.

#### Scenario: E2E success path

Given a valid local fixture model

When the E2E suite runs

Then Runtime resolves, loads, instantiates, tokenizes, generates, streams, and
cleans up through normal contracts.

---

### Requirement: E2E Local-Only Scope

The first E2E suite SHALL run without GPU hardware, network access, Tachyon,
external model downloads, shell/process execution, Git access, workspace
scanning, or tool execution.

#### Scenario: CPU-only machine

Given the suite runs on CPU-only CI

When E2E conformance executes

Then supported tests complete without GPU hardware.

---

### Requirement: E2E Fixture Model

The suite SHALL define a minimal deterministic Qwen-like decoder-only fixture
model.

#### Scenario: Fixture loaded

Given fixture artifact is loaded

When Model Loading validates it

Then it passes normal Model Artifact and Qwen baseline validation.

---

### Requirement: E2E Fixture Tokenizer

The suite SHALL include a tokenizer fixture using the Tokenizer Contract.

#### Scenario: Tokenize fixture prompt

Given fixture prompt text

When tokenization runs

Then deterministic token IDs are produced.

---

### Requirement: E2E Required Path

The suite SHALL validate model resolution, model loading, Model Instance
creation, session creation, tokenization, generation, prefill, decode, Sampling,
streaming, result, session close, and cleanup.

#### Scenario: Required path completed

Given fixture request is valid

When generation completes

Then result and usage metadata are returned and resources are cleaned up.

---

### Requirement: E2E No Shortcut Rule

The suite SHALL fail if inference bypasses required Runtime contracts.

#### Scenario: Direct Provider shortcut

Given implementation directly invokes Reference CPU Provider without Kernel
Registry

When E2E no-shortcut validation runs

Then the suite fails with e2e-boundary-violation.

---

### Requirement: E2E Reference CPU Path

The first E2E suite SHALL execute through Reference CPU Provider selected by
Kernel Registry and Runtime Dispatch.

#### Scenario: CPU selected

Given Reference CPU policy is enabled

When graph execution runs

Then Reference CPU kernels are selected through Kernel Registry.

---

### Requirement: E2E Operator Coverage

The suite SHALL exercise required-now Operators for the first decoder baseline.

#### Scenario: Operator coverage report

Given success path completes

When E2E report is generated

Then embedding, RMSNorm, matmul, RoPE, attention, softmax, SiLU, add, mul, and
residual-add coverage is reported.

---

### Requirement: E2E Graph Validation

The suite SHALL validate Model Component-produced prefill and decode graphs
before execution.

#### Scenario: Invalid graph fixture

Given invalid graph fixture is used

When Runtime validates it

Then E2E expects graph validation failure.

---

### Requirement: E2E Generation Validation

The suite SHALL validate generation parameters, stop behavior, deterministic
sampling, finish reason, usage accounting, cancellation, and streaming event
sequence.

#### Scenario: Max new tokens reached

Given max new tokens is 1

When generation runs

Then generation stops with expected finish reason.

---

### Requirement: E2E Sampling Validation

The suite SHALL validate Sampling Contract usage.

The initial success path SHOULD use greedy deterministic sampling.

#### Scenario: Greedy sample

Given logits are produced

When Sampling runs

Then deterministic next token is selected.

---

### Requirement: E2E Streaming Validation

The suite SHALL validate ordered streaming events.

#### Scenario: Stream ordered

Given generation runs with streaming enabled

When events are collected

Then prefill events occur before decode-token events and completion occurs last.

---

### Requirement: E2E Session Validation

The suite SHALL validate session lifecycle and cleanup.

#### Scenario: Closed session

Given session was closed

When generation is requested with the closed session

Then Runtime returns session-closed or equivalent structured error.

---

### Requirement: E2E KV Cache Validation

If the suite includes KV cache validation, raw KV cache contents SHALL not be
exposed.

#### Scenario: KV cache redacted

Given KV cache is used during decode

When diagnostics are returned

Then raw cache contents are absent.

---

### Requirement: E2E Tensor Validation

The suite SHALL validate Tensor Descriptor, Tensor Resource, dtype, layout,
readiness, no raw pointer exposure, output metadata, and cleanup.

#### Scenario: Tensor output

Given CPU matmul produces output

When dispatch completes

Then Tensor Resource readiness is updated and no raw pointer is exposed.

---

### Requirement: E2E Memory Validation

The suite SHALL validate Memory Manager participation and cleanup.

#### Scenario: Resource cleanup

Given generation completes and session closes

When cleanup runs

Then inference-scoped resources are released or retained only according to
policy.

---

### Requirement: E2E CLI Boundary Validation

The suite SHALL include a CLI-boundary case verifying Runtime receives explicit
prompt/context and no ambient CLI authority.

#### Scenario: Runtime file access denied

Given Runtime is asked to read workspace file during E2E

When request is validated

Then Runtime rejects the request.

---

### Requirement: E2E Diagnostics And Redaction

The suite SHALL validate diagnostics and observability redaction defaults.

#### Scenario: Redacted diagnostics

Given generation fails

When diagnostics are reported

Then raw prompt, raw weights, raw tensor values, raw cache contents, handles,
memory pointers, and secrets are not exposed.

---

### Requirement: E2E Failure Cases

The suite SHALL include structured failure cases for invalid model reference,
untrusted artifact, incompatible tokenizer, unsupported operator, missing
kernel, invalid tensor shape, memory admission failure, closed session,
cancellation, timeout, policy denial, raw handle access denial, Runtime file
access denial, and Runtime tool execution denial.

#### Scenario: Unsupported operator

Given graph uses unsupported operator

When E2E failure test runs

Then Runtime reports structured unsupported operator error.

---

### Requirement: E2E Determinism

The first E2E success path SHALL be deterministic where feasible.

#### Scenario: Deterministic output

Given deterministic fixture weights and greedy sampling

When success path runs twice

Then output tokens or structured output metadata match.

---

### Requirement: E2E Report Format

The E2E suite SHALL produce a machine-readable report without sensitive raw
values by default.

#### Scenario: Report generated

Given suite completes

When report is written

Then it contains suite, fixture, runtime, provider, operator, kernel, test,
redaction, and duration metadata.

---

### Requirement: E2E CI Integration

CI SHALL be able to run the first E2E suite without GPU hardware, and the
suite SHOULD remain lightweight.

#### Scenario: CI run

Given CI environment has no GPU

When E2E local inference conformance runs

Then supported E2E tests pass or skip only explicitly unsupported optional cases.

---

### Requirement: E2E Browser And Tachyon Boundary

The E2E local suite SHALL not require browser execution or Tachyon.

#### Scenario: Tachyon unavailable

Given Tachyon is unavailable

When local E2E conformance runs

Then tests do not require Tachyon.

---

### Requirement: E2E Error Categories

E2E conformance failures SHALL use structured error categories.

#### Scenario: Determinism failed

Given repeated success path outputs differ unexpectedly

When E2E validates determinism

Then it reports e2e-determinism-failed.

---

### Requirement: E2E Observability

Runtime observations for E2E conformance SHALL NOT expose raw prompts,
weights, tensors, cache contents, handles, paths, secrets, or memory pointers.

#### Scenario: E2E report emitted

Given E2E report is generated

When observability records it

Then only redacted report metadata is emitted.

### Requirement: Authoritative E2E Uses Production Runtime Path
The authoritative first-native E2E suite SHALL instantiate production Runtime APIs and SHALL NOT execute the model through a separate harness engine.

#### Scenario: Same path as CLI
- **WHEN** authoritative first-native E2E runs generation
- **THEN** it exercises the same RuntimeInferenceApi path used by CLI generation.

### Requirement: E2E Evidence Is Observational
First-native E2E evidence SHALL be collected from bounded observations emitted by the Runtime layers that actually executed work.

#### Scenario: Provider evidence comes from provider submission
- **WHEN** E2E asserts Provider execution
- **THEN** the assertion is based on Provider submission and completion observations emitted by the Provider/dispatch path.

#### Scenario: Evidence cannot be self-declared
- **WHEN** a test helper sets only a boolean claiming a layer executed
- **THEN** that value is insufficient for authoritative E2E conformance.

### Requirement: E2E Proves No First-Native Shortcuts
The authoritative first-native E2E suite SHALL fail if model execution bypasses Component validation, graph validation, PreparedExecutionPlan, Kernel Registry, Provider dispatch, Runtime-owned KV, or Sampling.

#### Scenario: Shortcut removed
- **WHEN** any required layer is disabled or unavailable
- **THEN** E2E fails with a structured error rather than silently falling back.

