# conformance Specification

## Purpose
TBD - created by archiving change define-reference-cpu-provider-and-kernel-baseline. Update Purpose after archive.
## Requirements
### Requirement: Reference CPU Conformance Baseline

Reference CPU Provider SHALL provide or participate in conformance baselines for
supported Operators.

#### Scenario: Matmul conformance

Given Reference CPU matmul is implemented

When conformance runs

Then its output is validated against Operator semantics and tolerance profile.

---

### Requirement: Reference CPU Fixtures Avoid GPU Dependency

Reference CPU conformance fixtures SHALL not require external GPU hardware.

#### Scenario: CPU-only environment

Given tests run on CPU-only machine

When Reference CPU conformance executes

Then supported fixtures can run without GPU.

---

### Requirement: Reference CPU Can Compare Optimized Kernels

Any comparison of Reference CPU outputs against optimized Provider Kernels SHALL respect the declared tolerance profile for the Operator under test.
Reference CPU outputs MAY be used for such comparisons.

#### Scenario: CUDA comparison

Given CUDA matmul Kernel exists

When conformance compares outputs

Then Reference CPU output may be used as baseline if policy allows it.

### Requirement: First Scope Conformance Suite

Conformance SHALL include fixtures for each required-now Operator.

#### Scenario: Required operator fixture

Given `softmax` is required-now

When conformance suite runs

Then softmax fixtures are included.

---

### Requirement: First Scope Conformance Is CPU-Compatible

First scope conformance SHALL be runnable without external GPU hardware.

#### Scenario: CPU-only conformance

Given only Reference CPU Provider is available

When first scope conformance runs

Then supported required-now fixtures can execute.

---

### Requirement: First Scope Conformance Reports Placeholders

Placeholder Operators SHALL be reported as pending or unsupported rather than
passing silently.

#### Scenario: Placeholder conformance

Given `paged-attention` is placeholder

When conformance report is generated

Then it is reported as placeholder, pending, or unsupported.

### Requirement: Tensor Contract Conformance

Conformance SHALL validate Tensor Descriptor, Tensor Resource, Layout, DType, aliasing, views, Resource Affinity, and metadata safety behavior.

#### Scenario: Raw pointer exposure

Given Tensor metadata is returned during conformance

When result is inspected

Then no raw pointer or native handle is exposed.

---

### Requirement: Reference CPU Tensor Conformance

Reference CPU conformance SHALL validate host contiguous Tensor Resource support for required-now operators.

#### Scenario: CPU tensor conformance

Given host contiguous f32 tensors

When Reference CPU matmul conformance runs

Then tensor metadata and output readiness are validated.

---

### Requirement: Qwen Baseline Conformance

Conformance SHALL include Qwen baseline fixtures for config validation, tensor
inventory, graph production, operator scope, tokenizer compatibility, KV cache
metadata, adapter metadata, quantization rejection, authority, and handle
safety.

#### Scenario: Qwen conformance

Given Qwen Component claims baseline support

When conformance runs

Then it must pass Qwen baseline fixtures.

---

### Requirement: Qwen Baseline CPU Smoke Conformance

Conformance SHALL define a CPU smoke path requirement for a minimal Qwen-like
graph, which SHOULD run where all required Reference CPU kernels exist.

#### Scenario: CPU smoke graph

Given minimal Qwen-like fixture graph

When Reference CPU executes it

Then conformance validates graph planning, dispatch, and output metadata.

---

### Requirement: CLI Boundary Conformance

Conformance SHALL validate that `magnetar-cli` and Runtime preserve the
inference boundary.

#### Scenario: File access boundary

Given CLI reads file content for prompt

When Runtime receives request

Then Runtime has no filesystem authority.

---

### Requirement: Runtime Does Not Execute CLI-Owned Capabilities

Conformance SHALL validate Runtime does not execute tools, shell, Git, network,
or workspace operations.

#### Scenario: Generated shell text

Given model output contains shell command text

When Runtime emits output

Then no process execution occurs.

---

### Requirement: CLI Preserves Runtime Structured Errors

Conformance SHALL validate CLI preserves Runtime structured error categories
when displaying or wrapping errors.

#### Scenario: Runtime model loading error

Given Runtime returns model-loading-failed

When CLI displays failure

Then structured category is preserved.

---

### Requirement: Local Inference Conformance Suite

Conformance SHALL include a local inference suite that validates the full
Runtime inference path.

#### Scenario: Run local suite

Given conformance is executed

When local inference suite runs

Then the suite validates complete Runtime inference behavior.

---

### Requirement: E2E Conformance Uses Normal Runtime Contracts

E2E conformance SHALL use normal Runtime contracts and SHALL NOT use hidden
shortcuts.

#### Scenario: Shortcut detected

Given test path bypasses Model Loading

When conformance validates the path

Then the suite fails.

---

### Requirement: E2E Conformance Report

Conformance SHALL include E2E report output in machine-readable form.

#### Scenario: Report included

Given E2E suite completes

When conformance results are collected

Then E2E report is included with structured pass/fail/skipped status.

---

### Requirement: E2E Conformance Closes Baseline

E2E local inference conformance SHALL be the closing gate for the Runtime
baseline implementation.

#### Scenario: Baseline completion

Given implementation claims first baseline complete

When conformance runs

Then E2E local inference conformance must pass.

---

### Requirement: Conformance Runs Without GPU

Baseline conformance SHALL run without GPU hardware.

#### Scenario: CPU-only CI

Given CI has CPU only

When baseline conformance runs

Then required conformance suites can execute.

---

### Requirement: Conformance Detects Shortcuts

Conformance SHALL detect shortcuts that bypass Runtime contracts.

#### Scenario: Memory bypass

Given Provider writes output without Memory Manager tracking

When conformance validates output metadata

Then conformance fails.

---

### Requirement: Post-Baseline Provider Conformance

Conformance SHALL support Provider-specific profiles for optimized and
hardware-specific Providers.

#### Scenario: CUDA conformance

Given CUDA Provider is available

When CUDA conformance profile runs

Then it validates Provider, Kernel, Tensor, Memory, Operator, and observability
contracts.

---

### Requirement: Reference Comparison

Any comparison of optimized Provider output against Reference CPU fixtures SHALL use a declared tolerance profile.
Optimized Provider conformance MAY compare outputs against Reference CPU
fixtures within declared tolerance.

#### Scenario: Optimized matmul comparison

Given optimized matmul output is produced

When compared to Reference CPU output

Then difference must be within tolerance.

---

### Requirement: Benchmark Separation

Benchmarks SHALL be reported separately from correctness conformance.

#### Scenario: Benchmark fast but wrong

Given benchmark passes performance target

But correctness conformance fails

Then Provider is not accepted as conformant.

### Requirement: Model Format Conformance

Conformance SHALL include fixtures for supported model formats.

#### Scenario: safetensors conformance

Given safetensors support is enabled

When conformance runs

Then valid and invalid safetensors fixtures are validated.

---

### Requirement: Model Format Conformance Uses Normalized Artifacts

Format conformance SHALL validate that parsed files normalize into Model
Artifact, Tokenizer Artifact, or Adapter Artifact contracts.

#### Scenario: tokenizer.json conformance

Given tokenizer.json fixture is parsed

When conformance validates it

Then normalized Tokenizer Artifact metadata is produced.

---

### Requirement: Format Conformance Checks Redaction

Format conformance SHALL validate redaction of raw weights, tokenizer data,
file contents, handles, pointers, and secrets.

#### Scenario: Parser error

Given format parser fails

When diagnostics are emitted

Then raw file contents are not logged by default.

### Requirement: Source Cache Conformance

Conformance SHALL validate model source and cache behavior.

#### Scenario: Cache hit validation

Given cached artifact exists

When conformance loads it

Then trust, integrity, format, and loading validations still run.

---

### Requirement: Source Cache Boundary Conformance

Conformance SHALL validate Runtime does not gain arbitrary filesystem, network,
credential, or cache mutation authority.

#### Scenario: Arbitrary directory scan

Given Runtime is asked to scan arbitrary model directory

When conformance runs

Then request is denied.

---

### Requirement: Cache Residency Conformance

Conformance SHALL validate cache presence is distinct from memory residency.

#### Scenario: Cached but not loaded

Given artifact is cached but not loaded

When Memory Manager is inspected

Then no model tensors are resident.

### Requirement: Server API Conformance

Conformance SHALL validate Server API boundaries and Runtime API usage.

#### Scenario: Server conformance

Given server API implementation exists

When conformance runs

Then server requests use Runtime Inference API and preserve redaction.

---

### Requirement: Server Boundary Conformance

Conformance SHALL validate server does not read arbitrary files, execute tools,
execute shell/processes, execute Git, or download arbitrary models during
generation.

#### Scenario: Server filesystem violation

Given generation request asks server to read arbitrary file

When conformance runs

Then request is denied.

---

### Requirement: Server Streaming Conformance

Conformance SHALL validate server streaming preserves Runtime event ordering and
redaction.

#### Scenario: Stream order

Given Runtime emits ordered generation events

When server streams them

Then order and redaction are preserved.

