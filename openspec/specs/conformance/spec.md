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

