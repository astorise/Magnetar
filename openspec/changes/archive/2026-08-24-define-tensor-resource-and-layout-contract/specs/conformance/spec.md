## ADDED Requirements

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