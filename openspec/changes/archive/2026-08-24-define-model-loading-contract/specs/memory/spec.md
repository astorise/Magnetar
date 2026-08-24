## ADDED Requirements

### Requirement: Memory Manager Owns Model Residency

Memory Manager SHALL track loaded model residency.

Model residency SHALL include placement, size, dtype, ownership, pressure, and
Resource Affinity metadata where applicable.

#### Scenario: Model loaded on Device

Given weights are materialized on Device memory

When residency is recorded

Then Memory Manager tracks Device residency and associated Resource Affinity.

---

### Requirement: Memory Manager Evaluates Loading Feasibility

Memory Manager SHALL evaluate model loading feasibility before materialization.

#### Scenario: Model too large

Given model loading requires more memory than policy permits

When feasibility is evaluated

Then Memory Manager rejects, queues, or delays loading according to policy.

---

### Requirement: Memory Manager Releases Model Residency

Memory Manager SHALL release model residency memory when unload policy requires
it.

#### Scenario: Unload releases memory

Given a loaded model owns Device memory

When Runtime unloads the model

Then Memory Manager releases associated memory records.

---

### Requirement: Memory Manager Accounts For Quantization Transform Workspace

Memory Manager SHALL account for temporary workspace required by quantization,
dequantization, or Provider-specific model transforms.

#### Scenario: Dequantization workspace

Given INT8 weights must be converted to BF16 during loading

When loading is planned

Then Memory Manager accounts for temporary BF16 workspace.

---

### Requirement: Memory Manager Supports Pending Model Loading

Memory Manager SHALL support policy-controlled queuing for model loading allocations.

#### Scenario: Loading queued

Given memory pressure prevents immediate allocation

And policy permits waiting

When model loading requests memory

Then Memory Manager may place the loading request in a pending allocation queue.
