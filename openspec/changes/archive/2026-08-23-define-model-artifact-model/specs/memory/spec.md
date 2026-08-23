## ADDED Requirements

### Requirement: Memory Manager Evaluates Model Artifact Feasibility

The Runtime Memory Manager SHALL evaluate Model Artifact loading feasibility
before model data becomes resident.

Feasibility SHALL consider storage dtype, compute dtype requirements,
quantization workspace, sharding metadata, adapter residency placeholders,
placement constraints, transfer staging, and memory pressure.

#### Scenario: Model loading exceeds memory policy

Given a Model Artifact requires more memory than current policy permits

When Runtime requests loading feasibility

Then the Memory Manager rejects the load with a structured feasibility failure.

---

### Requirement: Memory Manager Distinguishes Artifact Bytes From Residency

The Runtime Memory Manager SHALL distinguish persisted Model Artifact bytes from
resident model memory.

Resident model memory MAY differ from artifact bytes due to decompression,
dtype conversion, quantization workspace, sharded placement, provider-owned
buffers, or Runtime-owned staging.

#### Scenario: Quantized weights require workspace

Given Model Artifact bytes are stored in a quantized dtype

When Memory Manager plans residency

Then the plan accounts for compute-ready memory and dequantization workspace
separately from compressed storage bytes.
