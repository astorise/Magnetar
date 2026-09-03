## ADDED Requirements

### Requirement: Weight Materialization Sources Real Artifact Bytes

Model Loading's weight-materialization phase SHALL be able to construct materialized tensor data from real Model Artifact bytes, using a format parser's generic tensor inventory to locate each tensor's byte range, not only from a pre-materialized in-memory source.

The construction step SHALL depend only on generic Model Artifact types, never on a concrete format parser crate.

#### Scenario: Materialize from a real Safetensors file

Given a real `.safetensors` file's bytes and its parsed generic tensor inventory

When weight materialization runs

Then it reads each tensor's declared byte range from the file bytes and produces the same materialized tensor data structure the existing in-memory materialization path produces

And no format-specific type crosses into the materialization step itself.

#### Scenario: Unsupported storage dtype is rejected structurally

Given a tensor's declared storage dtype is not one the Runtime's host tensor representation supports

When weight materialization attempts to read it

Then it returns a structured error rather than silently reinterpreting the bytes.

#### Scenario: Real and in-memory materialization agree

Given the same logical weights are available both as an in-memory source and as real artifact bytes

When both are materialized independently

Then they produce equal tensor data.
