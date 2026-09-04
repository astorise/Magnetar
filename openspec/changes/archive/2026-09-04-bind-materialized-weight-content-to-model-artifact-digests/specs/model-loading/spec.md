## MODIFIED Requirements

### Requirement: Weight Materialization Sources Real Artifact Bytes

Model Loading's weight-materialization phase SHALL be able to construct materialized tensor data from real Model Artifact bytes, using a format parser's generic tensor inventory to locate each tensor's byte range, not only from a pre-materialized in-memory source.

The construction step SHALL depend only on generic Model Artifact types, never on a concrete format parser crate.

When a tensor's inventory entry declares a content digest (see `model-artifact`'s "Tensor Content Digest Binding"), the weight-materialization transaction SHALL verify that the tensor data actually supplied for materialization hashes to that declared digest before admitting or writing it, and SHALL reject the attempt otherwise. A tensor whose inventory entry declares no digest is not subject to this check.

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

#### Scenario: Tensor content matching its declared digest is accepted

Given a tensor's inventory entry declares a content digest

When the data supplied for materialization hashes to that declared digest

Then the weight-materialization transaction admits and writes it normally.

#### Scenario: Tensor content not matching its declared digest is rejected

Given a tensor's inventory entry declares a content digest

When the data supplied for materialization does not hash to that declared digest

Then the weight-materialization transaction rejects the attempt before admission or write, and the affected weight resource is not bound to the Model Instance.

#### Scenario: A tensor with no declared digest is unaffected

Given a tensor's inventory entry declares no content digest

When any data is supplied for materialization under that tensor's name

Then the weight-materialization transaction does not reject it on content-digest grounds.
