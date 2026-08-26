## ADDED Requirements

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