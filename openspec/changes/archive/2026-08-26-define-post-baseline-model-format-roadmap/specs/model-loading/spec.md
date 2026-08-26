## ADDED Requirements

### Requirement: Model Loading Consumes Normalized Artifacts

Model Loading SHALL consume normalized Model Artifact metadata regardless of
source format.

#### Scenario: GGUF normalized

Given GGUF metadata is normalized into Model Artifact

When loading runs

Then Model Loading uses standard validation flow.

---

### Requirement: Model Loading Does Not Bypass Validation For Formats

Supported formats SHALL not bypass Model Loading validation.

#### Scenario: safetensors shortcut

Given safetensors parser succeeds

When loading runs

Then Model Loading still validates trust, integrity, tensor inventory, memory,
component compatibility, and policy.

---

### Requirement: Sharded Loading Validates Shards

Model Loading SHALL validate shard index, shard presence, digest, and tensor
mapping for sharded artifacts.

#### Scenario: Missing shard

Given shard index references missing file

When loading validates artifact

Then loading fails before Model Instance creation.