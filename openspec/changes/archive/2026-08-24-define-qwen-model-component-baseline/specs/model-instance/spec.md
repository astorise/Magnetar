## ADDED Requirements

### Requirement: Qwen Model Instance References Qwen Component

A Qwen Model Instance SHALL reference the Qwen Component or native architecture
implementation used to validate and execute it.

#### Scenario: Qwen instance metadata

Given Qwen Model Instance is ready

When Runtime reports metadata

Then Qwen Component identity may be included.

---

### Requirement: Qwen Component Metadata Participates In Cache Compatibility

Qwen Component version and config fingerprint SHALL be permitted to participate
in KV Cache and Prefix Cache compatibility.

#### Scenario: Component version changed

Given Prefix Cache entry was produced under Qwen Component version A

When version B changes graph semantics

Then Runtime rejects reuse unless compatibility is proven.