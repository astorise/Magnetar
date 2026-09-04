## MODIFIED Requirements

### Requirement: Provider Preferences Are Non-Authoritative

Provider and Device preferences in API requests SHALL be policy inputs only. Runtime SHALL own Provider and Device selection, except at Model Instance creation time when the loading phase resolved no provider or device binding, which is a documented limitation (see `model-instance`'s "Model Instance References Architecture Implementation" requirement) rather than a second authority model: today's implementation applies the caller's Resource Affinity directly as effective placement in that specific case, with no Runtime-side arbitration step, until instance-creation-time resolution is implemented.

#### Scenario: Caller requests CUDA

Given caller prefers CUDA

When Runtime resolves execution

Then Runtime may consider preference but SHALL not violate Resource Affinity,
capability, memory, readiness, or policy constraints.

---
