## ADDED Requirements

### Requirement: CLI Model Resolution Does Not Bypass Loading

CLI-friendly names, aliases, or paths SHALL not bypass Model Loading validation.

#### Scenario: Alias load

Given CLI alias resolves to model reference

When Runtime loads it

Then Model Loading validates artifact, trust, component, memory, provider, and
policy.

---

### Requirement: CLI Local Paths Become Authorized Sources

If CLI supports local model paths, it SHALL convert them to authorized artifact
source references before Runtime loading.

#### Scenario: Local model

Given user provides local path

When CLI calls Runtime

Then Runtime receives client-provided artifact source reference and still
validates it.