## ADDED Requirements

### Requirement: Compilation Lifecycle Observability

Runtime SHALL make Provider compilation lifecycle observable through structured events.

#### Scenario: Compilation completes

Given Provider finishes compilation

When observation is emitted

Then request ID, artifact digest, compiler identity and duration may be
reported.

---

### Requirement: Source Contents Redacted

Raw kernel source SHALL be redacted by default.

#### Scenario: Compiler syntax error

Given compiler reports offending line

When diagnostic is exported

Then policy controls source excerpt and full source is not logged automatically.

---

### Requirement: Compiler Paths Redacted

Provider compiler temporary paths SHALL be redacted by default.

#### Scenario: NVCC reports temp file

Given compiler output contains `/tmp/...`

When diagnostic is exported

Then path is removed or normalized.

---

### Requirement: Compiler Environment Redacted

Environment variables and secrets SHALL not be emitted in compilation
observability.

#### Scenario: Compiler process inherits environment

Given Provider records compiler failure

When observation is emitted

Then environment contents are absent.

---

### Requirement: Native Compiler Handles Redacted

Native compiler/driver objects SHALL remain absent from diagnostics.

#### Scenario: Shader compiler object exists

Given Provider tracks native object internally

When diagnostic is emitted

Then only opaque Runtime metadata is exposed.