## ADDED Requirements

### Requirement: Provider Conformance In CI

CI SHALL run Provider conformance for at least one hardware-independent
Provider target.

#### Scenario: Core conformance regression

Given a change breaks Provider core conformance

When CI runs

Then the conformance job fails.

---

### Requirement: Feature-Profile Conformance

When a Provider advertises a feature profile, CI or targeted conformance SHALL
test that profile.

#### Scenario: Data movement advertised

Given a Provider advertises data movement support

When relevant conformance runs

Then data movement tests are included.

---

### Requirement: Conformance Report Artifacts

Provider conformance SHALL produce machine-readable report output.

#### Scenario: Failed Provider

Given a Provider fails conformance

When CI completes

Then diagnostics identify the failed profile and failed requirement.

---

### Requirement: Hardware-Independent Default

Required CI conformance SHALL not depend on real GPU hardware, vendor drivers,
Tachyon, or network access.

#### Scenario: CI runner without GPU

Given CI runner has no GPU

When required conformance runs

Then the required conformance profile still completes using hardware-independent
targets.

---

### Requirement: Optional Hardware Conformance

Hardware-specific conformance SHALL be optional and separately enabled.

#### Scenario: Local CUDA test

Given a developer enables CUDA conformance locally

When compatible hardware and drivers are present

Then CUDA-specific conformance may run outside default CI.

---

### Requirement: Non-Conformant Provider Visibility

Provider compatibility documentation SHALL identify conformance status.

#### Scenario: Experimental Provider

Given a Provider has not passed required conformance profiles

When it is documented

Then it is clearly marked experimental or non-conformant.
