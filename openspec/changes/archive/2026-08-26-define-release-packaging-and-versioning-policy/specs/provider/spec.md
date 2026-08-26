## ADDED Requirements

### Requirement: Provider Release Status

Release metadata SHALL declare Provider support status.

#### Scenario: Provider list

Given `v0.1` release notes are generated

When Provider support is listed

Then Reference CPU is included and CUDA/Metal/OpenVINO/QNN/WebGPU are deferred
or experimental.

---

### Requirement: Provider ABI Compatibility Status

Release metadata SHALL declare Provider ABI compatibility status.

#### Scenario: ABI status

Given Provider ABI is not stable

When release notes are generated

Then Provider ABI is marked unstable or experimental.

---

### Requirement: Provider Handles Remain Hidden In Release

Release public APIs SHALL not expose raw Provider, Device, Kernel, or native
framework handles.

#### Scenario: Release docs

Given Provider diagnostics are documented

When examples are inspected

Then raw native handles are absent.