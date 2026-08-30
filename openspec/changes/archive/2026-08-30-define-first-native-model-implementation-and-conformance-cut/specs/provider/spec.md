## ADDED Requirements
### Requirement: Reference CPU Provider Has Conformance Gate

Reference CPU Provider SHALL pass its required conformance profile before full
Qwen E2E is considered valid.

#### Scenario: Provider executes Kernel but ignores preparation

Given direct function execution works

When PreparedKernel lifecycle is not honored

Then Provider gate fails.

### Requirement: Provider Registration Failure Is Fatal To Relevant Setup

Mandatory Kernel registration errors SHALL not be silently discarded.

#### Scenario: Duplicate invalid Kernel registration

Given registration fails

When Provider initializes profile

Then failure is surfaced and required profile cannot become ready.

### Requirement: Synchronous Provider Still Uses Completion Contract

Reference CPU Provider SHALL be allowed to execute synchronously while preserving
CompletionToken semantics.

#### Scenario: Kernel returns inline

Given execution succeeds

When Provider returns

Then associated logical completion is terminal-successful.

### Requirement: Provider Has No Qwen Forward Function

Reference CPU Provider SHALL remain model-family neutral.

#### Scenario: Public Provider surface inspected

Given first Qwen model is supported

Then no mandatory `execute_qwen()` API exists.
