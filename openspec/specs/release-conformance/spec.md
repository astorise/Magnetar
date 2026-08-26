# release-conformance Specification

## Purpose
TBD - created by archiving change define-first-release-conformance-and-compatibility-gates. Update Purpose after archive.
## Requirements
### Requirement: First Release Conformance Gates

Magnetar SHALL define release-blocking conformance gates for the first stable
release.

#### Scenario: Release candidate

Given `v0.1` release candidate is prepared

When release validation runs

Then required conformance gates execute.

---

### Requirement: Stable Release Requires Required Gates

Stable release SHALL require all required gates to pass.

#### Scenario: Required gate fails

Given E2E local conformance fails

When stable release is attempted

Then release is blocked.

---

### Requirement: Source And Build Gates

Release SHALL pass source and build gates.

#### Scenario: Formatting failure

Given formatting gate fails

When release is attempted

Then release is blocked.

---

### Requirement: OpenSpec Gate

Release SHALL pass OpenSpec validation.

#### Scenario: Invalid OpenSpec

Given OpenSpec validation fails

When release is attempted

Then release is blocked.

---

### Requirement: WIT Gate

Release SHALL pass WIT validation for included WIT packages.

#### Scenario: Raw handle in WIT

Given WIT exposes raw Provider handle

When WIT gate runs

Then release is blocked.

---

### Requirement: Rust Public API Gate

Release SHALL audit public Rust APIs for scope and handle safety.

#### Scenario: Raw tensor pointer exported

Given public API exports raw tensor pointer

When API gate runs

Then release is blocked.

---

### Requirement: Runtime Contract Gate

Release SHALL validate Runtime inference-only boundary.

#### Scenario: Runtime shell access

Given Runtime exposes shell execution

When Runtime gate runs

Then release is blocked.

---

### Requirement: Runtime Inference API Gate

Release SHALL validate Runtime Inference API baseline behavior.

#### Scenario: Streaming API broken

Given streaming API fails required baseline test

When release gates run

Then release is blocked.

---

### Requirement: Reference CPU Gate

Release SHALL validate Reference CPU Provider baseline.

#### Scenario: CPU Provider missing

Given Reference CPU Provider does not register

When release gates run

Then release is blocked.

---

### Requirement: Operator First Scope Gate

Release SHALL validate required-now Operator coverage.

#### Scenario: Matmul missing

Given matmul required-now coverage is missing

When release gates run

Then release is blocked.

---

### Requirement: Tensor Memory Gate

Release SHALL validate Tensor Resource and Memory Manager baseline.

#### Scenario: Raw pointer exposed

Given Tensor Resource diagnostics expose raw pointer

When release gates run

Then release is blocked.

---

### Requirement: Kernel Registry Dispatch Gate

Release SHALL validate Kernel Registry and Dispatch baseline.

#### Scenario: Direct Provider bypass

Given E2E path bypasses Kernel Registry

When release gates run

Then release is blocked.

---

### Requirement: Model Loading Gate

Release SHALL validate Model Artifact and Model Loading baseline.

#### Scenario: Trust bypass

Given cached model loads without trust validation

When release gates run

Then release is blocked.

---

### Requirement: Qwen Baseline Gate

Release SHALL validate Qwen-like baseline behavior.

#### Scenario: QwenProvider introduced

Given QwenProvider exists

When release gates run

Then release is blocked.

---

### Requirement: Generation Sampling Gate

Release SHALL validate Generation and Sampling baseline.

#### Scenario: Stop condition ignored

Given generation ignores stop condition

When release gate runs

Then release is blocked.

---

### Requirement: CLI Boundary Gate

Release SHALL validate CLI/Runtime authority boundary.

#### Scenario: CLI network authority delegated

Given Runtime receives ambient CLI network authority

When release gate runs

Then release is blocked.

---

### Requirement: E2E Local Inference Gate

Stable release SHALL pass E2E local inference conformance.

#### Scenario: E2E shortcut

Given E2E success path bypasses Model Loading

When release validation runs

Then release is blocked.

---

### Requirement: Redaction Gate

Stable release SHALL pass redaction gates.

#### Scenario: Raw prompt leaked

Given release diagnostics log raw prompt by default

When redaction gate runs

Then release is blocked.

---

### Requirement: Security Supply-Chain Gate

Stable release SHALL pass release security and supply-chain gates.

#### Scenario: Secret detected

Given release artifact contains secret

When release gate runs

Then release is blocked.

---

### Requirement: Compatibility Matrix

Release SHALL publish a compatibility matrix.

#### Scenario: Provider ABI status

Given release notes are generated

When Provider ABI is listed

Then status is marked stable-for-v0.1-baseline, preview, experimental, unstable,
deferred, or unsupported.

---

### Requirement: Allowed Skips Are Explicit

Allowed skips SHALL be explicit and reported.

#### Scenario: CUDA skipped

Given CUDA is out of scope

When release report is generated

Then CUDA Provider conformance is skipped with reason.

---

### Requirement: Required Baseline Gates Cannot Be Skipped

Required baseline gates SHALL not be skipped for stable release.

#### Scenario: E2E skipped

Given E2E local conformance is skipped

When stable release is attempted

Then release is blocked.

---

### Requirement: Exceptions Are Documented

Release exceptions SHALL be documented.

#### Scenario: Undocumented exception

Given release exception exists without documentation

When release validation runs

Then release is blocked.

---

### Requirement: Release Reports

Release SHALL produce machine-readable and human-readable gate reports.

#### Scenario: Report generated

Given release gates complete

When reports are emitted

Then each gate has status, version, target, feature set, and redaction status.

---

### Requirement: Release Candidate Validation

Release candidates SHALL run required gates and clearly report known failures.

#### Scenario: RC with known failure

Given `v0.1.0-rc.1` has known failure

When release notes are read

Then it is marked pre-release and stable publication is blocked.

---

### Requirement: Stable Cutover Gate

Stable cutover SHALL occur only after all required gates pass and release
metadata is complete.

#### Scenario: Missing compatibility matrix

Given compatibility matrix is missing

When stable release is attempted

Then release is blocked.

### Requirement: Cutover Requires Release Gates

Cutover SHALL require all release conformance gates to pass or be validly
skipped as allowed.

#### Scenario: Disallowed skip

Given OpenSpec validation is skipped

When cutover runs

Then release is blocked.

---

### Requirement: Cutover Reports Gate Status

Cutover SHALL include gate status in release reports.

#### Scenario: Gate report

Given gates complete

When cutover generates reports

Then pass/fail/skipped/exception status is visible.

