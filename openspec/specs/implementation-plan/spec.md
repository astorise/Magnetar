# implementation-plan Specification

## Purpose
TBD - created by archiving change implementation-plan-runtime-baseline. Update Purpose after archive.
## Requirements
### Requirement: Runtime Baseline Implementation Plan

Magnetar SHALL define an implementation plan for the first executable Runtime
baseline.

#### Scenario: Plan exists

Given architecture changes are complete

When implementation begins

Then developers have an ordered baseline plan.

---

### Requirement: Implementation Order

Implementation SHALL proceed from lower-level Runtime contracts toward
higher-level inference API.

#### Scenario: Start implementation

Given no baseline code exists for Tensor Resources

When implementation begins

Then Tensor/Memory scaffolding is implemented before E2E inference.

---

### Requirement: PR Sequence

The implementation plan SHALL define a PR sequence from Runtime skeleton to E2E
local conformance.

#### Scenario: PR planning

Given baseline work is scheduled

When tasks are split

Then each PR has clear scope and acceptance gates.

---

### Requirement: No Shortcut Rule

Implementation SHALL not bypass core Runtime contracts in the E2E success path.

#### Scenario: Direct Provider call

Given E2E success path directly invokes Reference CPU Provider

When no-shortcut validation runs

Then implementation fails conformance.

---

### Requirement: Acceptance Criteria

Runtime baseline SHALL define acceptance criteria for modules, APIs, tests,
conformance, redaction, and E2E local inference.

#### Scenario: Baseline completion

Given implementation claims baseline complete

When acceptance gates run

Then all required checks pass.

---

### Requirement: Deferred Work Is Explicit

Work outside the baseline SHALL be explicitly deferred.

#### Scenario: CUDA request

Given CUDA Provider is requested during baseline

When plan is checked

Then CUDA is identified as deferred work.

---

### Requirement: CI Gates

Implementation SHALL define CI gates for formatting, checking, tests,
conformance, OpenSpec validation, and coverage.

#### Scenario: CI run

Given baseline PR is submitted

When CI runs

Then relevant gates execute without requiring GPU hardware.

---

### Requirement: CPU-Only Baseline

The implementation plan SHALL keep the first baseline CPU-only.

#### Scenario: GPU unavailable

Given CI has no GPU

When baseline conformance runs

Then required tests still run.

---

### Requirement: Redaction By Default

Implementation SHALL verify diagnostics and observability are redacted by
default.

#### Scenario: Prompt submitted

Given inference request includes prompt text

When observations are emitted

Then raw prompt text is absent by default.

---

### Requirement: Post-Baseline Roadmap

Implementation plan SHALL identify post-baseline work without adding it to the
baseline acceptance gate.

#### Scenario: Optimized kernel

Given SIMD matmul is planned

When baseline acceptance runs

Then SIMD optimization is not required.
