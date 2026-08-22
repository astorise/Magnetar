## ADDED Requirements

### Requirement: Repository Continuous Integration

Magnetar SHALL execute automated repository quality validation for pull requests
and changes to the primary development branch.

CI SHALL validate source quality without requiring manual developer execution.

#### Scenario: Pull request validation

Given a pull request modifying Magnetar source

When the pull request CI workflow runs

Then the configured repository quality gates are evaluated

And a failing required gate prevents the workflow from reporting success.

---

### Requirement: Repository-Owned Rust Toolchain

Magnetar SHALL define the Rust toolchain required to validate the repository.

CI SHALL use a repository-defined compatible toolchain rather than relying on an
unspecified runner default.

#### Scenario: CI runner changes default Rust version

Given a GitHub Actions runner with a different default Rust version

When Magnetar CI runs

Then Magnetar uses the repository-defined Rust toolchain.

---

### Requirement: Formatting Quality Gate

CI SHALL validate Rust formatting across the complete workspace.

Formatting differences SHALL fail the formatting quality gate.

#### Scenario: Unformatted Rust source

Given a Rust source file that does not satisfy repository formatting rules

When the formatting quality gate runs

Then the gate fails.

---

### Requirement: Compilation Quality Gate

CI SHALL compile-check all Magnetar workspace members and relevant targets.

Compilation errors SHALL fail the compilation quality gate.

#### Scenario: Runtime API no longer compiles

Given a change introducing a Rust compilation error

When the compilation quality gate runs

Then the gate fails before the change is considered valid.

---

### Requirement: Clippy Quality Gate

CI SHALL execute Clippy for the Magnetar workspace.

Clippy warnings covered by repository policy SHALL be treated as errors.

The repository SHALL NOT globally suppress meaningful Clippy findings solely to
satisfy CI.

#### Scenario: New Clippy regression

Given a change introducing a Clippy warning covered by repository policy

When the Clippy quality gate runs

Then the gate fails.

---

### Requirement: Automated Test Gate

CI SHALL execute the repository's automated Rust tests.

The gate SHALL include unit tests and any integration or documentation tests
present in the workspace.

A failing required test SHALL fail CI.

#### Scenario: Existing behavior regresses

Given an existing automated test describing required Runtime behavior

And a change violates that behavior

When CI executes the test suite

Then the test gate fails.

---

### Requirement: Documentation Validation

CI SHALL validate Rust API documentation.

Documentation warnings covered by repository policy SHALL be treated as
failures.

#### Scenario: Public documentation becomes invalid

Given a change introducing invalid Rust documentation

When the documentation quality gate runs

Then the gate fails.

---

### Requirement: WIT Contract Validation

Every tracked Magnetar WIT package SHALL be syntactically and structurally
validated by CI.

The validator SHALL detect invalid packages, interfaces, worlds, and unresolved
references supported by the selected WIT tooling.

#### Scenario: Invalid Compute WIT

Given a change that makes `magnetar:compute` WIT invalid

When WIT validation runs

Then CI fails before the contract can be merged.

---

### Requirement: OpenSpec Validation

Magnetar SHALL automatically validate canonical and active OpenSpec artifacts
using the repository-supported OpenSpec validation mechanism.

#### Scenario: Invalid active change

Given an active OpenSpec change with invalid structure

When repository specification validation runs

Then the validation gate fails.

---

### Requirement: Cross-Platform Validation

Magnetar SHALL preserve buildability across its supported host development
platforms.

At minimum, CI SHALL validate the Runtime on:

- Linux
- Windows
- macOS

Platform-independent expensive analysis MAY execute on only one platform.

#### Scenario: Windows-only compilation regression

Given a change that compiles on Linux but breaks Windows dynamic-library code

When cross-platform CI runs

Then the Windows validation detects the regression.

---

### Requirement: Code Coverage Measurement

Magnetar SHALL automatically measure test coverage for production Rust source.

Coverage SHALL be generated using LLVM-compatible Rust coverage instrumentation
or an equivalent repository-approved mechanism.

#### Scenario: Test suite completes

Given a successful test execution with coverage enabled

When coverage processing completes

Then CI produces a machine-readable coverage result

And a human-inspectable coverage artifact.

---

### Requirement: Initial Coverage Baseline

The first enforced Magnetar coverage threshold SHALL originate from a measured
repository baseline.

The baseline SHALL NOT be an arbitrary aspirational percentage.

#### Scenario: Coverage gate is introduced

Given the current Magnetar test suite

When coverage gating is introduced

Then the repository measures the existing coverage

And records that measured value as the initial accepted baseline.

---

### Requirement: Coverage Non-Regression Ratchet

Magnetar SHALL apply a coverage ratchet.

A normal change SHALL NOT reduce protected coverage below the accepted
repository baseline.

Coverage improvements SHOULD allow the baseline to move upward.

#### Scenario: Pull request reduces coverage

Given an accepted line coverage baseline

And a pull request produces lower protected line coverage

When the coverage gate evaluates the result

Then the gate fails unless the baseline reduction is explicitly reviewed and
version controlled.

---

### Requirement: Version-Controlled Coverage Policy

Coverage thresholds, scopes, tolerances, and exclusions SHALL be represented by
repository-owned configuration or data.

CI-only hidden state SHALL NOT be the sole source of coverage policy.

#### Scenario: Baseline changes

Given a deliberate change to the accepted coverage baseline

When the baseline is updated

Then the modification appears in repository history.

---

### Requirement: Coverage Scope Integrity

Coverage SHALL primarily measure production Runtime implementation source.

New production modules SHALL be included automatically unless an explicit,
documented exclusion applies.

#### Scenario: Runtime is modularized

Given `lib.rs` is split into multiple Runtime modules

When coverage runs after the refactor

Then the new production modules remain part of the protected coverage scope.

---

### Requirement: Explicit Coverage Exclusions

Coverage exclusions SHALL be explicit and justified.

A module SHALL NOT be excluded merely because it is difficult to test.

#### Scenario: New exclusion

Given a production path is proposed for coverage exclusion

When the exclusion is committed

Then the repository records the exclusion and its rationale.

---

### Requirement: Quality Gate Stability

Required CI jobs SHALL expose stable status-check identities suitable for GitHub
branch protection.

Refactoring workflow implementation SHALL NOT unnecessarily rename required
status checks.

#### Scenario: Branch protection references a test gate

Given branch protection requires the stable Magnetar test status

When workflow internals are refactored

Then the required status name remains usable unless an intentional migration is
performed.

---

### Requirement: Workflow Concurrency

Magnetar CI SHALL cancel superseded executions for the same pull request or
development branch when their results are no longer relevant.

#### Scenario: Pull request receives a new commit

Given CI is running for an older commit

When a newer commit is pushed to the same pull request

Then obsolete work may be cancelled

And validation proceeds for the newest revision.

---

### Requirement: Least-Privilege CI

GitHub Actions workflows SHALL use only the permissions required for repository
quality validation.

Normal pull-request validation SHALL NOT require production secrets.

#### Scenario: External pull request

Given an untrusted pull request

When quality validation runs

Then the workflow can execute required source validation without exposing
repository secrets.

---

### Requirement: CI Cache Safety

CI SHALL cache Cargo dependencies and build artifacts where safe.

A cache hit SHALL NOT bypass required validation or cause stale output to be
accepted as a current successful result.

#### Scenario: Cached compilation artifacts exist

Given reusable Cargo cache data

When CI validates a new commit

Then required commands still execute against the current source tree.

---

### Requirement: Coverage Artifact Availability

Coverage CI SHALL produce an artifact suitable for local or external analysis.

The artifact SHOULD include LCOV or an equivalent interoperable representation.

#### Scenario: Developer investigates uncovered code

Given a completed coverage workflow

When the developer inspects workflow artifacts

Then a detailed coverage report is available.

---

### Requirement: Quality Gate Separation from Coverage Expansion

This change SHALL establish the quality infrastructure and current coverage
baseline.

It SHALL NOT require comprehensive testing of every Runtime failure mode,
Provider conformance rule, Component sandbox property, or concurrency race.

Those broader tests SHALL be introduced through dedicated follow-up changes.

#### Scenario: Existing coverage is below a future desired target

Given the measured initial coverage is below the project's long-term goal

When this change is completed

Then CI still uses the measured baseline

And later changes increase coverage through additional meaningful tests rather
than artificial threshold manipulation.

---

### Requirement: Local CI Reproducibility

The repository SHALL document local commands equivalent to the principal CI
quality gates.

#### Scenario: Developer validates before pushing

Given a developer has the repository toolchain installed

When they follow the documented local quality commands

Then they can execute formatting, linting, testing, WIT validation, and coverage
checks without requiring a GitHub Actions runner.
