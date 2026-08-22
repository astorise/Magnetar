# Establish CI Quality and Coverage Gates

## Why

Magnetar has accumulated a substantial Runtime implementation through the
successive Provider, Capability, Device, Resource Affinity, Compute, Memory
Planning, Scheduler, Provider Execution, Health, and Observability changes.

The repository currently relies primarily on local Rust unit tests and does not
have a repository-level continuous integration contract ensuring that changes
remain buildable, formatted, lint-clean, tested, WIT-valid, and measurable for
test coverage.

The upcoming architecture recadrage will modify foundational Runtime contracts,
remove legacy concepts, modularize the Runtime, introduce a real WebAssembly
Component Runtime, and later integrate production AI execution Providers.

These changes must not proceed without automated regression protection.

Magnetar therefore needs a stable repository quality contract before further
architectural refactoring.

The initial quality gate SHALL measure existing coverage rather than invent an
arbitrary coverage percentage. Coverage SHALL subsequently follow a ratchet
model so that the repository cannot silently regress while stronger tests are
added progressively.

## What Changes

Introduce repository-wide CI and quality requirements for Magnetar.

The change defines:

- GitHub Actions workflows for pull requests and the main branch
- a repository-owned Rust toolchain definition
- formatting validation
- workspace compilation validation
- Clippy validation with warnings treated as errors
- unit and integration test execution
- documentation compilation with warnings treated as errors
- WIT contract validation
- OpenSpec validation where supported by the repository tooling
- test coverage measurement using LLVM-based Rust coverage
- machine-readable and human-consumable coverage artifacts
- an initial measured coverage baseline
- a non-regression coverage ratchet
- deterministic CI job names suitable for branch protection
- cross-platform validation on Linux, Windows, and macOS
- bounded GitHub Actions permissions
- cancellation of superseded workflow executions
- dependency and build caching that does not change correctness
- explicit separation between quality gates and later test-coverage expansion

This change SHALL NOT attempt to reach an arbitrary high coverage target.

This change SHALL NOT introduce large new functional test suites for every
Runtime subsystem. Broader contract, failure, concurrency, and conformance
testing belongs to later dedicated changes.

## Impact

Magnetar changes become continuously validated before merging.

Architecture refactors will have immediate feedback for compilation failures,
API breakage, formatting errors, Clippy regressions, test failures, invalid WIT
contracts, and coverage regressions.

Coverage becomes measurable and progressively improvable rather than an
informal objective.

The CI contract also creates stable status checks that can later be configured
as required GitHub branch-protection checks.

The change does not alter Runtime execution semantics, Provider resolution,
Resource Affinity, Compute execution, Scheduler behavior, or Component
contracts.