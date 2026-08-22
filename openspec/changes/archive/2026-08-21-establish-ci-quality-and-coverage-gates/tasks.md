# Tasks

## 1. Rust Toolchain

- [x] Add a repository-owned Rust toolchain configuration.
- [x] Pin the toolchain channel/version used by CI.
- [x] Include the `rustfmt`, `clippy`, and LLVM coverage prerequisites required
      by the repository.
- [x] Ensure local development and CI use compatible Rust toolchains.

## 2. GitHub Actions Structure

- [x] Create `.github/workflows/`.
- [x] Add a pull-request CI workflow.
- [x] Run the quality workflow for changes targeting the primary development
      branch.
- [x] Run the quality workflow for direct changes to the primary branch.
- [x] Configure workflow concurrency.
- [x] Cancel superseded executions for the same branch or pull request.
- [x] Use minimal GitHub token permissions.
- [x] Avoid requiring repository secrets for normal quality validation.

## 3. Formatting Gate

- [x] Add a stable CI job for Rust formatting.
- [x] Validate the complete workspace with `cargo fmt`.
- [x] Fail CI when committed Rust source is not formatted.
- [x] Document the equivalent local command.

## 4. Compilation Gate

- [x] Add a stable CI job for workspace compilation.
- [x] Check all workspace members.
- [x] Check all relevant targets.
- [x] Check all relevant feature combinations supported by the workspace.
- [x] Fail CI on compilation errors.

## 5. Clippy Gate

- [x] Add a stable CI job for Clippy.
- [x] Run Clippy across the workspace.
- [x] Include all relevant targets.
- [x] Treat Clippy warnings as CI failures.
- [x] Avoid repository-wide lint suppression solely to make CI pass.

## 6. Test Gate

- [x] Execute all existing workspace tests in CI.
- [x] Execute unit tests.
- [x] Execute integration tests when present.
- [x] Execute documentation tests when present.
- [x] Ensure test failures make the workflow fail.
- [x] Preserve deterministic test behavior where Runtime semantics require it.

## 7. Documentation Gate

- [x] Compile Rust documentation in CI.
- [x] Treat Rustdoc warnings as errors.
- [x] Build documentation without external dependencies where possible.
- [x] Detect broken public API documentation introduced by refactors.

## 8. WIT Validation

- [x] Add a CI validation step for every tracked Magnetar WIT package.
- [x] Parse all WIT packages using the selected repository WIT tooling.
- [x] Fail CI for syntactically invalid WIT.
- [x] Fail CI for unresolved WIT package/interface/world references.
- [x] Document the local WIT validation command.
- [x] Keep WIT validation independent from Provider implementation tests.

## 9. OpenSpec Validation

- [x] Validate active OpenSpec changes using repository-supported OpenSpec
      tooling where available.
- [x] Validate canonical OpenSpec specifications.
- [x] Fail CI for structurally invalid OpenSpec artifacts.
- [x] Keep archived changes immutable except for explicit repository migration.

## 10. Cross-Platform Validation

- [x] Define Linux CI coverage for the complete required quality gate.
- [x] Validate compilation and tests on Windows.
- [x] Validate compilation and tests on macOS.
- [x] Ensure platform-specific dynamic-library code remains buildable.
- [x] Avoid allowing one platform to silently diverge from the portable Runtime
      API.

## 11. Coverage Measurement

- [x] Add LLVM-based Rust code coverage tooling.
- [x] Measure workspace source coverage.
- [x] Generate a machine-readable coverage report.
- [x] Generate an LCOV-compatible coverage artifact.
- [x] Upload the coverage report as a CI artifact.
- [x] Exclude generated code and other explicitly justified non-source artifacts
      from coverage measurements.
- [x] Do not exclude difficult Runtime modules merely to increase the reported
      percentage.

## 12. Coverage Baseline

- [x] Measure the coverage of the repository at the introduction of this change.
- [x] Record the measured baseline in repository-owned configuration or data.
- [x] Record at least line coverage.
- [x] Record branch/function coverage when reliably supported by the chosen
      tooling.
- [x] Do not invent a target percentage unrelated to the measured repository.

## 13. Coverage Ratchet

- [x] Introduce a non-regression coverage policy.
- [x] Prevent pull requests from silently reducing protected coverage below the
      accepted baseline.
- [x] Allow the baseline to move upward as test coverage improves.
- [x] Require explicit review for intentional baseline reductions.
- [x] Keep threshold adjustments version controlled.
- [x] Ensure the ratchet compares equivalent coverage scopes.

## 14. Coverage Scope

- [x] Measure production Runtime source rather than test source as the primary
      coverage scope.
- [x] Include new Runtime modules automatically unless explicitly excluded.
- [x] Document every coverage exclusion.
- [x] Prevent new modules from bypassing coverage solely because the Runtime is
      modularized later.

## 15. CI Artifact and Reporting

- [x] Expose test failures directly in GitHub Actions.
- [x] Publish coverage artifacts for failed or successful coverage jobs where
      practical.
- [x] Make the measured line coverage visible in the workflow summary.
- [x] Report the accepted coverage baseline.
- [x] Report whether the current result passes the non-regression ratchet.

## 16. Stable Quality Status Checks

- [x] Define stable job names for formatting.
- [x] Define stable job names for compilation.
- [x] Define stable job names for Clippy.
- [x] Define stable job names for tests.
- [x] Define stable job names for WIT validation.
- [x] Define stable job names for coverage.
- [x] Keep required quality job names stable enough for GitHub branch
      protection.

## 17. CI Performance

- [x] Cache Cargo registry and build artifacts where safe.
- [x] Ensure caching cannot cause a stale successful result to replace actual
      validation.
- [x] Avoid running coverage redundantly on every operating-system matrix entry.
- [x] Run expensive checks only once when platform-independent.
- [x] Preserve cross-platform compilation and test coverage separately.

## 18. Security

- [x] Pin third-party GitHub Actions to reviewed versions or immutable revisions
      according to repository policy.
- [x] Use least-privilege workflow permissions.
- [x] Do not expose repository secrets to pull-request quality jobs.
- [x] Do not execute untrusted external Provider binaries as part of the normal
      quality gate.
- [x] Treat dynamically loaded test Providers as controlled test fixtures only.

## 19. Developer Documentation

- [x] Document all CI-equivalent local commands.
- [x] Document how to run formatting locally.
- [x] Document how to run Clippy locally.
- [x] Document how to run the complete test suite locally.
- [x] Document how to validate WIT locally.
- [x] Document how to generate local coverage.
- [x] Document how the coverage ratchet is updated.

## 20. Validation

- [ ] Verify the full CI workflow on a pull request.
- [ ] Verify a formatting violation fails CI.
- [ ] Verify a Clippy warning fails CI.
- [ ] Verify a failing unit test fails CI.
- [ ] Verify invalid WIT fails CI.
- [ ] Verify a coverage regression fails the coverage gate.
- [ ] Verify an improved coverage result can raise the baseline.
- [ ] Verify Windows, Linux, and macOS validation completes successfully.
