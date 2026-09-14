## MODIFIED Requirements

### Requirement: Repository Continuous Integration

Magnetar SHALL execute automated repository quality validation for pull requests
and changes to the primary development branch.

CI SHALL validate source quality without requiring manual developer execution.

Every first-party crate SHALL be covered by CI, including a crate moved out of the repository root's `[workspace]` because it depends on externalized submodules -- such a crate SHALL be checked out with those submodules and built, tested, and linted by a CI job, not silently left untested because it no longer participates in the root workspace's own plain `cargo check`/`clippy`/`test`/`fmt --all` invocations.

#### Scenario: Pull request validation

Given a pull request modifying Magnetar source

When the pull request CI workflow runs

Then the configured repository quality gates are evaluated

And a failing required gate prevents the workflow from reporting success.

#### Scenario: A first-party crate outside the root workspace is still covered

Given a first-party crate moved out of the repository root's `[workspace]` because it depends on one or more externalized submodules

When CI runs

Then a job checks that crate out together with the submodules it depends on and builds, tests, and lints it

And that crate is not silently absent from every CI job merely because it does not participate in the root workspace
