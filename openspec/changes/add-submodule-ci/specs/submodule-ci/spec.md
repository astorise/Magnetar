## ADDED Requirements

### Requirement: Module Repositories Run Their Own Tests

Each Component, Provider, Format and Loader repository SHALL run its own formatting, Clippy, test and dependency checks on every pull request and on every push to its default branch. Tests SHALL run with the repository's committed lockfile.

#### Scenario: Pull request in a Provider repository

- **WHEN** a pull request is opened or updated in a Provider repository
- **THEN** the Provider's formatting, Clippy, test and dependency checks run and report on that pull request

#### Scenario: Failing test

- **WHEN** a module change makes one of the module's tests fail
- **THEN** the pull request shows a failed check before it is merged

#### Scenario: Stale lockfile

- **WHEN** the Runtime's dependency set has changed and the module's lockfile no longer matches
- **THEN** the module's check fails and names the lockfile as the cause

### Requirement: One Definition Of The Module Checks

The checks SHALL be defined once, in a reusable workflow in the Magnetar repository, and called by each module repository. Third-party actions used by that workflow SHALL be pinned by commit.

#### Scenario: Changing a check

- **WHEN** a check is added to or changed in the reusable workflow
- **THEN** every module repository runs the new definition on its next run, without an edit in the module repository

### Requirement: Magnetar Revision Used By Module Checks

A module that depends on the Runtime SHALL be built inside a Magnetar checkout, at its submodule path. The checkout SHALL use Magnetar's branch with the same name as the pull request's head branch when it exists, and Magnetar's default branch otherwise. The run SHALL state which Magnetar revision it used.

#### Scenario: Module-only change

- **WHEN** a module pull request has no Magnetar branch of the same name
- **THEN** the module is tested against Magnetar's default branch

#### Scenario: Coordinated change

- **WHEN** a module pull request's branch has the same name as a Magnetar branch
- **THEN** the module is tested against that Magnetar branch, and the run log names it

#### Scenario: Sibling module

- **WHEN** a module depends on another module by path
- **THEN** the other module is checked out at the commit that the Magnetar revision pins

### Requirement: CUDA Provider Hardware Checks

The CUDA Provider repository SHALL run its tests, including the hardware-gated ones, on a GPU runner for pushes to its default branch and for pull requests from branches of the repository itself. The job SHALL fail when no GPU is available; it SHALL NOT pass by skipping.

#### Scenario: Pull request from a repository branch

- **WHEN** a pull request from a branch of the CUDA Provider repository is opened or updated
- **THEN** the Provider's tests run on real hardware and report on that pull request

#### Scenario: Pull request from a fork

- **WHEN** a pull request comes from a fork
- **THEN** the GPU job does not run, and the hosted checks still run

#### Scenario: No GPU on the runner

- **WHEN** the GPU job starts and no GPU is visible
- **THEN** the job fails

### Requirement: Module Checks Use No Secret

The module checks SHALL read the Magnetar repository and the module repository with a read-only token and SHALL NOT use a repository secret.

#### Scenario: Token permissions

- **WHEN** the permissions of the module check workflow are inspected
- **THEN** they grant read access to contents only

### Requirement: Magnetar Integration Checks Remain

Magnetar SHALL keep verifying the pinned combination of modules in its own CI. The module checks SHALL NOT replace Magnetar's integration jobs, its scheduled GPU run, or the integration test crates, which stay in the Magnetar repository.

#### Scenario: Pin update

- **WHEN** Magnetar updates a module's pin
- **THEN** Magnetar's integration jobs build and test that module at the pinned commit, as before

#### Scenario: Cross-repository tests

- **WHEN** a test needs the Runtime and more than one module
- **THEN** it lives in Magnetar's integration test crates and runs in Magnetar's CI
