## ADDED Requirements

### Requirement: Every released contract version lives in the API repository and never changes

The `magnetar-component-api` repository SHALL hold, as `wit/<version>/model-component-graph.wit`, every released version of the `magnetar:model-component-graph` WIT package that the Runtime serves, and SHALL record the checksum of each in `wit/SHA256SUMS`. A file of a released version SHALL NOT be edited or removed, and a recorded checksum SHALL NOT be changed or removed.

#### Scenario: A released file is edited

- **WHEN** a pull request changes `wit/1.8.0/model-component-graph.wit`
- **THEN** the repository's tests fail because the file differs from its recorded checksum, and its CI refuses the pull request

#### Scenario: A version is released

- **WHEN** a new directory `wit/<version>/` holds a file that declares `package magnetar:model-component-graph@<version>;` and `scripts/sync.py` has run
- **THEN** its checksum is recorded, and the cargo feature and the bindings module of that version exist

#### Scenario: A version whose content cannot be established

- **WHEN** the content a version had cannot be cross-checked against the history or a Component's copy
- **THEN** the version is not carried

### Requirement: A release only adds to the one before it

For every pair of consecutive released versions, the later SHALL contain every interface, function and named type of the earlier, with the same function signatures and the same existing members, in the same order. A variant or an enum MAY gain cases at its end; a record or a flags type SHALL NOT gain members; nothing else SHALL change. A change that does not fit SHALL be a new major version.

#### Scenario: A function is removed or its signature changes

- **WHEN** a new version removes a function of the previous one, or changes the type of one of its parameters or of its result
- **THEN** the repository's tests fail and name the function

#### Scenario: A record grows

- **WHEN** a new version adds a field to a record of the previous one
- **THEN** the repository's tests fail, since a Component built before the field was added no longer matches the layout it receives or sends

#### Scenario: An enum or a variant gains a case at its end

- **WHEN** a new version appends a case to an enum or a variant of the previous one
- **THEN** the tests pass, the Runtime encoding each value against the calling Component's own types

### Requirement: The bindings of a version are generated once, one version per build

The crate SHALL generate the guest bindings of a contract version in a module enabled by one cargo feature of that version, and a build SHALL enable exactly one of those features. Enabling two SHALL fail to compile with a message that says that a build targets one contract version.

#### Scenario: A Component targets a version

- **WHEN** a Component enables the feature of contract 1.8.0 and implements the generated `Guest` trait
- **THEN** it builds for `wasm32-wasip2` into a Component that imports the 1.8.0 `graph-builder` and exports the producer world, without a copy of the WIT file in its repository

#### Scenario: Two versions in one build

- **WHEN** a build enables the features of two contract versions
- **THEN** it fails with the message that exactly one contract-version feature is expected

### Requirement: A Component depends on the API pinned by commit and carries no copy of the contract

While `magnetar-component-api` is not published, a Model Component that targets it SHALL depend on it as a git dependency pinned by commit, SHALL record that commit in its own `Cargo.lock`, and SHALL NOT carry a copy of `model-component-graph.wit`. A branch or a tag SHALL NOT be used as the pin.

#### Scenario: A Component moves onto the API

- **WHEN** a Component replaces its `wit/` copy and its call to the bindings generator by the dependency and a version feature
- **THEN** its recorded graphs, its numerical checks and the Component Model validation of its `.wasm` pass unchanged

#### Scenario: A Component's standalone CI

- **WHEN** a Component's repository is checked out alone
- **THEN** it builds and tests against the pinned commit without a Magnetar checkout

### Requirement: The API repository ships a Component that builds

The repository SHALL contain a minimal Model Component that targets the latest released version, and its CI SHALL build it for `wasm32-wasip2` and validate the result as a Component.

#### Scenario: The template is broken by a release

- **WHEN** a release changes what a Component must implement
- **THEN** the template fails to build in the repository's CI until it is updated, in the same pull request
