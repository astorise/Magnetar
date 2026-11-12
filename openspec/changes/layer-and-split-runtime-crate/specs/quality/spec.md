## ADDED Requirements

### Requirement: Runtime Layering Gate

Continuous integration SHALL check on every pull request that no Runtime module or crate depends on a higher layer than its own, using the version-controlled layer map. A violation SHALL fail the pull request and SHALL name the module, the dependency and both layers. Upward dependencies that predate the gate MAY be listed as debt in the layer map; the gate SHALL fail when a listed dependency no longer exists, so that the list only shrinks.

#### Scenario: Upward dependency

- **WHEN** a pull request makes a kernel pipeline module use an item of the Inference API
- **THEN** the layering gate fails and reports the module, the item's module and their layers

#### Scenario: Removed debt

- **WHEN** a pull request removes an upward dependency that the layer map lists as debt, without removing the entry
- **THEN** the layering gate fails and names the stale entry

#### Scenario: Layer map change

- **WHEN** a pull request moves a module to another layer in the layer map
- **THEN** the change is visible in the pull request diff and the gate checks the new assignment

### Requirement: Quality Gates Cover Every Runtime Crate

Formatting, compilation, Clippy, test, documentation, coverage and publication dry-run gates SHALL cover every Runtime crate, not only `magnetar-runtime`. The coverage baseline SHALL NOT lose covered lines when code moves between files or crates.

#### Scenario: New crate in the gates

- **WHEN** a Runtime layer crate is added to the workspace
- **THEN** each gate runs on it in the same pull request

#### Scenario: Coverage after a move

- **WHEN** code moves to another file or crate
- **THEN** the re-measured baseline has at least as many covered lines as before, and no exclusion is added for the moved code

#### Scenario: Publication dry run

- **WHEN** the publication dry run executes
- **THEN** it packages every Runtime crate in dependency order and fails if one cannot be packaged
