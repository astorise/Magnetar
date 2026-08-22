## ADDED Requirements

### Requirement: Component Distribution Is Source Neutral

Magnetar's Component distribution model SHALL be independent from any single
source implementation.

Tachyon, local directories, local caches, registries, and clients MAY act as
sources, but none is required by the architecture.

#### Scenario: Local-only operation

Given no Tachyon service is available

When Magnetar loads a trusted local Inference Component package

Then distribution and validation still work.

---

### Requirement: Tachyon Distribution Does Not Imply Execution Trust

Tachyon SHALL NOT imply execution trust when it distributes
Magnetar-compatible Inference Components.

Magnetar SHALL still validate and trust those Components locally before
execution.

#### Scenario: Tachyon supplies Component

Given Tachyon provides a Component Artifact Package

When Magnetar receives it

Then Magnetar computes digest, validates manifest, checks compatibility,
validates inference authority, applies trust policy, and only then prepares it.

---

### Requirement: Distribution Is Limited To Inference Components

Magnetar Component distribution SHALL be limited to Components within Magnetar's
inference Runtime scope.

General-purpose agent tools are outside Magnetar Runtime distribution scope.

#### Scenario: Shell Component offered

Given a source offers a shell-execution Component

When Magnetar classifies it

Then Magnetar rejects it as outside inference scope.

---

### Requirement: magnetar-cli May Be A Distribution Source

`magnetar-cli` SHALL be treated as a Component Distribution Source when it
provides Component Artifact Packages to Magnetar.

When it does so, it is a source of artifacts, not a bypass around Magnetar
validation.

#### Scenario: CLI provides local Component

Given `magnetar-cli` submits a local Component package to Magnetar

When Magnetar receives it

Then Magnetar applies the same digest, manifest, WIT, compatibility, authority,
and trust validation as for any other source.

---

### Requirement: Client Authority Does Not Transfer To Magnetar

Authority held by a client or source SHALL NOT transfer into Magnetar Component
authority.

#### Scenario: CLI has filesystem access

Given `magnetar-cli` can read a workspace file

When it provides prompt context or a Component package to Magnetar

Then Magnetar does not gain filesystem authority.

---

### Requirement: Distribution Is Not Remote Execution

The Component distribution contract SHALL provide artifact bytes and metadata.

It SHALL NOT define remote execution of Components.

#### Scenario: Remote source available

Given an external source stores a Component Artifact Package

When Magnetar fetches it

Then Magnetar executes it only locally after validation, trust, preparation,
linking, and instantiation.
