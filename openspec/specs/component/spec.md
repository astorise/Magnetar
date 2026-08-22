# component Specification

## Purpose

Define the host-side contracts and lifecycle management for portable WebAssembly
Components, independently of concrete hardware implementations.
## Requirements
### Requirement: Component Discovery

The runtime SHALL discover WebAssembly Components during initialization.

#### Scenario: Discover components

Given a valid component directory

When the runtime initializes

Then every compatible component is discovered.

---

### Requirement: Component Contracts

Every Component SHALL declare its portable dependencies and exports through WIT
interfaces.

Component imports SHALL identify required interfaces rather than concrete native
implementations or Component instance names.

The Runtime SHALL validate required imports before instantiation.

#### Scenario: Validate contracts

Given a Component imports one or more WIT interfaces

When the Runtime prepares the Component for instantiation

Then every required interface is validated for compatibility and authorization

And unresolved mandatory imports prevent instantiation.

---

### Requirement: Component Isolation

Components SHALL execute independently from hardware implementations and from
engine-native Runtime objects.

Components SHALL NOT receive:

- Provider handles
- Device handles
- native pointers
- engine Store handles
- engine Linker handles
- engine ResourceTable handles
- native queue or stream handles

through portable contracts.

#### Scenario: Execute on different Runtime environments

Given the same compatible Component

And two Runtime environments with different native Providers

When the Component executes

Then no Component modification is required

And the Component does not observe the native implementation handles.

---

### Requirement: Component Lifecycle

The Runtime SHALL manage Component definition and Component instance lifecycle.

Generic Components SHALL NOT be required to implement universal `start` or
`stop` exports.

A successfully instantiated Component Instance SHALL become available according
to its exported WIT contracts.

Application-specific lifecycle behavior MAY be defined through explicit WIT
interfaces.

#### Scenario: Instantiate Component without start export

Given a valid Component has no generic `start` export

When the Runtime validates, prepares, links, and instantiates it successfully

Then the Component Instance becomes available for invocation.

#### Scenario: Runtime shutdown

Given one or more Component Instances exist

When the Runtime shuts down

Then new invocation is prevented

And the Runtime releases each instance according to Component Runtime lifecycle
policy

Without requiring an implicit portable `stop` function.

---

### Requirement: Dependency Resolution

Component dependencies SHALL be expressed through WIT imports.

Components SHALL NOT require direct dependency on another Component's logical
name as the canonical dependency mechanism.

The Runtime SHALL resolve and authorize required interfaces before
instantiation.

#### Scenario: Resolve Capability dependency

Given a Component imports `magnetar:compute/run`

When the Runtime constructs its Link Plan

Then the import is linked to an authorized Runtime Compute Capability endpoint

And the Component does not name the Component or Provider implementing the
underlying behavior.

---

### Requirement: Component Runtime Observability

The Component Runtime SHALL support structured Runtime observations for
important lifecycle and execution events.

Observations MAY include:

- definition identity
- instance identity
- preparation
- instantiation
- invocation
- interruption
- trap
- resource-limit violation
- destruction

#### Scenario: Component traps

Given a Component invocation traps

When Runtime observability records the failure

Then the observation identifies the relevant Component instance and stable trap
category

Without exposing engine-native handles or secret data.

