## ADDED Requirements

### Requirement: Runtime Owns Component Policy

The Magnetar Runtime SHALL own policy governing Component registration,
validation, authorization, linking, instantiation, invocation, and destruction.

ComponentEngine SHALL execute those decisions but SHALL NOT define global
Magnetar policy.

#### Scenario: Import is denied

Given ComponentEngine could technically link a filesystem interface

And Runtime policy does not authorize that import

When the Component is instantiated

Then the interface is not linked.

---

### Requirement: ComponentEngine Is an Internal Execution Boundary

The Runtime SHALL interact with concrete WebAssembly execution through an
engine-neutral ComponentEngine boundary.

#### Scenario: Runtime prepares Component

Given valid Component executable bytes

When preparation begins

Then the Runtime delegates engine-specific validation or compilation to
ComponentEngine

Without exposing the engine's compiled representation as Runtime public API.

---

### Requirement: Runtime Constructs Component Link Plans

The Runtime SHALL create Component Link Plans from:

- Component WIT imports
- available Runtime interfaces
- Capability contracts
- compatibility rules
- interface-level authorization
- Runtime policy

The concrete engine Linker SHALL be constructed from the approved Link Plan.

#### Scenario: Unauthorized import exists

Given a Component imports interfaces X and Y

And only X is authorized

When the Link Plan is built

Then X may be linked

And instantiation fails for mandatory unauthorized Y.

---

### Requirement: Linking and Provider Resolution Are Separate

Runtime Component linking SHALL remain separate from Provider and Device
resolution.

Linking a Provider-backed Capability SHALL expose a Runtime endpoint rather than
a native Provider handle.

#### Scenario: Compute import linked

Given a Component imports Compute

When the Link Plan is completed

Then Compute is available through the Runtime endpoint

And no specific Provider is selected merely by Component instantiation.

---

### Requirement: Runtime Owns Component Instance Identity

Every Component Instance SHALL receive Runtime-owned identity.

Component code SHALL NOT control or forge this identity.

#### Scenario: Create instance

Given a prepared Component

When ComponentEngine instantiates it

Then the Runtime associates the engine instance with a new ComponentInstanceId.

---

### Requirement: Runtime Tracks Definition and Instance Separately

The Runtime SHALL distinguish reusable Component definitions from their live
instances.

#### Scenario: One definition, multiple instances

Given a Component definition is prepared once

When two instances are created

Then the Runtime tracks one definition identity and two separate instance
identities.

---

### Requirement: Runtime Does Not Require Generic Component Start

Runtime lifecycle SHALL NOT assume that every Component exports a generic start
operation.

#### Scenario: Component has only application exports

Given a valid Component exports its application-specific WIT interface

When it is instantiated successfully

Then the Runtime may make the instance ready without invoking a generic start
function.

---

### Requirement: Runtime Does Not Require Generic Component Stop

Runtime shutdown SHALL NOT depend on a universal Component stop export.

The Runtime SHALL prevent new calls, coordinate interruption or draining, and
destroy engine instances according to Runtime policy.

#### Scenario: Shutdown Component without stop export

Given a ready Component has no stop function

When Runtime shutdown occurs

Then the Runtime can safely terminate and destroy the instance without requiring
such an export.

---

### Requirement: Runtime Derives Dependencies from Imports

The Runtime SHALL derive Component dependency requirements from WIT imports
rather than direct Component-name dependency lists.

#### Scenario: Implementation providing import changes

Given Component A imports interface X

And Runtime configuration changes which authorized implementation serves X

When Component A is instantiated again

Then its Component metadata does not require modification merely because the
implementation changed.

---

### Requirement: Runtime Does Not Automatically Compose Components

The Runtime SHALL NOT automatically connect Component exports to matching
Component imports solely by global interface discovery.

#### Scenario: Matching export exists

Given one registered Component exports X

And another imports X

When no explicit composition policy exists

Then the Runtime does not automatically create a direct dependency between the
instances.

---

### Requirement: Runtime Enforces Fail-Closed Linking

An interface absent from an instance's authorized Link Plan SHALL be unavailable
to that Component Instance.

#### Scenario: Network not linked

Given a Component's Link Plan contains no network interface

When the Component executes

Then no ambient network authority is available through the Component Runtime.

---

### Requirement: Runtime Owns Component Resource Policy

Runtime policy SHALL determine semantic Component execution limits.

ComponentEngine SHALL implement enforceable engine-specific mechanisms without
changing portable Component contracts.

#### Scenario: Deadline configured

Given Runtime policy assigns an execution deadline

When a Component invocation exceeds that deadline

Then ComponentEngine interruption is requested

And the resulting error is normalized by Runtime.

---

### Requirement: Runtime Normalizes Component Engine Failures

Concrete engine failures SHALL be translated into stable Runtime Component
errors before crossing canonical Magnetar APIs.

#### Scenario: Engine reports trap

Given ComponentEngine returns an engine-specific trap

When the Runtime handles the result

Then callers receive a stable Component error and optional redacted diagnostic.

---

### Requirement: Runtime Preserves Error-Domain Separation

Runtime Component execution errors SHALL remain distinct from Provider errors,
Device health, and Compute errors unless an explicit mapping is required.

#### Scenario: Component trap before Provider call

Given the Component traps before invoking Compute

When the Runtime reports the failure

Then it is classified as a Component execution failure

And not as `provider-unavailable`.

---

### Requirement: Runtime Coordinates Component Shutdown

Runtime shutdown SHALL:

1. prevent admission of new Component invocations
2. allow or interrupt active invocations according to policy
3. coordinate outstanding Runtime resources
4. destroy Component instances
5. release engine-owned instance state

#### Scenario: Runtime shuts down with active Component

Given a Component invocation is active

When Runtime shutdown begins

Then shutdown follows configured interruption or draining policy

And eventually releases the engine instance without requiring Component-specific
native cleanup APIs.

---

### Requirement: Component Observability Does Not Control Execution

Observability of Component lifecycle and execution SHALL remain non-authoritative
with respect to Runtime execution semantics.

#### Scenario: Component exporter is unavailable

Given Component lifecycle observations cannot be exported

When another Component executes

Then its linking, invocation, and execution correctness remain unaffected.

---

### Requirement: Component Engine Replacement Does Not Change Runtime Architecture

The Runtime SHALL NOT embed architectural assumptions that require one concrete
WebAssembly engine implementation.

#### Scenario: Replace Wasmtime

Given a future engine satisfies ComponentEngine requirements

When Magnetar adopts that engine

Then canonical Component, Capability, Provider, Device, and Resource Affinity
contracts remain valid.

