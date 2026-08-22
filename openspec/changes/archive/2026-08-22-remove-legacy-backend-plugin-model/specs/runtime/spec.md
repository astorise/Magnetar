## ADDED Requirements

### Requirement: Provider-Optional Runtime Initialization

The Runtime SHALL initialize without requiring any registered Provider.

Provider availability SHALL become relevant only when Runtime work requires a
Capability whose implementation must be resolved.

#### Scenario: Start Runtime without Provider

Given a valid Runtime configuration

And no Providers are registered

When the Runtime initializes

Then initialization succeeds

And no hardware implementation is implicitly required.

---

### Requirement: Provider-Only Native Execution Path

The Runtime SHALL use Providers as the sole native execution extension
mechanism.

The Runtime SHALL NOT maintain a parallel Backend execution registry.

#### Scenario: Execute native compute

Given a Component requests a Compute Capability

When native execution is resolved

Then the Runtime selects a Provider implementing that Capability

And no Backend abstraction participates in the execution path.

---

### Requirement: No Direct Backend Selection Configuration

Runtime configuration SHALL NOT contain a Backend selector.

The Runtime SHALL NOT expose `preferred_backend` or an equivalent legacy
Backend preference.

Provider preference SHALL be expressed through Resolution Policy rather than a
direct native implementation selector.

#### Scenario: Prefer an execution implementation

Given an application wants to influence Provider selection

When Runtime policy is configured

Then the preference is expressed through Resolution Policy

And not through a Backend name.

---

### Requirement: Execution Context Is Backend Independent

Runtime execution contexts SHALL NOT contain legacy Backend identity.

Execution identity SHALL use only architectural concepts that are actually
required, such as:

- execution context identity
- ProviderBinding
- DeviceBinding
- CapabilityBinding
- Resource Affinity
- Execution Plan identity

#### Scenario: Create execution context

Given the Runtime creates an execution context

When no Provider has yet been resolved

Then the context does not require a Backend name.

---

### Requirement: Runtime Native Extension Uniqueness

The Runtime SHALL NOT provide multiple overlapping generic mechanisms for native
hardware execution.

Provider SHALL be the canonical native extension mechanism.

#### Scenario: Add new native accelerator

Given support for a new accelerator is implemented

When the integration is added to Magnetar

Then it is implemented as a Provider

And not as a Backend or Plugin.

---

## REMOVED Requirements

### Requirement: Backend Independence

The legacy requirement using Backend as the hardware execution abstraction is
removed.

Its intent is replaced by `Provider-Optional Runtime Initialization` and the
canonical Provider architecture.

#### Scenario: Historical Backend independence

Given the previous Runtime specification described initialization without a
Backend

When this change is archived

Then Runtime initialization without hardware remains supported

But the requirement is expressed using Provider terminology.
