## ADDED Requirements

### Requirement: Provider-Agnostic Tensor Value Contract

`ProviderExecutionApi` SHALL expose a tensor value read/write contract whose
type does not require any Provider to expose or accept a specific Provider's
host-visible tensor representation. A Provider implementing only
device-resident storage SHALL be able to satisfy this contract without ever
producing or consuming that host-visible representation.

#### Scenario: Device-resident Provider implements the contract

Given a Provider that holds all Tensor Resources in device memory

When it implements `ProviderExecutionApi`

Then it satisfies the tensor value read/write contract without implementing
conversion to any specific Provider's host tensor type.

#### Scenario: Reference CPU continues to satisfy the contract

Given Reference CPU, a host-visible Provider

When it implements the tensor value read/write contract

Then its existing host-visible representation is one valid value the
contract carries, not a required universal shape.

---

### Requirement: Existing Host-Typed Tensor Access Remains Available

`ProviderExecutionApi` SHALL keep its pre-existing host-typed tensor
read/write methods available and unchanged in signature after this contract
is introduced, so callers that intentionally require host-visible bytes
(test fixtures, hand-written oracles) are not forced to migrate.

#### Scenario: A host-typed caller is unaffected

Given a caller that invokes the pre-existing host-typed tensor read/write
methods

When this contract exists alongside them

Then the call compiles and behaves exactly as it did before this contract
was introduced.

---

### Requirement: Device-Resident Values May Decline Host Materialization

A value produced through the Provider-agnostic tensor value contract SHALL
be able to represent "held privately by the Provider, no host-visible bytes
available" as a first-class outcome, not an error indistinguishable from
"resource not found."

#### Scenario: Opaque value returned for a device-resident resource

Given a Provider reads back a Tensor Resource it stores only device-resident

When it returns a value through this contract

Then the returned value identifies itself as host-unavailable, distinct
from the resource not existing at all.

---

### Requirement: Host Materialization Failure Is Structured

The Provider-agnostic tensor value contract SHALL return a structured error
identifying the resource as device-resident when a caller that requires
host-visible bytes receives a value that declines host materialization, not
a silent default, panic, or ambiguous `None`.

#### Scenario: Caller requests host bytes from an opaque value

Given code that needs actual tensor bytes (e.g. sampling, weight binding, or
KV history concatenation)

When it receives a value that declined host materialization

Then it receives a structured error naming the resource and that it is
device-resident, and does not proceed as if bytes were available.

---

### Requirement: Multi-Output Kernel Dispatch

A Kernel invocation dispatched through `ProviderExecutionApi` SHALL be able
to report more than one produced Tensor Resource, addressed by output
index, rather than the caller assuming a single output at a fixed position.

#### Scenario: Two-output Kernel reports both resources

Given a Kernel invocation declares two outputs and both are produced

When the dispatch result is reported

Then it identifies each output's Tensor Resource by its own output index,
and a caller can resolve either independently.

#### Scenario: Single-output Kernel is unaffected

Given a Kernel invocation declares exactly one output, as every Reference
CPU Kernel does today

When the dispatch result is reported

Then it identifies that one output at index 0, behaviorally identical to
dispatch before this requirement existed.
