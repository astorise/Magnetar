## MODIFIED Requirements

### Requirement: Compute Capability

The Runtime SHALL expose a Compute Capability.

Components SHALL use this Capability for portable mathematical execution.

The canonical executable Compute Capability identifier SHALL be:

`magnetar:compute/run@2.0.0`.

Components SHALL request Compute without directly selecting a Provider or
Device.

#### Scenario: Resolve canonical Compute Capability

Given a Provider advertising `magnetar:compute/run@2.0.0`

And a Component imports `magnetar:compute/run@2.0.0`

When the Component submits compatible compute work

Then the Runtime resolves the Provider and Device

And the selected native execution identity is not supplied by the Component.

---

### Requirement: Hardware Independence

The Compute Capability SHALL remain independent from concrete hardware
implementations.

Portable Compute requests SHALL NOT contain concrete Provider or Device
selection.

#### Scenario: Execute on different Providers

Given multiple Providers implement compatible Compute v2 contracts

When the same portable Component executes on different systems

Then the Component does not require modification to name a CPU, GPU, Provider,
or Device.

---

### Requirement: WIT Contract

The executable Compute Capability SHALL be defined through WIT.

The canonical executable package SHALL be:

`magnetar:compute@2.0.0`.

The `run` interface SHALL provide coarse compute submission, opaque tensor and
graph resources, portable tensor descriptors, operation lifecycle semantics,
portable data-movement intent, and stable structured errors.

A reference Component world SHALL import the `run` interface rather than export
the Runtime's implementation of it.

#### Scenario: Validate Component-facing world

Given a Component consumes Compute

When its WIT world is inspected

Then `magnetar:compute/run@2.0.0` is an imported Capability

And the Component does not implement the Runtime Compute Capability itself.

---

### Requirement: Version Compatibility

Compute implementations SHALL follow semantic versioning.

Breaking WIT changes SHALL use a new major version.

`magnetar:compute/run@1.1.0` SHALL NOT automatically satisfy a requirement for
`magnetar:compute/run@2.0.0`.

#### Scenario: Provider only supports Compute v1

Given a Provider advertises only `magnetar:compute/run@1.1.0`

When a Component requires `magnetar:compute/run@2.0.0`

Then the Provider is not considered contract-compatible.

---

### Requirement: Opaque Provider-Owned Resources

Tensor storage and graph representation SHALL remain opaque to Components.

Components SHALL NOT receive or provide as routing controls:

- raw pointers
- backend storage
- GPU handles
- native tensor objects
- Provider handles
- Device handles
- ProviderBinding
- DeviceBinding
- AffinityGroupId
- native queue identities
- native stream identities

through the portable Compute contract.

#### Scenario: Use tensor resource

Given a Component receives an opaque tensor resource

When the tensor participates in another operation

Then the Runtime obtains authoritative placement and affinity from its own
resource state

And not from Component-supplied native identifiers.

---

### Requirement: Data Movement and Conversion Operations

The Compute Capability SHALL support explicit data movement and conversion
semantics.

These operations MAY include:

- upload
- download
- copy
- materialize
- transfer
- dtype conversion
- placement conversion

Portable data movement SHALL describe semantic placement intent.

Portable data movement SHALL NOT directly identify a target Provider, Device,
or Runtime AffinityGroupId.

The Runtime SHALL resolve the concrete destination.

#### Scenario: Transfer tensor

Given a tensor belongs to a Provider or Device

And a Component requests an explicit transfer using `runtime-selected`
placement

When the Runtime plans the transfer

Then Resource Affinity and compatibility are validated

And the Runtime selects the concrete eligible Provider and Device.

---

### Requirement: Structured Compute Errors

Compute execution SHALL return stable structured errors.

Provider-specific diagnostic messages MAY be included as optional diagnostics.

Runtime-produced diagnostics MAY identify the Provider or Device involved in a
resolved execution attempt.

These identifiers SHALL describe Runtime decisions or failures and SHALL NOT
define Component routing semantics.

#### Scenario: Provider failure

Given the Runtime selected a Provider

And the Provider reports a native hardware-specific failure

When the Runtime returns a Compute error

Then the error contains stable semantics

And MAY identify the resolved Provider in diagnostic output

But the diagnostic identity does not become a future routing instruction.

---

### Requirement: Compute Boundary Exclusions

The portable Compute WIT contract SHALL exclude native execution control,
including:

- raw device handles
- raw Provider handles
- Provider selection
- Device selection
- Runtime AffinityGroupId selection
- GPU pointers
- native queues
- native streams
- backend storage
- Rust trait objects
- backend-specific kernel names as portable requirements
- autograd state
- training behavior

#### Scenario: Review portable Compute surface

Given a reviewer inspects the Compute v2 WIT request surface

When routing-related fields are reviewed

Then no Component-provided Provider, Device, or Runtime affinity-group selector
exists.

---

## ADDED Requirements

### Requirement: Portable Placement Intent

Compute v2 SHALL define portable placement intent for explicit data movement.

The initial placement intents SHALL include:

- `preserve-source-affinity`
- `runtime-selected`
- `host-accessible`

Placement intent SHALL describe semantic requirements rather than native target
identity.

#### Scenario: Runtime-selected placement

Given a Component requests a data transfer with `runtime-selected`

When the Runtime resolves the transfer

Then compatible placement is selected from Runtime-owned policy and state

And the Component does not name the destination Provider or Device.

---

### Requirement: Preserve Source Affinity Intent

`preserve-source-affinity` SHALL request that authoritative affinity associated
with the source resource be preserved.

It SHALL NOT authorize migration to another Provider or Device.

#### Scenario: Provider-pinned tensor

Given a tensor resource is Provider-pinned

When a Component requests an operation using `preserve-source-affinity`

Then the resulting plan remains compatible with that Provider binding

Or the Runtime rejects the request.

---

### Requirement: Runtime-Selected Placement Intent

`runtime-selected` SHALL allow the Runtime to select compatible destination
placement.

Selection SHALL apply mandatory Resource Affinity constraints before policy
preference.

#### Scenario: Multiple compatible Providers

Given a portable transfer may execute through multiple Providers

And no authoritative affinity requires one candidate

When placement is resolved

Then Resolution Policy may select among compatible candidates.

---

### Requirement: Host-Accessible Placement Intent

`host-accessible` SHALL describe a requirement that resulting data can be
accessed through a portable host data path.

`host-accessible` SHALL NOT mean "select the CPU Provider".

#### Scenario: Provider supports host-visible memory

Given a non-CPU Provider can expose the result through a host-accessible path

When a Component requests `host-accessible`

Then the Runtime may use that Provider without selecting a CPU Provider solely
because of the placement intent.

---

### Requirement: Explicit Host Staging Policy

Compute v2 SHALL represent Component host-staging acceptance explicitly.

The initial policies SHALL include:

- `forbid`
- `permit`

`permit` SHALL mean that host staging is semantically acceptable.

It SHALL NOT override Runtime policy, Provider support, memory planning, or
Resource Affinity.

#### Scenario: Runtime policy forbids staging

Given a Component permits host staging

And Runtime policy forbids host staging for the operation

When execution is planned

Then host staging is not used.

---

### Requirement: Host Staging Can Be Forbidden

When a Component selects `forbid` host staging, the Runtime SHALL NOT silently
stage the resource through host memory.

#### Scenario: Only compatible path requires host staging

Given an explicit transfer cannot be completed without host staging

And the Component specified `forbid`

When the Runtime plans the operation

Then it returns a structured movement or affinity failure

And does not silently stage through host memory.

---

### Requirement: No Portable Provider Selection

A Compute Component request SHALL NOT contain ProviderId, ProviderBinding,
Provider name, or equivalent concrete Provider routing input.

#### Scenario: Component wants accelerator execution

Given a Component prefers accelerated execution

When it submits portable Compute

Then it expresses portable requirements

And does not provide a concrete Provider identifier.

---

### Requirement: No Portable Device Selection

A Compute Component request SHALL NOT contain DeviceId, DeviceBinding, native
device handle, or equivalent concrete Device routing input.

#### Scenario: Multiple GPUs exist

Given several GPU Devices are available

When a Component submits portable Compute

Then the Runtime selects the eligible Device

And the Component does not submit `gpu:0` or another concrete Device identity.

---

### Requirement: No Portable Affinity Group Selection

A Component SHALL NOT construct or select Runtime Resource Affinity by providing
an `AffinityGroupId`.

AffinityGroupId SHALL remain Runtime-owned identity.

#### Scenario: Component consumes bound tensor

Given a tensor belongs to an affinity group

When the Component submits dependent work

Then the Runtime discovers the affinity from the tensor resource

And the Component does not need the numeric affinity-group identifier.

---

### Requirement: Diagnostic Identity Is Output Metadata

Runtime-produced diagnostic and observability output SHALL treat Provider and
Device identities as descriptive output metadata when those identities appear.

Their presence in output SHALL NOT imply that equivalent input fields are part
of the portable Compute request contract.

#### Scenario: Inspect failed resolution

Given all Compute candidates are rejected

When a diagnostic is returned

Then candidate identities MAY be reported

But the Component cannot use those diagnostic fields as direct routing handles
in the Compute contract.

---

### Requirement: Provider-Specific Values Do Not Imply Resolution

Provider-specific extension values retained by Compute SHALL NOT be interpreted
as portable Provider selection.

Provider-specific extensions SHALL remain explicitly non-portable.

#### Scenario: Provider-specific operation extension

Given an explicitly non-portable operation extension is used

When compatibility is evaluated

Then the Runtime validates the extension against an already eligible Provider
context

And does not treat the extension's textual identifier as a general portable
routing request.

---

### Requirement: Component-Facing Compute World Direction

Magnetar Compute SHALL define Components as consumers of the Compute Capability.

A reference world MAY be equivalent to:

```wit
world compute-consumer {
    import run;
}
```

Individual Components MAY define their own worlds importing the same interface.

#### Scenario: Model Component imports multiple Capabilities

Given a future Model Component needs Compute and generation-related contracts

When its world is defined

Then it may import `magnetar:compute/run@2.0.0` together with its other
Capabilities

Without using the Compute package's reference world directly.

---

### Requirement: Native Provider Boundary Is Not WIT Compute World

Native Providers SHALL NOT be required to implement a WASM Component world to
serve the Compute Capability.

The Runtime SHALL bridge the portable Compute Capability to the native Provider
execution API.

#### Scenario: CUDA Provider executes Component request

Given a Component imports Compute v2

And the Runtime resolves a CUDA Provider

When execution begins

Then the Component crosses the portable WIT boundary into the Runtime

And the Runtime crosses its native Provider API boundary to CUDA.

---

### Requirement: Compute v2 Breaking Migration

Compute v2 SHALL represent the removal of Component-provided Provider, Device,
and AffinityGroup routing fields as a major Compute contract revision.

#### Scenario: Existing v1 Component

Given a Component was built against Compute v1.1

When only Compute v2 is available

Then the Component is not silently treated as a v2 Component

And compatibility requires rebuilding or an explicit compatibility mechanism.
