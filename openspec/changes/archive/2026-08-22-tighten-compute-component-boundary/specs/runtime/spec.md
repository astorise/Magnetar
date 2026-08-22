## ADDED Requirements

### Requirement: Runtime Owns Compute Placement Resolution

The Runtime SHALL translate portable Compute placement intent into concrete
native placement.

Concrete placement MAY include:

- ProviderBinding
- DeviceBinding
- CapabilityBinding
- memory placement
- Resource Affinity
- transfer steps
- materialization steps
- host staging decisions

These concrete bindings SHALL remain Runtime-owned.

#### Scenario: Resolve portable transfer

Given a Component requests `runtime-selected` placement

When the Runtime prepares execution

Then the Runtime determines the concrete Provider and Device

And stores the resulting bindings internally.

---

### Requirement: Portable Placement and Resolved Binding Are Separate Models

The Runtime SHALL maintain a conceptual distinction between:

```text
Portable Placement Intent
```

and:

```text
Resolved Native Binding
```

A portable placement request SHALL NOT itself become authoritative Resource
Affinity.

#### Scenario: Receive placement request

Given a Component requests `runtime-selected`

When placement is resolved to Provider A and Device 0

Then the Runtime may create internal bindings to Provider A and Device 0

But those bindings were not supplied by the Component.

---

### Requirement: Runtime Derives Source Resource Affinity

For an existing opaque tensor resource, the Runtime SHALL obtain authoritative
Resource Affinity from Runtime-managed resource state.

The Runtime SHALL NOT trust caller-supplied affinity identifiers as a
replacement for the resource's actual binding.

#### Scenario: Bound tensor submitted

Given a tensor is bound to Provider A and Device 0

When a Component submits the tensor as an input

Then the Runtime derives those bindings from the tensor resource

And does not ask the Component to restate them.

---

### Requirement: Placement Resolution Order

Compute placement SHALL apply constraints in an order that preserves mandatory
correctness.

At minimum, the Runtime SHALL evaluate:

1. portable contract validity
2. source resource validity
3. Resource Affinity
4. Capability compatibility
5. Provider advertisement compatibility
6. Device compatibility and availability
7. memory and data-movement feasibility
8. Resolution Policy preferences

Policy preference SHALL NOT override mandatory compatibility or Resource
Affinity.

#### Scenario: Policy prefers incompatible Provider

Given Resource Affinity requires Provider A

And Resolution Policy prefers Provider B

When Compute placement is resolved

Then Provider A remains required

And Provider B is rejected for that dependent operation.

---

### Requirement: Resolved Data Movement Plan

The Runtime SHALL represent resolved data movement separately from the portable
Component descriptor.

A resolved movement plan MAY contain:

- source resource identity
- source Resource Affinity
- selected Provider
- selected Device
- selected Capability implementation
- destination placement
- transfer requirement
- materialization requirement
- host staging decision
- resulting Resource Affinity

#### Scenario: Plan placement conversion

Given a Component requests an explicit placement conversion

When resolution succeeds

Then the Runtime creates a concrete movement plan

Before Provider execution is submitted.

---

### Requirement: Resolved Movement Plan Is Native

Resolved Provider and Device bindings SHALL NOT be serialized back into the
portable data-movement request as authoritative handles.

#### Scenario: Plan uses GPU Device

Given the Runtime resolves movement to a GPU Device

When the plan is stored

Then the DeviceBinding remains in Runtime-native state

And is not inserted into the Component's WIT descriptor.

---

### Requirement: Explicit Data Movement Remains Required

Runtime-owned placement resolution SHALL NOT authorize implicit cross-Provider
or cross-Device migration.

When incompatible placement requires movement, an explicit movement,
materialization, copy, transfer, upload, download, or placement-conversion
semantic step SHALL exist.

#### Scenario: Consumer requires incompatible placement

Given a tensor is bound to Device A

And dependent work is planned for incompatible Device B

When execution planning occurs

Then an explicit movement step is required

And the Runtime does not silently move the tensor as an invisible side effect.

---

### Requirement: Runtime-Selected Does Not Override Affinity

`runtime-selected` SHALL only allow selection within the set of candidates
permitted by authoritative Resource Affinity and compatibility.

#### Scenario: Provider-pinned resource

Given a resource is Provider-pinned to Provider A

And a Component specifies `runtime-selected`

When execution is resolved

Then `runtime-selected` does not authorize Provider B.

---

### Requirement: Host Staging Requires Dual Permission

Host staging SHALL require both:

- portable operation semantics that permit staging
- Runtime execution policy that permits staging

Provider and memory-planning support SHALL also be validated.

#### Scenario: Component permits but Runtime denies

Given a Component specifies `permit`

And Runtime policy forbids host staging

When the plan is evaluated

Then staging is rejected.

---

### Requirement: Host Staging Is Never Implicit When Forbidden

If the portable request specifies `forbid`, the Runtime SHALL NOT introduce a
hidden host-staging step.

#### Scenario: Device-to-device transfer needs host intermediate

Given peer transfer is unavailable

And the only implementation uses host staging

And the Component specified `forbid`

When transfer planning occurs

Then execution is rejected with a structured error.

---

### Requirement: Runtime May Use Administrative Placement Constraints

Native Runtime policy SHALL keep administrative concrete Provider or Device
constraints outside portable Compute WIT when such constraints are introduced.

Such constraints SHALL remain outside the portable Compute WIT.

Administrative constraints SHALL still respect Resource Affinity and Capability
compatibility.

#### Scenario: Administrator constrains one Runtime

Given Runtime policy administratively restricts Compute to an eligible Device

When a portable Component submits Compute

Then the Runtime applies that native policy

Without requiring the Component to know the Device identity.

---

### Requirement: Compute Diagnostics Reflect Resolved Placement

The Runtime SHALL treat Provider and Device identities reported through
structured diagnostics and observability as descriptions of selected or
rejected placement resolution results.

These identities SHALL be descriptive output.

#### Scenario: Provider rejected by memory constraints

Given a candidate Provider is rejected during planning

When diagnostics are produced

Then the Runtime may identify that Provider and the rejection reason

Without turning the diagnostic identity into a portable execution handle.

---

### Requirement: Execution Plan Owns Final Binding

The validated ComputeExecutionPlan SHALL contain the concrete execution binding
used by the Scheduler.

The Scheduler SHALL consume this resolved binding rather than interpreting
portable Component placement intent independently.

#### Scenario: Scheduler receives plan

Given placement has already resolved Provider A and Device 0

When the Scheduler accepts the ComputeExecutionPlan

Then it schedules that validated binding

And does not rerun portable placement interpretation.

---

### Requirement: Memory Planning Consumes Resolved Placement

Memory Planning SHALL use resolved Provider and Device placement when
determining allocation, reuse, transfer, and materialization requirements.

Portable Component intent SHALL NOT directly control native allocation.

#### Scenario: Plan tensor allocation

Given a placement has resolved to one Device

When Memory Planning prepares tensor storage

Then it validates the selected Provider and Device constraints

Rather than allocating based on Component-supplied Device identity.

---

### Requirement: No Component-Created Runtime Affinity

Resource Affinity SHALL be created and maintained by the Runtime from actual
resource ownership and execution state.

Portable Components MAY request affinity preservation but SHALL NOT manufacture
Runtime affinity bindings.

#### Scenario: New tensor output

Given Provider execution creates a tensor output

When the Runtime registers the tensor resource

Then the Runtime attaches the actual Provider and Device affinity

And the Component receives only the opaque resource and portable metadata.

---

### Requirement: Placement Failure Is Structured

Failure to resolve portable placement intent SHALL use stable Compute errors.

Applicable failures MAY include:

- no-compatible-provider
- policy-rejected-provider
- provider-unavailable
- device-unavailable
- unsupported-data-movement
- incompatible-resource-affinity
- provider-pinned-resource
- device-bound-resource
- affinity-group-mismatch
- invalid-transfer
- materialization-required

#### Scenario: No valid transfer destination

Given a transfer request has no candidate satisfying affinity, Capability, and
movement requirements

When placement resolution completes

Then the Runtime returns a structured Compute error

And does not silently weaken the request.

---

### Requirement: Compute v1 and v2 Are Distinct Contracts

The Runtime SHALL treat Compute v1.1 and Compute v2.0 as different major
contracts.

Support for one SHALL NOT imply support for the other.

#### Scenario: Resolve v2 request

Given a Component imports Compute v2

And a Provider advertises only Compute v1.1

When the Runtime resolves candidates

Then that Provider is not selected as a v2-compatible implementation.

---

### Requirement: Compatibility Translation Must Be Explicit

Compatibility translation from Compute v1 requests to Compute v2 SHALL require
an explicit adapter.

It SHALL define how legacy concrete Provider/Device target fields are handled.

The Runtime SHALL NOT silently preserve those fields as portable routing
authority.

#### Scenario: Legacy Component names Provider

Given a legacy v1 Component requests a concrete target Provider

When an explicit compatibility adapter is used

Then the adapter applies documented migration policy

And the v2 portable contract itself remains free of Provider routing input.
