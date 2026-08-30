## ADDED Requirements
### Requirement: Device Remains Descriptive In Multi Device Placement

Device SHALL expose identity/capability/status but SHALL not become placement
controller.

#### Scenario: Two GPUs

Given Runtime knows both Devices

When placement is selected

Then decision resides in Runtime rather than `Device.place()` API.

### Requirement: Device Pair Metadata May Be Described

Runtime SHALL support querying Provider/Device metadata describing relationships between
Devices.

#### Scenario: Peer link

Given GPU pair has direct transfer capability

When placement evaluates communication cost

Then relation metadata may inform decision.

### Requirement: Device Does Not Expose Native Topology Handle

Device relationship metadata SHALL not expose native PCI/CUDA/OS handles as
execution authority.

#### Scenario: Device topology

Given Provider knows PCI topology

When Runtime receives metadata

Then only safe capability/performance classification is exposed.

### Requirement: Device Loss Is Individually Observable

One Device SHALL be able to transition unavailable without forcing all Devices into same
state.

#### Scenario: GPU1 lost

Given GPU0 remains healthy

When Device status updates

Then GPU0 remains individually usable.
