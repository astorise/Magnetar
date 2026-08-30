## ADDED Requirements
### Requirement: Device Remains Descriptive

Device SHALL describe memory and peer-access capabilities without becoming
allocation/mapping API.

#### Scenario: GPU memory capability

Given Runtime inspects Device

When metadata is queried

Then Device may report memory classes/capacity but exposes no native allocator.

### Requirement: Device Does Not Expose Native Pointer

Device API SHALL not expose native memory address.

#### Scenario: Device Resource exists

Given Tensor is GPU-resident

When Device metadata is inspected

Then allocation pointer is absent.

### Requirement: Device Describes Memory Capacity

Device status SHALL expose memory capacity/pressure metadata useful to Memory
Manager.

#### Scenario: GPU nearly full

Given Provider observes memory pressure

When Runtime evaluates residency

Then Device/Provider status can inform eviction/admission policy.

### Requirement: Device Peer Capability Is Descriptive

Device relationship metadata SHALL indicate peer-access capability when direct
peer access is available.

#### Scenario: Two GPUs

Given direct peer access unsupported

When Runtime plans zero-copy

Then it does not assume access solely from common vendor.
