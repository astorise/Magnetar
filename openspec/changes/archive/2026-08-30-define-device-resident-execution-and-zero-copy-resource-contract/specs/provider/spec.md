## ADDED Requirements
### Requirement: Provider Advertises Memory Capabilities

Provider SHALL advertise supported logical memory capabilities.

#### Scenario: Integrated GPU

Given Provider supports shared coherent host/Device allocation

When capabilities are queried

Then Runtime can evaluate zero-copy eligibility.

### Requirement: Provider Owns Native Mapping

Provider SHALL implement native mapping mechanics privately.

#### Scenario: Vulkan memory map

Given Resource is host-visible

When Runtime requests logical mapping

Then Vulkan memory handles/pointers remain Provider-private.

### Requirement: Provider Reports Coherency Semantics

Provider SHALL accurately describe whether mapped memory requires visibility
maintenance.

#### Scenario: Non-coherent memory

Given host mapping is non-coherent

When host/Device ownership changes

Then Provider performs required native flush/invalidate semantics.

### Requirement: Provider Advertises Peer Access

Provider SHALL expose Device-pair peer-access capabilities when direct peer
access is available.

#### Scenario: GPU0 and GPU1

Given GPU0 can directly read GPU1 memory

When capabilities are queried

Then peer-read is explicit.

### Requirement: Provider Does Not Invent Host Staging

Provider SHALL report when an operation requires host staging rather than
silently hiding it from Runtime policy.

#### Scenario: Unsupported peer transfer

Given Device-to-Device copy internally requires host temporary

When policy forbids staging

Then Provider cannot perform hidden fallback.

### Requirement: Provider Resolves Native Resource Handle

Runtime SHALL pass logical/opaque Resource identity into Provider-controlled
execution boundary.

#### Scenario: Kernel submit

Given Tensor Resource is Device-resident

When Provider launches Kernel

Then Provider resolves its own native pointer internally.

### Requirement: Provider Does Not Export Native Memory By Default

Native memory handles SHALL remain private absent explicit interoperability
capability.

#### Scenario: CUDA allocation

Given Runtime asks Resource metadata

Then CUDA IPC handle is not returned.
