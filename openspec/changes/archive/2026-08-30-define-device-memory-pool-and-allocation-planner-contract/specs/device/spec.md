## ADDED Requirements
### Requirement: Device Exposes Capacity Not Allocation Authority

Device SHALL expose memory capacity/status but SHALL NOT expose allocation API.

#### Scenario: GPU memory metadata

Given GPU has 24 GiB memory

When Runtime inspects Device

Then capacity information is available without `allocate()` method.

### Requirement: Device SHALL Expose Pressure Estimate

Device/Provider status SHALL expose memory pressure useful to pool policy.

#### Scenario: External process uses GPU memory

Given available memory drops

When status refreshes

Then Runtime SHALL reduce pool-growth/admission accordingly.

### Requirement: Device SHALL Describe Allocation Granularity

Device capability metadata SHALL include logical allocation granularity or
alignment constraints.

#### Scenario: Memory-domain alignment

Given Device memory requires 64 KiB allocation granularity

When Memory Manager plans pool blocks

Then it can account for requirement without native heap handle.

### Requirement: Device Does Not Own Compaction

Device abstraction SHALL NOT expose compaction or pool-rebalancing operations.

#### Scenario: Fragmentation

Given pool needs compaction

When Runtime reacts

Then Memory Manager/Provider perform operation, not Device API.