## MODIFIED Requirements

### Requirement: Memory Manager Releases Model Residency

Memory Manager SHALL release model residency memory when unload policy requires
it.

Releasing a Tensor Resource's residency SHALL remove its `TensorResidency`
record, not only change the state of the `MemoryAllocation` it references. A
resource whose Provider-owned storage and Memory Manager allocation have both
been released SHALL NOT continue to be reported as resident.

#### Scenario: Unload releases memory

Given a loaded model owns Device memory

When Runtime unloads the model

Then Memory Manager releases associated memory records.

#### Scenario: Released residency is not reported as resident

Given a Tensor Resource's Provider-owned storage and Memory Manager allocation have both been released, whether by weight materialization rollback or by Model Instance unload

When residency is queried for that Tensor Resource

Then Memory Manager reports no current residency for it
