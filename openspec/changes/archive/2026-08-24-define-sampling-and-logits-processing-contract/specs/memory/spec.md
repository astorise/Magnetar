## ADDED Requirements

### Requirement: Memory Manager Supports Sampling Buffers

Memory Manager SHALL account for Sampling temporary buffers.

Buffers MAY include logits, probabilities, masks, sorted token workspace, top-k
workspace, top-p workspace, RNG state, history buffers, and penalty workspace.

#### Scenario: Sampling workspace denied

Given top-p Sampling requires workspace memory

When Memory Manager denies allocation

Then Sampling fails with memory-allocation-failed or queues according to policy.

---

### Requirement: Memory Manager Controls Logits Materialization

Memory Manager SHALL participate in logits materialization decisions.

#### Scenario: Host logits materialization

Given Device-resident logits must be materialized on host

When Memory Manager policy denies staging

Then Sampling fails or chooses another compatible path.