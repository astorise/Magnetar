## ADDED Requirements
### Requirement: Placement Decisions Are Observable

Runtime SHALL expose redacted multi-Device placement decisions.

#### Scenario: Pipeline split

Given blocks are divided GPU0/GPU1

When diagnostics are requested

Then stage-to-stable-Device mapping may be shown.

### Requirement: Tensor Partition Is Observable

Runtime SHALL be able to expose logical partition metadata without payload contents.

#### Scenario: Weight split

Given tensor is partitioned in two shards

When diagnostics run

Then shard count, logical ranges, and stable Device bindings may be reported.

### Requirement: Cross Device Movement Is Observable

Runtime SHALL permit observing Device-to-Device transfer classes and sizes.

#### Scenario: Peer transfer

Given activation moves directly GPU0 to GPU1

When trace is emitted

Then transfer class/bytes may be reported without native peer handle.

### Requirement: Placement Failure Is Explainable

Runtime SHALL expose structured exclusion/invalidation reason.

#### Scenario: GPU1 rejected for stage

Given workspace insufficient

When diagnostics inspect placement

Then memory-infeasible reason is available.

### Requirement: Degraded Placement Is Observable

Activation of degraded/fallback plan SHALL be explicit.

#### Scenario: One GPU lost

Given one-GPU fallback activates

When observability updates

Then degraded mode is distinguishable from normal placement.

### Requirement: Multi Device Observability Redacts Native State

Observability SHALL NOT expose native Device pointers, peer handles, queues,
streams, model data, KV contents, prompts, secrets, or credentials.

#### Scenario: Peer copy trace

Given Provider uses native CUDA handles

When trace is exported

Then only logical stable identities remain.
