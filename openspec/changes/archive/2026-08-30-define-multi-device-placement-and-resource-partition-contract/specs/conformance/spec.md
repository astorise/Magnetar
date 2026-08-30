## ADDED Requirements
### Requirement: Runtime Placement Authority Conformance

Conformance SHALL prove Model Component and Provider cannot override concrete
Runtime Device placement.

#### Scenario: Provider preference conflicts

Given Runtime policy selects GPU0

When Provider prefers GPU1

Then final placement remains Runtime-authorized GPU0.

### Requirement: Device Neutral Model Component Conformance

Conformance SHALL prove portable Model Component does not require concrete GPU
identity.

#### Scenario: Same model on another machine

Given Device names differ

When Model Component is reused

Then portable graph remains valid.

### Requirement: Tensor Partition Bounds Conformance

Conformance SHALL reject invalid shard ranges.

#### Scenario: Shard outside Tensor

Given shard extends past parent dimension

When descriptor validates

Then partition fails.

### Requirement: Tensor Partition Completeness Conformance

Conformance SHALL detect unintended gaps/overlap.

#### Scenario: Missing range

Given non-replicated partition omits values

When validation runs

Then partition is rejected.

### Requirement: Replica Partition Distinction Conformance

Conformance SHALL prove full replica is not misclassified as partition shard.

#### Scenario: Two complete weight copies

Given both contain entire Tensor

When metadata is inspected

Then replication semantics are explicit.

### Requirement: Shard Cannot Masquerade As Full Tensor

Conformance SHALL prove non-partition-aware Kernel cannot consume one shard as
complete Tensor.

#### Scenario: MatMul expects full weight

Given only half-shard is supplied

When binding is validated

Then execution is denied.

### Requirement: No Implicit Collective Conformance

Conformance SHALL prove partition metadata does not silently introduce
all-gather/all-reduce.

#### Scenario: Consumer requires combined result

Given no collective/reconstruction operation exists

When Plan builds

Then Runtime reports unsupported path rather than inventing collective.

### Requirement: Explicit Cross Device Movement Conformance

Conformance SHALL prove Device boundary creates explicit movement/access
dependency.

#### Scenario: GPU0 output to GPU1

Given no peer direct-read use selected

When Plan executes

Then transfer operation exists.

### Requirement: Host Staging Policy Conformance

Conformance SHALL prove cross-Device placement cannot bypass host-staging
prohibition.

#### Scenario: Peer path missing

Given only host staging works

And policy forbids it

When Plan is validated

Then placement fails.

### Requirement: Multi Device Peer Capability Conformance

Conformance SHALL prove direct peer path requires explicit Device-pair
capability.

#### Scenario: Same-model GPUs

Given peer capability absent

When direct access is attempted

Then it is denied.

### Requirement: Per Device Capacity Conformance

Conformance SHALL prove aggregate free memory across Devices does not mask one
Device's infeasibility.

#### Scenario: GPU0 has space, GPU1 full

Given stage requires GPU1 local workspace

When admission runs

Then GPU0 free bytes do not satisfy it.

### Requirement: Heterogeneous Device Conformance

Conformance SHALL prove placement respects different Kernel/feature
capabilities.

#### Scenario: fp8 Kernel available only on GPU1

Given stage requires that Kernel

When candidates are built

Then GPU0 is excluded.

### Requirement: Transfer Aware Ranking Conformance

Conformance SHALL prove fastest isolated Kernel need not win if placement
movement cost dominates.

#### Scenario: Remote Device within host

Given GPU1 Kernel is faster but transfer is costly

When total placement cost ranks candidates

Then GPU0 may win.

### Requirement: Exact Prepared Placement Conformance

Conformance SHALL prove ready Plan uses exact Device binding it validated.

#### Scenario: Device preference changes

Given active Plan uses GPU0

When GPU1 becomes idle

Then existing Plan does not silently switch.

### Requirement: No Mid Flight Placement Migration Conformance

Conformance SHALL prove in-flight invocation keeps coherent Plan generation.

#### Scenario: Re-placement occurs

Given old stage execution is pending

When new Plan is published

Then pending work retains original binding until completion.

### Requirement: KV Locality Conformance

Conformance SHALL prove decode placement considers authoritative KV residency.

#### Scenario: Session KV on GPU1

Given GPU0 has no compatible KV copy

When GPU0 decode considered

Then required migration cost/path is explicit.

### Requirement: Session Migration Conformance

Conformance SHALL prove moving Session Device requires explicit state movement.

#### Scenario: GPU1 to GPU0

Given KV exists GPU1 only

When migration completes

Then GPU0 does not execute until required ResourceReadiness is satisfied.

### Requirement: Multi Device Device Loss Conformance

Conformance SHALL prove Plan requiring lost Device receives no new work.

#### Scenario: GPU1 lost

Given Plan requires GPU1

When request arrives

Then Plan fails guard or fallback activates.

### Requirement: Degraded Plan Validation Conformance

Conformance SHALL prove degraded plan independently satisfies capacity and
Kernel requirements.

#### Scenario: One-GPU fallback too large

Given model does not fit remaining GPU

When degraded plan is evaluated

Then it is rejected.

### Requirement: Device Recovery Conformance

Conformance SHALL prove recovered Device does not immediately receive work
without normal readiness/preparation.

#### Scenario: GPU reset completes

Given Device status returns healthy

When Runtime considers it

Then pools/Kernels/Plans are rebuilt/revalidated first.

### Requirement: Placement Cache Revalidation Conformance

Conformance SHALL prove cached placement cannot bypass current Device/memory/
peer/Kernel state.

#### Scenario: Cached plan requires peer path

Given peer capability no longer exists

When cache is loaded

Then Plan is stale/invalid.

### Requirement: Multi Device Native Handle Isolation Conformance

Conformance SHALL prove placement metadata contains no native Device or peer
handles.

#### Scenario: CUDA topology

Given Provider uses native IDs internally

When placement/debug state is inspected

Then only stable logical identities appear.

### Requirement: Multi Device Observability Redaction Conformance

Conformance SHALL prove placement traces contain no Device pointers, native
peer handles, native queues/streams, model data, KV contents, prompts, secrets,
or credentials.

#### Scenario: Device-loss report

Given detailed Provider internals exist

When report is exported

Then only safe logical metadata remains.
