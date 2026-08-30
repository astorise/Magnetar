## ADDED Requirements
### Requirement: Pool Capacity Is Observable

Runtime SHALL expose redacted logical memory-pool capacity and pressure.

#### Scenario: KV pressure

Given KV pool approaches high watermark

When diagnostics are requested

Then configured/leased/reclaimable/pending-reclaim summaries SHALL be shown.

### Requirement: Allocation Reuse Is Observable

Runtime SHALL expose when planned storage is reused.

#### Scenario: Workspace reuse

Given same slot backs successive non-overlapping workspaces

When tracing is enabled

Then reuse SHALL be observed without native address.

### Requirement: Fragmentation Is Observable

Runtime SHALL permit distinguishing fragmentation from total-capacity OOM.

#### Scenario: Large block allocation fails

Given total free bytes remain substantial

When allocator reports fragmentation

Then diagnostic category reflects that reason.

### Requirement: Reservation Conflicts Are Observable

Runtime SHALL report when allocation fails due to protected capacity.

#### Scenario: Autotuning blocked

Given protected decode reservation prevents tuning workspace

When request denied

Then reason SHALL be reported as reservation conflict.

### Requirement: Pending Reclaim Is Observable

Logical release waiting on CompletionToken SHALL be observable.

#### Scenario: Cancellation retention

Given Device work remains in flight

When memory pressure exists

Then diagnostics SHALL report bytes pending reclaim.

### Requirement: Native Allocator State Is Redacted

Observability SHALL not expose native pool handles, Device pointers, mapped
addresses, Tensor contents, weights, KV contents, prompts, secrets, or
credentials.

#### Scenario: CUDA pool backing

Given Provider has native memory-pool handle

When trace is exported

Then only logical DeviceMemoryPoolId and aggregate metadata appear.