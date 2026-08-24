## ADDED Requirements

### Requirement: Provider Advertises Batch Capabilities

Providers SHALL advertise batching capabilities where supported.

Capabilities MAY include max batch size, max sequence length, max total tokens,
supported dtypes, supported KV cache layout, paged attention support,
Provider-assisted sampling support, and workspace requirements.

#### Scenario: Batch unsupported

Given a Provider does not advertise batch execution

When Scheduler needs batching

Then Runtime does not assume batching support.

---

### Requirement: Provider Pressure Influences Batching

Provider readiness, admission, health, and pressure SHALL influence batch
formation and size.

#### Scenario: Provider prefer-not

Given Provider admission is prefer-not

When Scheduler forms a batch

Then policy may reduce or delay workload for that Provider.

---

### Requirement: Provider Batch Failure Is Structured

Provider failures during batched execution SHALL map to stable Runtime batching
or Provider errors.

#### Scenario: Batch execution fails

Given Provider fails a batched decode step

When Runtime handles the result

Then it maps errors per operation where possible.
