## ADDED Requirements

### Requirement: Execution Stream Lifecycle Is Observable

Runtime SHALL expose logical stream lifecycle observations.

#### Scenario: Stream drains

Given Runtime retires Model Instance

When stream begins draining

Then logical stream identifier/state may be observed.

### Requirement: Completion Lifecycle Is Observable

Runtime SHALL be able to expose sampled/redacted CompletionToken state
transitions.

#### Scenario: Kernel finishes asynchronously

Given token moves pending to completed

When instrumentation enabled

Then completion duration may be observed.

### Requirement: Dependency Wait Is Observable

Runtime SHALL permit diagnosis of dependency-related latency.

#### Scenario: Transfer waits for compute

Given transfer submission depends on pending compute

When diagnostics run

Then logical dependency wait may be recorded.

### Requirement: Cancellation State Is Observable

Observability SHALL distinguish request cancellation from physical completion.

#### Scenario: Kernel continues after cancellation

Given Provider cannot interrupt Kernel

When diagnostics are viewed

Then cancellation may be recorded while completion remains pending.

### Requirement: Resource Reuse Delay Is Observable

Runtime SHALL be able to expose that resource reuse was delayed by in-flight
completion.

#### Scenario: Workspace pressure

Given workspace remains fenced

When new allocation is required

Then observability may identify synchronization-related retention.

### Requirement: Native Synchronization Is Redacted

Observability SHALL NOT expose native stream, event, semaphore, fence, command
queue, Tensor address, model weight, KV content, prompt, secret, or credential.

#### Scenario: CUDA diagnostic

Given native CUDA event exists

When trace is exported

Then only logical token/stream identifiers are exposed.
