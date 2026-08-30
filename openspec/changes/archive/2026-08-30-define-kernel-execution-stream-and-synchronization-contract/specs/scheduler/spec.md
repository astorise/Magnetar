## ADDED Requirements

### Requirement: Scheduler Does Not Own Native Streams

Scheduler SHALL operate on logical workload/admission concepts rather than
Provider-native stream or queue objects.

#### Scenario: CUDA scheduling

Given Scheduler chooses latency-sensitive decode work

When Runtime dispatches it

Then Scheduler does not receive CUstream.

### Requirement: Scheduler May Supply Concurrency Intent

Scheduler SHALL be allowed to supply logical concurrency/priority hints to
execution subsystem.

#### Scenario: Background prefill

Given decode is latency-sensitive

When Scheduler orders work

Then Runtime may map workloads to different logical ExecutionStreams according
to policy.

### Requirement: Scheduler Observes Completion Logically

Scheduler SHALL be allowed to use CompletionToken terminal state to advance batches and
Sessions.

#### Scenario: Decode batch completes

Given token transitions completed

When Scheduler processes event

Then next logical step may be admitted.

### Requirement: Scheduler Does Not Treat Cancellation As Completion

Scheduler SHALL distinguish cancelled request from physical execution
quiescence where resource lifetime depends on completion.

#### Scenario: Cancelled sequence slot

Given Device work remains pending

When Scheduler wants to reuse slot

Then Memory/Runtime readiness prevents unsafe reuse.
