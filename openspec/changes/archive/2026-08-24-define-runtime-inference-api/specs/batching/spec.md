## ADDED Requirements

### Requirement: Inference API Requests Participate In Batching

Runtime Inference API SHALL allow generation requests to participate in Continuous Batching according to Runtime policy.

#### Scenario: Queued request

Given multiple compatible generation requests are queued

When Scheduler forms a batch

Then Runtime may batch them while preserving per-request output streams.

---

### Requirement: Batching Remains Runtime-Owned

Runtime Inference API callers SHALL not directly mutate batch slots or Scheduler internal state.

#### Scenario: Caller sets batch slot

Given caller submits requested batch slot ID

When Runtime validates request

Then caller-provided batch slot is ignored or rejected.