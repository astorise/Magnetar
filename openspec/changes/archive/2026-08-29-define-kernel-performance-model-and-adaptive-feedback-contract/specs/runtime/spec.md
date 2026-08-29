## ADDED Requirements

### Requirement: Runtime Coordinates Adaptive Performance Feedback

Runtime MAY collect and aggregate bounded Kernel performance evidence, and collected evidence SHALL remain bounded and redacted per Kernel Performance Model requirements.

#### Scenario: Production inference

Given adaptive feedback enabled

When sampled Kernel executions complete

Then Runtime updates compatible Performance Model.

---

### Requirement: Runtime Keeps Feedback Separate From Correctness

Runtime SHALL NOT use performance observations to establish numerical
correctness.

#### Scenario: Unqualified candidate behaves quickly

Given latency is excellent

When qualification is absent

Then Runtime does not treat candidate as correct.

---

### Requirement: Runtime Schedules Re-Tuning Outside Hot Path

Runtime SHALL not block active token decode for adaptive benchmarking.

#### Scenario: Performance regression confirmed

Given decode is active

When Runtime reacts

Then it queues/schedules bounded re-tuning or fallback outside current hot path.

---

### Requirement: Runtime May Escalate Externally

Runtime MAY emit external Optimization Plane request when bounded tuning is insufficient, and Runtime SHALL NOT generate new Kernel source itself when escalating.

#### Scenario: No authorized specialization meets SLO

Given re-tuning fails

When policy allows escalation

Then Runtime emits optimization signal rather than invoking code generator.

---

### Requirement: Runtime Preserves Known-Good Execution

Failure of Performance Model or re-tuning subsystem SHALL not remove healthy
known-good Kernel.

#### Scenario: Performance aggregation fails

Given Kernel remains otherwise eligible

When feedback subsystem reports internal error

Then inference may continue according to fail-safe policy.