## ADDED Requirements
### Requirement: Generation May Reference Session

A GenerationRequest SHALL be able to reference an Inference Session.

When a session is referenced, Generation SHALL use session model binding,
tokenizer binding, policy, memory budget, cancellation state, and observability
correlation.

#### Scenario: Generate with session

Given a ready session

When generation runs inside it

Then session bindings are applied.

---

### Requirement: Generation Supports One-Shot Session

Generation SHALL support Runtime one-shot requests through implicit short-lived session semantics.

#### Scenario: One-shot generation

Given a caller submits a one-shot generation request

When Runtime executes it

Then Generation uses session semantics and cleans up session-scoped resources
after completion.

---

### Requirement: Generation Respects Session Concurrency

Generation SHALL respect session concurrency policy.

#### Scenario: Session active

Given a session allows only one active operation

And one generation is active

When a second generation is requested

Then Runtime queues or rejects according to session policy.

---

### Requirement: Generation Respects Session Cancellation

Generation SHALL observe session cancellation state.

#### Scenario: Session cancelled

Given a session is cancelled

When generation is active

Then generation stops or fails according to cancellation policy.

---

### Requirement: Generation Uses Session Streaming State

When streaming generation runs inside a session, streaming state SHALL be
associated with the session operation.

#### Scenario: Streaming chunks

Given generation streams token IDs

When tokenizer streaming decode has partial state

Then the session operation preserves that state.