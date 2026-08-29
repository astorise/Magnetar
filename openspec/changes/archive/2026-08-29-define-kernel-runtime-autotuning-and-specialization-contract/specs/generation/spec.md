## ADDED Requirements

### Requirement: Generation Does Not Start Autotuning In Token Loop

Prefill/decode execution SHALL use already selected/default Kernel rather than
launching tuning synchronously.

#### Scenario: Decode tuning cache miss

Given no tuning result exists

When next token is generated

Then generation proceeds with known-good candidate or explicit fallback.

---

### Requirement: Prefill And Decode Have Independent Tuning Context

Autotuning MAY produce separate recommendations for prefill and decode, and Runtime SHALL NOT conflate prefill and decode tuning context when selecting a winner.

#### Scenario: Different winners

Given Kernel A is best for prefill and Kernel B for decode

When generation executes

Then Runtime may use A and B at their respective safe phases.

---

### Requirement: Dynamic Tuning Publication Preserves In-Flight Execution

Background tuning result SHALL not replace Kernel state underneath active
invocation.

#### Scenario: New tuning winner published between decode operations

Given previous operation uses Kernel generation N

When N+1 becomes preferred

Then N remains valid until operation completes.