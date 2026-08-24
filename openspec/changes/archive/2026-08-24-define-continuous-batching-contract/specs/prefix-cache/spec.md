## ADDED Requirements

### Requirement: Prefix Cache May Reduce Prefill Work In Batching

Runtime SHALL allow Prefix Cache hits to reduce prefill work before batching where policy permits.

#### Scenario: Prefix hit

Given an operation has an exact Prefix Cache hit

When Scheduler plans prefill

Then it schedules only the remaining work according to Runtime plan.

---

### Requirement: Prefix Cache Policy Applies During Batching

Batching SHALL not bypass Prefix Cache privacy, sharing, or Resource Affinity
policy.

#### Scenario: Sharing denied

Given a prefix entry matches

But sharing policy denies reuse

When batching plans prefill

Then Runtime does not reuse the entry.
