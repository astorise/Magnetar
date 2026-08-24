## ADDED Requirements

### Requirement: Generation May Query Prefix Cache

Generation SHALL support querying Prefix Cache before prefill when policy enables it.

#### Scenario: Prefix lookup before prefill

Given a session enables Prefix Cache

When a generation request starts

Then Runtime checks for reusable prefix state before full prefill.

---

### Requirement: Generation Handles Prefix Cache Hit

When Prefix Cache returns a compatible hit, Generation SHALL continue from the
reused prefix boundary.

#### Scenario: Exact hit

Given Prefix Cache returns exact-prefix-hit

When Generation proceeds

Then prefill work for the reused prefix is skipped or reduced according to
Runtime plan.

---

### Requirement: Generation Handles Prefix Cache Miss

When Prefix Cache returns miss or non-reusable result, Generation SHALL fall
back to normal prefill unless policy requires failure.

#### Scenario: Miss

Given Prefix Cache returns miss

When Generation runs

Then full prefill proceeds.

---

### Requirement: Generation Validates Prefix Cache Reuse

Generation SHALL not blindly trust Prefix Cache lookup results.

Runtime SHALL validate compatibility and policy before using the backing KV
cache.

#### Scenario: Incompatible hit

Given Prefix Cache returns an incompatible hit

When Generation validates reuse

Then Generation does not reuse the entry.
