## ADDED Requirements

### Requirement: Session Prefix Cache Policy

An Inference Session SHALL define or reference Prefix Cache policy.

Policy MAY enable, disable, scope, limit, share, retain, or evict prefix cache
entries.

#### Scenario: Prefix cache disabled

Given session policy disables prefix cache

When generation runs inside the session

Then Runtime skips Prefix Cache lookup.

---

### Requirement: Session Close Applies Prefix Cache Policy

When a session closes, session-scoped Prefix Cache entries SHALL be released,
retained, or transferred according to policy.

#### Scenario: Close session

Given a session owns session-scoped prefix entries

When the session closes

Then Runtime applies the configured cleanup policy.
