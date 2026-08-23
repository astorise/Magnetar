## ADDED Requirements
### Requirement: Memory Manager Tracks Session Memory

Memory Manager SHALL support session-scoped memory accounting.

#### Scenario: Session memory usage

Given a session allocates output token buffers

When Memory Manager reports usage

Then the allocation is associated with the session.

---

### Requirement: Memory Manager Enforces Session Budget

Memory Manager SHALL enforce session memory budgets according to Runtime policy.

#### Scenario: Session budget exceeded

Given a session has a memory budget

And a generation operation exceeds it

When memory admission is evaluated

Then Memory Manager rejects, queues, or delays according to policy.

---

### Requirement: Memory Manager Releases Session Resources

Memory Manager SHALL release session-scoped memory when the session closes,
expires, fails, or is cancelled according to policy.

#### Scenario: Close session

Given a session owns temporary workspace memory

When the session closes

Then Memory Manager releases the workspace.