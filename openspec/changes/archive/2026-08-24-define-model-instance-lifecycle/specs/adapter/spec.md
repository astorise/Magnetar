## ADDED Requirements

### Requirement: Adapter State Belongs To Model Instance Context

Runtime SHALL support associating active adapter state with a Model Instance, session, or
operation according to policy.

#### Scenario: Instance-level adapter

Given adapter A is activated at model-instance scope

When generation uses that instance

Then adapter A is part of the active instance context.

---

### Requirement: Adapter Mutation Affects Model Instance Lifecycle

Adapter merge, unmerge, activation, or deactivation SHALL affect Model Instance
readiness, mutability, cache compatibility, and batching compatibility.

#### Scenario: Merge adapter

Given adapter merge mutates model residency

When merge occurs

Then Model Instance records semantic mutation and invalidates dependent state
according to policy.
