## ADDED Requirements

### Requirement: Model Loading Produces Model Instance

Successful Model Loading SHALL produce or update a Runtime-owned Model Instance
when target usage requires inference execution.

#### Scenario: Loading complete

Given model materialization succeeds

When Runtime publishes the result

Then a Model Instance is created or updated.

---

### Requirement: Model Loading Does Not Bypass Instance Readiness

Successful materialization alone SHALL not imply Model Instance readiness.

#### Scenario: Provider warmup pending

Given model weights are materialized

But required Provider warmup is pending

When Runtime reports instance state

Then the Model Instance is not yet ready.

---

### Requirement: Loading Failure Prevents Ready Instance

If Model Loading fails, Runtime SHALL not expose a ready Model Instance.

#### Scenario: Materialization failed

Given materialization fails

When loading ends

Then Runtime reports failed loading and no ready instance is available.
