## ADDED Requirements

### Requirement: Generation Leases

The Runtime SHALL give every holder of dispatch-ready state a counted lease on each prepared kernel generation it can launch: a Ready Plan, a CPU or CUDA step capture, an in-flight request's plan. It SHALL release the lease when the holder is dropped, released, completed or cancelled.

#### Scenario: A capture holds a lease

- **WHEN** a pooled decode capture recorded launches of generation N
- **THEN** generation N's lease count includes that capture until the capture is released

### Requirement: Drain Before Release

The Runtime SHALL move a retired generation to draining and SHALL destroy it, letting the Provider release its native module, only when its lease count is zero and its rollback window allows.

#### Scenario: Retirement while requests run

- **WHEN** generation N is retired while a request still decodes on it
- **THEN** N stays loaded until that request completes and its capture is released, and is released afterwards

### Requirement: Promotion At Plan Boundaries

Promoting a generation SHALL publish a new registry revision without mutating any existing plan, capture or in-flight request. Plans resolved after the revision SHALL use the promoted generation. Step capture keys SHALL include the generations a capture launches, so a step resolved to a new generation records a new capture instead of replaying the old one.

#### Scenario: Promotion during decode

- **WHEN** N+1 is promoted while a request is mid-decode on N
- **THEN** that request finishes on N with its captures, and the next request's plan and captures use N+1

#### Scenario: A stale capture is not replayed

- **WHEN** a decode step resolves to N+1 and a capture keyed on N exists for the same store set
- **THEN** the step records a new capture keyed on N+1

### Requirement: Rollback And Revocation Without Restart

The Runtime SHALL roll back by re-promoting a retained generation within its rollback window, and SHALL report a structured error outside it. Revocation SHALL remove a generation from selection at once. In-flight work SHALL follow the configured `RevocationInFlightPolicy` (allow to complete, cancel if safe, fail closed), and the revoked generation SHALL drain before release.

#### Scenario: Rollback after a regression

- **WHEN** N+1 is rolled back within the window
- **THEN** new plans use N again, and N+1 drains and is released

#### Scenario: Revocation with fail-closed policy

- **WHEN** N+1 is revoked under `FailClosed` while requests run on it
- **THEN** those requests fail at the next step boundary with a structured error, and new requests select another eligible generation

### Requirement: No Leaked Native Handles

A generation SHALL reach destroyed, and its Provider module SHALL be released, once every lease is gone. A draining generation whose leases never reach zero SHALL be observable.

#### Scenario: Cancellation releases leases

- **WHEN** a request on a draining generation is cancelled
- **THEN** its leases are released, and the generation is destroyed if no other lease remains
