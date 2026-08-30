## ADDED Requirements
### Requirement: Prepared Plan Captures Exact Device Bindings

PreparedExecutionPlan SHALL identify concrete Provider/Device for every
multi-Device segment.

#### Scenario: Pipeline Plan

Given stage 0 uses GPU0 and stage 1 uses GPU1

When Plan becomes ready

Then these bindings are explicit and generation-stable.

### Requirement: Prepared Plan Captures Movement Edges

Cross-Device Resource transitions SHALL be present in the prepared execution
strategy.

#### Scenario: Activation transfer

Given GPU0 output feeds GPU1 input

When Plan is prepared

Then transfer/peer-access edge exists between stages.

### Requirement: Prepared Plan Binds Per Device Allocation Plans

A multi-Device Prepared Plan SHALL be able to reference distinct AllocationPlans for each
participating Device.

#### Scenario: Different workspace pools

Given GPU0 and GPU1 require different workspace geometry

When Plan is built

Then memory slots bind to corresponding Device pools.

### Requirement: Placement Guards Are Checked

Prepared Plan SHALL validate hard placement assumptions before use.

#### Scenario: Required peer path disappeared

Given Plan requires direct peer transfer

When capability is no longer available

Then Plan does not execute unchanged.

### Requirement: Placement Staleness Does Not Mutate Plan

A more attractive Device placement SHALL result in new Plan generation rather
than in-place binding rewrite.

#### Scenario: GPU pressure shifts

Given current Plan is still safe

When alternative placement becomes better

Then old Plan may be marked stale and replacement built.

### Requirement: Device Loss Hard Invalidates Plan

A Plan requiring lost Device SHALL receive no new work.

#### Scenario: Stage Device unavailable

Given GPU1 is lost

When new invocation tries active Plan

Then guard fails/invalidation is enforced.

### Requirement: In Flight Placement Remains Coherent

An in-flight Plan generation SHALL retain its original Device bindings until
safe completion or failure.

#### Scenario: Replacement published mid-stage

Given invocation is executing stage 0

When new Plan generation is activated

Then the invocation does not silently jump to a mixed generation.
