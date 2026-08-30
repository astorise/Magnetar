## ADDED Requirements
### Requirement: Scheduler May Respect Session Placement Affinity

Scheduler SHALL be able to use logical Session/Plan placement affinity when admitting work.

#### Scenario: Decode Session owns GPU1 KV

Given GPU1 Plan remains healthy

When next token is scheduled

Then Scheduler may prefer the same ready Plan.

### Requirement: Scheduler Does Not Perform Device Placement Optimization

Scheduler SHALL not independently override Runtime MultiDevicePlacementPlan.

#### Scenario: GPU0 queue shorter

Given active Plan requires GPU1 stage

When Scheduler sees queue pressure

Then it cannot silently move stage without valid replacement Plan.

### Requirement: Admission Is Per Placement Plan

Scheduler SHALL admit work only when required Plan Devices/resources are
available.

#### Scenario: Mandatory GPU1 unavailable

Given Plan needs GPU0 and GPU1

When new request arrives

Then Scheduler does not start only half the Plan.

### Requirement: Cross Device Backpressure Is Supported

A slow downstream Device SHALL be able to create backpressure on upstream stages.

#### Scenario: GPU1 stage saturated

Given GPU0 produces faster than GPU1 consumes

When queues reach policy limit

Then Scheduler may throttle upstream submissions.
