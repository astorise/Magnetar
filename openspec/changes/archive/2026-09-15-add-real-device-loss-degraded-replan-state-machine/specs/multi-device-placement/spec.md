## ADDED Requirements

### Requirement: Device-Loss Invalidation Is Verified Against a Real, Device-Derived Plan

The existing `MultiDevicePlacementPlan` state machine's `Ready` -> `Invalidated` transition, and its refusal to revert an `Invalidated` Plan back to `Ready`, SHALL be exercised against a Plan built from at least one real Device's own real metadata, not only synthetic fixtures. This requirement does not itself require real hardware-failure detection -- the invalidating transition MAY be caller-driven.

#### Scenario: A real, Device-derived Plan is invalidated and cannot silently revert

- **GIVEN** a `Ready` `MultiDevicePlacementPlan` built from real Device metadata
- **WHEN** it is transitioned to `Invalidated`
- **THEN** it no longer accepts new work
- **AND** a subsequent attempt to transition it back to `Ready` is rejected, leaving its state unchanged
