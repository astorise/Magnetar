## ADDED Requirements

### Requirement: E2E Uses Model Instance Lifecycle

E2E conformance SHALL create and use a Runtime-owned Model Instance.

#### Scenario: Instance ready

Given fixture model loads successfully

When Model Instance is published

Then it reaches ready state before session creation.

---

### Requirement: E2E Validates Instance Cleanup

E2E conformance SHALL validate Model Instance and related resources are cleaned
up or retained only according to policy.

#### Scenario: Unload after test

Given E2E test completes

When cleanup runs

Then Model Instance lifecycle follows policy and does not leak active resources.