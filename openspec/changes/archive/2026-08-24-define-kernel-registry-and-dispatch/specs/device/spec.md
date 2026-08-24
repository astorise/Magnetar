## ADDED Requirements
### Requirement: Device State Affects Kernel Eligibility

Device readiness, availability, pressure, loss, reset, and memory state SHALL
affect Kernel Registry eligibility and Dispatch revalidation.

#### Scenario: Device lost

Given selected Kernel targets Device A

When Device A is lost before dispatch

Then Runtime rejects, replans, or falls back according to policy.

---

### Requirement: Device Metadata Feeds Kernel Selection

Device metadata SHALL feed Kernel candidate filtering and ranking.

#### Scenario: Required feature missing

Given Kernel requires a Device feature flag

When Device metadata lacks that flag

Then the Kernel is not eligible.

---

### Requirement: Device Pressure Affects Kernel Ranking

Device pressure SHALL be available to Kernel ranking and MAY affect fallback.

#### Scenario: Device pressure high

Given two compatible Devices exist

And Device A has high pressure

When ranking runs

Then Runtime may prefer Device B according to policy.