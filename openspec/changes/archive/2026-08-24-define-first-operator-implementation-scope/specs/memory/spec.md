## ADDED Requirements
### Requirement: First Scope Memory Is Host-Compatible

The first operator implementation scope SHALL be executable with host memory
through Reference CPU Provider.

#### Scenario: Host tensor

Given required-now operator input is host-resident

When CPU Kernel dispatch runs

Then Memory Manager tracks input/output host residency.

---

### Requirement: Unsupported Layout Movement Is Explicit

Unsupported layout movement SHALL be explicit when first scope requires layout
conversion through Memory Manager and graph planning.

#### Scenario: Non-contiguous input

Given input layout is unsupported

When first scope planning runs

Then Runtime inserts explicit conversion where available or rejects the graph.
