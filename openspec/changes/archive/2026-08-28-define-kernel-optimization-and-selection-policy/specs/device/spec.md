## ADDED Requirements

### Requirement: Device Pressure May Influence Optimization

Runtime SHALL NOT let device pressure signals override eligibility constraints, though pressure MAY affect ranking among otherwise eligible candidates.

#### Scenario: GPU saturated

Given GPU and CPU candidates are both eligible

And policy allows dynamic pressure-aware selection

When GPU pressure is high

Then CPU may rank higher.

---

### Requirement: Device Unavailability Is Eligibility Constraint

Unavailable Device SHALL not remain eligible merely because benchmark is fast.

#### Scenario: GPU offline

Given GPU Device is unavailable

When ranking occurs

Then GPU candidates are excluded.

---

### Requirement: Device Metadata Does Not Perform Selection

Device SHALL expose state and capabilities but SHALL not select Kernels.

#### Scenario: Device health queried

Given Runtime reads Device health

When Kernel choice is made

Then Runtime policy owns decision.