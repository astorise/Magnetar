## ADDED Requirements

### Requirement: Device Performance Context May Be Observed

Device state MAY contribute contextual performance metadata, and such metadata SHALL be limited to device-level pressure/utilization indicators and SHALL NOT include raw workload or model content.

#### Scenario: GPU saturated

Given Kernel latency rises while Device pressure is high

When model aggregates evidence

Then pressure may be correlated with timing.

---

### Requirement: Device Pressure Is Not Kernel Correctness

High Device pressure SHALL not mark Kernel unqualified.

#### Scenario: Slow execution under load

Given Device is overloaded

When latency increases

Then Performance Model may mark degraded performance without changing
qualification state.