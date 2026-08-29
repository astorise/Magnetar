## ADDED Requirements

### Requirement: Generation Supplies Non-Sensitive Workload Context

Generation MAY expose workload dimensions required for Kernel performance bucketing, and exposed workload dimensions SHALL exclude raw prompt or user content.

#### Scenario: Decode step

Given sequence length and batch size are known

When Performance Model records execution

Then those values may be bucketed without prompt content.

---

### Requirement: Generation Does Not Wait For Adaptive Re-Tuning

Active generation SHALL not synchronously wait for new performance tuning
unless an explicit non-hot-path readiness policy applies.

#### Scenario: Regression detected

Given known-good fallback exists

When decode continues

Then generation can continue with selected/fallback Kernel while re-tuning is
handled separately.