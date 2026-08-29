## ADDED Requirements

### Requirement: Runtime Coordinates Bounded Autotuning

Runtime MAY coordinate bounded Kernel Autotuning during authorized cold/warm lifecycle, and Runtime SHALL NOT coordinate tuning outside authorized lifecycle phases.

#### Scenario: Model warmup

Given policy enables tuning

When Model Instance warms up

Then Runtime may benchmark eligible specializations.

---

### Requirement: Runtime Does Not Generate New Kernel Algorithms

Runtime Autotuning SHALL not become arbitrary implementation search.

#### Scenario: No specialization wins

Given all bounded variants perform poorly

When tuning ends

Then Runtime reports result/fallback rather than generating new algorithm.

---

### Requirement: Runtime Protects Active Inference

Runtime SHALL apply admission/resource policy to autotuning.

#### Scenario: GPU under critical load

Given active inference requires Device

When optional tuning requests same resources

Then tuning may be postponed or cancelled.

---

### Requirement: Runtime Uses Current Eligibility After Tuning

A tuning winner SHALL be revalidated when selected.

#### Scenario: Qualification revoked

Given tuning record remains cached

When Kernel is later revoked

Then Runtime does not use it.

---

### Requirement: Runtime Supports No-Tuning Deployment

Live Runtime Autotuning SHALL be optional.

#### Scenario: Embedded offline deployment

Given precomputed tuning record is shipped

When Runtime starts

Then no live tuning is required.