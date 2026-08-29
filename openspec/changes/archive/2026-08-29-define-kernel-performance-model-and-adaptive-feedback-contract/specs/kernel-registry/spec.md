## ADDED Requirements

### Requirement: Registry Preserves Performance Evidence Identity

Registry SHALL associate performance evidence with the correct Kernel Artifact,
specialization, and generation context.

#### Scenario: New Kernel generation

Given generation N+1 replaces N

When performance evidence is queried

Then N observations do not silently become N+1 observations.

---

### Requirement: Registry Does Not Generate Performance Evidence

Kernel Registry SHALL not fabricate missing benchmark or online metrics.

#### Scenario: No online samples

Given candidate lacks observations

When ranking occurs

Then evidence remains missing rather than inferred from another candidate.