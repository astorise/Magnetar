## ADDED Requirements

### Requirement: Cache Storage Is Not Memory Residency

Artifact cache storage SHALL be distinct from Runtime memory residency.

#### Scenario: Cached artifact

Given artifact exists in cache

When Runtime memory is inspected

Then artifact tensors are not resident unless loaded through Memory Manager.

---

### Requirement: Model Loading Materializes From Cache Through Memory Manager

When loading from cache, Model Loading SHALL still materialize Tensor Resources
through Memory Manager.

#### Scenario: Load cached weights

Given cached weights are valid

When Model Loading materializes weights

Then Memory Manager tracks resulting Tensor Resources.