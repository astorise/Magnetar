## ADDED Requirements

### Requirement: Inference API Cutover Status

Cutover SHALL confirm Runtime Inference API status in compatibility matrix.

#### Scenario: Missing status

Given Runtime Inference API has no status

When cutover runs

Then release is blocked.

---

### Requirement: Inference API Baseline Verified Before Tag

Runtime Inference API baseline gates SHALL pass before stable tag creation.

#### Scenario: API gate after tag

Given stable tag is created before API gate passes

When cutover validates sequence

Then release is invalid.