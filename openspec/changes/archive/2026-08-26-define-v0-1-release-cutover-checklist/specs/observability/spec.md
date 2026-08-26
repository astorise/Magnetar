## ADDED Requirements

### Requirement: Cutover Observability Is Redacted

Cutover observations and reports SHALL be redacted by default.

#### Scenario: Release observation

Given cutover records failed secret scan

When report is emitted

Then secret value is not printed.

---

### Requirement: Cutover Events Are Correlatable

Cutover SHALL record correlation between gates, reports, artifacts, and release
metadata.

#### Scenario: Gate failure

Given Runtime gate fails

When cutover report is inspected

Then failure can be correlated to gate, target, feature set, and artifact.