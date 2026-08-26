## ADDED Requirements

### Requirement: Cutover Requires Release Gates

Cutover SHALL require all release conformance gates to pass or be validly
skipped as allowed.

#### Scenario: Disallowed skip

Given OpenSpec validation is skipped

When cutover runs

Then release is blocked.

---

### Requirement: Cutover Reports Gate Status

Cutover SHALL include gate status in release reports.

#### Scenario: Gate report

Given gates complete

When cutover generates reports

Then pass/fail/skipped/exception status is visible.