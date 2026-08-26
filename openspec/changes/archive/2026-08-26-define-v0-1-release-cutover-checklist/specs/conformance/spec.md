## ADDED Requirements

### Requirement: Conformance Reports Required For Cutover

Cutover SHALL require conformance reports for required baseline suites.

#### Scenario: Missing E2E report

Given E2E local conformance report is missing

When cutover validates artifacts

Then release is blocked.

---

### Requirement: Cutover Conformance Reports Are Redacted

Cutover conformance reports SHALL be redacted by default.

#### Scenario: Prompt in report

Given failure includes prompt text

When report is generated

Then raw prompt is absent by default.