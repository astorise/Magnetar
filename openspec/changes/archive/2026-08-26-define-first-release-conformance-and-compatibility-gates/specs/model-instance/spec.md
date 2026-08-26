## ADDED Requirements

### Requirement: Model Instance Release Gate

Model Instance lifecycle SHALL have release gate coverage.

#### Scenario: Active unload leak

Given active Model Instance unload leaks resources

When release validation runs

Then stable release is blocked.