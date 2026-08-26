## ADDED Requirements

### Requirement: Security Gates Are Release-Blocking

Security and supply-chain gates SHALL block stable release on failure.

#### Scenario: License failure

Given license audit fails without exception

When stable release is attempted

Then release is blocked.

---

### Requirement: Security Exceptions Appear In Release Reports

Security exceptions SHALL appear in release reports.

#### Scenario: Accepted advisory

Given advisory exception is approved

When report is generated

Then exception metadata and mitigation are included.