## ADDED Requirements

### Requirement: Optimization Tooling Belongs Outside Runtime Inference

CLI or companion tooling MAY orchestrate Kernel optimization, but SHALL NOT
delegate broader optimization-tooling authority to the Runtime Inference API.

#### Scenario: Future kernel optimize command

Given user starts `magnetar kernel optimize`

When command runs

Then broader tooling authority is not delegated to Runtime Inference API.

---

### Requirement: Tooling May Own External Credentials

Optimization tooling MAY own credentials required for external systems. Such credentials SHALL NOT be passed to Runtime as inference authority.

#### Scenario: AI generator API token

Given optimization command calls external generator

When token is used

Then credential remains CLI/tooling-owned and is not passed as Runtime
inference authority.

---

### Requirement: CLI Recommendation Cannot Bypass Runtime Policy

CLI SHALL NOT force recommended candidate active without normal eligibility and
promotion checks.

#### Scenario: User chooses generated candidate

Given candidate is unqualified

When CLI requests activation

Then Runtime/deployment policy rejects promotion.