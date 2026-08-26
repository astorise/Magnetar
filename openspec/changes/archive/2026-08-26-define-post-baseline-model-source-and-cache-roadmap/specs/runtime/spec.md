## ADDED Requirements

### Requirement: Runtime Source Access Is Authorized

Runtime SHALL only access model sources through explicit authorized contracts.

#### Scenario: Arbitrary path

Given Runtime receives arbitrary path string

When it attempts source resolution

Then it rejects it unless wrapped in authorized source metadata.

---

### Requirement: Runtime Cache Access Is Policy-Controlled

Runtime cache lookup and mutation SHALL be controlled by policy.

#### Scenario: Cache insert denied

Given policy denies cache mutation

When artifact normalization completes

Then Runtime does not insert it into cache.