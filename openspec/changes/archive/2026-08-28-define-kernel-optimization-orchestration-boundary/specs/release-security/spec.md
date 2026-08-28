## ADDED Requirements

### Requirement: Optimization Authority Is Outside Runtime Security Claim

Magnetar Runtime security boundary SHALL not claim control over arbitrary
external optimization infrastructure.

#### Scenario: External optimizer compromised

Given external optimizer produces malicious artifact

When artifact reaches Runtime

Then Runtime still enforces artifact trust, qualification and eligibility.

---

### Requirement: External Recommendation Is Untrusted Input

Optimization recommendation SHALL be validated as external input.

#### Scenario: Recommendation manipulated

Given recommendation claims candidate is qualified

But attached evidence is invalid

When Runtime validates it

Then promotion is rejected.

---

### Requirement: Optimization Credentials Not Packaged Into Runtime

Release artifacts SHALL not contain optimization-service secrets.

#### Scenario: Release build

Given CI uses external generator credential

When Magnetar binary/package is produced

Then credential is absent.