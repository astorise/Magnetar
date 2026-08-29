## ADDED Requirements

### Requirement: Ingestion Lifecycle Observability

Gateway SHALL emit redacted ingestion lifecycle observations.

#### Scenario: Successful import

Given transaction completes

When observations are inspected

Then staging, validation, decision and commit events may be correlated by
transaction ID.

---

### Requirement: Observed And Claimed Source Distinguished

Observability SHALL distinguish observed source from artifact claims.

#### Scenario: Provenance conflict

Given local import claims vendor origin

When event is emitted

Then both may be reported as distinct fields rather than one authenticated
source.

---

### Requirement: Quarantine Is Observable

Quarantine decision SHALL have structured reason.

#### Scenario: Signature missing

Given policy requires authenticated signature

When artifact is quarantined

Then reason indicates trust evidence unresolved/missing.

---

### Requirement: Ingestion Payload Redaction

Observability SHALL not expose raw Kernel artifact bytes by default.

#### Scenario: Compiler binary digest fails

Given blob mismatch occurs

When diagnostic is emitted

Then digest metadata may be exposed while raw binary is not.

---

### Requirement: Credentials Are Redacted

Artifact Source credentials SHALL never appear in ingestion observations.

#### Scenario: External fetch denied

Given authenticated registry source fails

When error is emitted

Then tokens/passwords are absent.