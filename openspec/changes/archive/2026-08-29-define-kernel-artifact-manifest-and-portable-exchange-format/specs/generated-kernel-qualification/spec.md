## ADDED Requirements

### Requirement: Manifest May Transport Qualification Evidence

Kernel Manifest MAY reference qualification evidence produced outside Runtime, and Runtime SHALL be able to consume such evidence references without depending on a specific external qualification producer.

#### Scenario: CI qualification

Given CI qualified generated Kernel

When bundle is imported

Then evidence digest/profile/suite metadata can be consumed.

---

### Requirement: Portable Evidence Is Not Automatically Current

Presence of qualification evidence SHALL not bypass freshness/revocation checks.

#### Scenario: Old suite

Given evidence used obsolete qualification suite

When current policy requires newer suite

Then evidence is insufficient.

---

### Requirement: Oracle Identity Is Preserved

Portable qualification reference SHOULD record oracle identity/version, and Runtime SHALL treat qualification evidence lacking oracle identity as unverifiable against a specific oracle.

#### Scenario: Reference CPU updated

Given old artifact was qualified against older Reference CPU

When evidence is inspected

Then oracle version remains known.

---

### Requirement: Evidence Integrity Is Content Addressed

Qualification evidence SHALL use immutable digest identity.

#### Scenario: Evidence modified

Given evidence blob changes

When digest verification runs

Then modification is detected.