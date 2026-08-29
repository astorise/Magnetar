## ADDED Requirements

### Requirement: Portable Kernel Manifest Is Untrusted Input

Kernel Exchange Bundle SHALL be treated as untrusted input until validation and
policy evaluation complete.

#### Scenario: Release includes generated Kernel bundle

Given bundle ships with release

When release security gates run

Then manifest/blob integrity and trust policy are validated.

---

### Requirement: Signature Metadata Is Not Signature Verification

Presence of signature envelope metadata SHALL not be reported as verified
signature unless cryptographic verification actually succeeded.

#### Scenario: Algorithm and digest only

Given signature record contains no authentic signature proof

When security report is generated

Then artifact is not reported as cryptographically verified.

---

### Requirement: Bundle Cannot Expand Runtime Authority

Portable manifest SHALL not grant filesystem/network/process/secret authority.

#### Scenario: Manifest asks for compiler path

Given untrusted manifest contains arbitrary path metadata

When Runtime ingests it

Then path does not become ambient execution authority.

---

### Requirement: Release Reports Artifact Digests

Release MAY record digests of shipped Kernel manifests/blobs, and when recorded, digests SHALL identify the exact manifest/blob content shipped.

#### Scenario: Optimized baseline Kernel shipped

Given bundle is part of release

When artifact inventory is produced

Then its immutable identities can be audited.