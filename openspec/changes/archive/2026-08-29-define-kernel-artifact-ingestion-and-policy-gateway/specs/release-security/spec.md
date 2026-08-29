## ADDED Requirements

### Requirement: Kernel Ingestion Is Security Boundary

Kernel Artifact Ingestion Gateway SHALL be treated as a release security
boundary for deployments accepting external Kernel artifacts.

#### Scenario: Generated Kernel feature enabled

Given release supports external Kernel bundles

When security review occurs

Then ingestion validation, quarantine, limits and trust policy are in scope.

---

### Requirement: Release Does Not Equate Bundled With Trusted

Kernel Artifact shipped in distribution SHALL still have explicit integrity and
trust status.

#### Scenario: Bundle packaged with release

Given artifact is physically present in release archive

When security report evaluates it

Then packaging presence alone does not imply cryptographic trust.

---

### Requirement: Production Fail-Closed Trust Is Documented

Release security documentation SHALL state any weakened trust mode required to
accept unsigned/generated Kernel artifacts.

#### Scenario: Development mode allows unsigned local artifact

Given feature exists

When release documentation is generated

Then it is identified as development/weakened policy rather than normal
production trust.

---

### Requirement: Ingestion Audit Supports Release Evidence

Release/conformance evidence SHALL be traceable to the originating ingestion
transaction when it includes Kernel ingestion policy fingerprint or accepted
artifact digests.

Such evidence MAY include Kernel ingestion policy fingerprint and accepted
artifact digests.

#### Scenario: Release ships optimized Kernel

Given optimized Kernel included

When release inventory is generated

Then manifest/blob identities and ingestion validation status are auditable.