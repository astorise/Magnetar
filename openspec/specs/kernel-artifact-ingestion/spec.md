# kernel-artifact-ingestion Specification

## Purpose
TBD - created by archiving change define-kernel-artifact-ingestion-and-policy-gateway. Update Purpose after archive.
## Requirements
### Requirement: Kernel Artifact Ingestion Gateway

External Kernel Exchange Bundles SHALL enter accepted Magnetar Kernel state
through an explicit ingestion gateway.

#### Scenario: External optimizer supplies bundle

Given optimizer produces Kernel Exchange Bundle

When bundle enters Magnetar deployment

Then it is processed through ingestion policy before accepted cache publication.

---

### Requirement: Import Does Not Imply Acceptance

Receiving Kernel Exchange Bundle SHALL NOT itself mark the artifact accepted.

#### Scenario: Bundle received

Given bundle bytes have arrived

When transaction is still validating

Then artifact is not visible as accepted Kernel candidate.

---

### Requirement: Ingestion Transaction

Each import SHALL use an isolated ingestion transaction.

#### Scenario: Two simultaneous imports

Given two bundles are imported concurrently

When one fails validation

Then the other transaction remains isolated.

---

### Requirement: Immutable Validation Snapshot

Gateway SHALL validate and commit the same immutable logical bytes.

#### Scenario: Source file changes after staging

Given source bundle is replaced after staging

When validation/commit continue

Then committed content remains staged validated snapshot, not replaced source.

---

### Requirement: Integrity Before Acceptance

Required artifact digests SHALL validate before acceptance.

#### Scenario: Compiled binary modified

Given one compiled blob fails digest check

When ingestion runs

Then transaction cannot reach accepted state.

---

### Requirement: Validation Has No Execution Side Effects

Ingestion validation SHALL not compile, prepare, execute or promote Kernel.

#### Scenario: Malicious manifest

Given malformed manifest is imported

When parsing fails

Then Provider execution/preparation has not been invoked.

---

### Requirement: Observed Source Is Separate From Manifest Provenance

Gateway SHALL distinguish externally observed import source from self-declared
manifest source.

#### Scenario: Local file claims vendor registry

Given local file manifest claims vendor source

When trust is evaluated

Then observed source remains local import.

---

### Requirement: Ingestion Source Does Not Imply Trust

Source classification SHALL not automatically produce trusted status.

#### Scenario: CI import

Given bundle comes through CI

When no authenticated artifact trust mechanism approves it

Then CI label alone is insufficient for trust.

---

### Requirement: Explicit Ingestion Decision

Ingestion SHALL produce explicit Accept, Quarantine, or Reject decision.

#### Scenario: Qualification missing

Given production policy requires qualification

When bundle has valid integrity but no qualification evidence

Then policy may quarantine or reject rather than silently accept for execution.

---

### Requirement: Quarantine Is Non-Executable

Quarantined Kernel Artifact SHALL not enter normal preparation or Registry
selection path.

#### Scenario: Trust unresolved

Given artifact is quarantined

When Registry discovers candidates

Then artifact is absent.

---

### Requirement: Acceptance Does Not Prepare

Accepted Kernel Artifact SHALL not automatically be considered Prepared Kernel.

#### Scenario: Valid CUBIN imported

Given ingestion commits CUBIN to accepted cache

When transaction completes

Then no PreparedKernelId is implied.

---

### Requirement: Acceptance Does Not Promote

Successful ingestion SHALL not alter active Kernel selection by itself.

#### Scenario: Imported candidate is faster

Given new accepted Kernel benchmarks faster

When ingestion completes

Then current active Kernel remains active until normal selection/promotion.

---

### Requirement: Atomic Commit

Accepted logical Kernel Artifact SHALL become visible atomically.

#### Scenario: Three required blobs

Given transaction imports three required blobs

When third cache publication fails

Then readers do not observe partially committed logical Kernel.

---

### Requirement: Idempotent Re-Import

Repeated import of identical content SHALL preserve same logical artifact
identity.

#### Scenario: CI retries upload

Given same bundle is imported twice

When both pass policy

Then duplicate executable identities are not created merely due to retry.

---

### Requirement: Cache Presence Does Not Bypass Policy

Existing content-addressed bytes SHALL not automatically satisfy current policy.

#### Scenario: Old cached artifact imported under stricter policy

Given bytes already exist

When new policy requires stronger trust

Then current trust policy is evaluated.

---

### Requirement: Revocation Survives Re-Import

Re-importing artifact SHALL not clear known revocation.

#### Scenario: Revoked digest deleted and uploaded again

Given same digest is re-imported

When gateway checks policy

Then known revocation still applies.

---

### Requirement: External Artifact References Require Authority

Gateway SHALL resolve external references only through authorized Artifact
Source.

#### Scenario: Manifest includes arbitrary URL

Given URL is outside authorized source policy

When ingestion runs

Then URL is not fetched.

---

### Requirement: External Content Still Requires Digest Validation

Transport authentication SHALL not replace artifact content validation.

#### Scenario: HTTPS download succeeds

Given downloaded bytes do not match declared digest

When staged

Then ingestion fails.

---

### Requirement: Ingestion Quotas

Gateway SHALL enforce configured resource limits.

#### Scenario: Decompression bomb

Given archive expands beyond transaction limit

When ingestion processes it

Then transaction fails before exhausting unbounded storage.

---

### Requirement: Transaction Cancellation

Cancellation of an uncommitted transaction SHALL leave the accepted cache
unchanged.

An uncommitted transaction MAY be cancelled by authorized caller.

#### Scenario: Large download cancelled

Given external retrieval is running

When cancellation occurs

Then further retrieval stops and accepted cache is unchanged.

---

### Requirement: Failure Atomicity

Failed ingestion SHALL not disturb active inference state.

#### Scenario: Replacement import corrupt

Given generation N is active

When N+1 import fails

Then generation N remains active.

---

### Requirement: Auditability

Gateway SHALL produce redacted audit evidence for every transaction.

#### Scenario: Artifact rejected

Given trust policy denies artifact

When audit is inspected

Then artifact digest, policy and rejection reason are available without raw
source contents.

