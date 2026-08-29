## ADDED Requirements

### Requirement: Import Acceptance Separation Conformance

Conformance SHALL prove receiving/parsing artifact does not make it accepted.

#### Scenario: Valid syntax but policy denied

Given manifest parses

When trust fails

Then artifact remains outside accepted cache.

---

### Requirement: Acceptance Preparation Separation Conformance

Conformance SHALL prove accepted artifact has no PreparedKernelId merely from
ingestion.

#### Scenario: CUBIN committed

Given import succeeds

When Provider state is inspected before preparation

Then no native prepared handle exists because of ingestion alone.

---

### Requirement: Acceptance Promotion Separation Conformance

Conformance SHALL prove successful commit does not replace active Kernel.

#### Scenario: Better candidate imported

Given active generation exists

When candidate commits

Then active generation stays unchanged.

---

### Requirement: Immutable Snapshot Conformance

Conformance SHALL prove source mutation cannot change committed bytes after
validation.

#### Scenario: Local bundle replaced mid-import

Given source path is modified

When transaction commits

Then committed digest/content matches staged validated snapshot.

---

### Requirement: Quarantine Isolation Conformance

Conformance SHALL prove quarantined artifacts cannot enter normal Registry
selection.

#### Scenario: Quarantined fastest Kernel

Given benchmark says fastest

When selection runs

Then candidate is absent.

---

### Requirement: Atomic Commit Conformance

Conformance SHALL prove partial logical artifact is never observable.

#### Scenario: Commit fault injection

Given failure occurs mid-publication

When readers query cache

Then they see prior state, not half-imported Kernel.

---

### Requirement: Idempotence Conformance

Conformance SHALL prove repeated identical import preserves artifact identity.

#### Scenario: Three retries

Given same bundle imported three times

When successful

Then one content identity exists while audit contains retry transactions.

---

### Requirement: Dedup Policy Conformance

Conformance SHALL prove existing blob cannot bypass new trust/policy checks.

#### Scenario: Digest already cached

Given new manifest is untrusted

When it references cached blob

Then manifest is still subject to current policy.

---

### Requirement: Revocation Re-Import Conformance

Conformance SHALL prove deleting/re-importing revoked artifact cannot restore
eligibility.

#### Scenario: Same digest returns

Given artifact is revoked

When imported again

Then revocation still blocks it.

---

### Requirement: External Authority Conformance

Conformance SHALL prove manifest URL cannot expand Runtime network authority.

#### Scenario: Arbitrary HTTPS locator

Given source not authorized

When ingestion runs

Then no request is made.

---

### Requirement: External Digest Conformance

Conformance SHALL prove fetched data is accepted only if digest matches.

#### Scenario: Registry object replaced

Given locator returns changed bytes

When ingested

Then transaction fails.

---

### Requirement: Quota Conformance

Conformance SHALL prove oversized/over-complex input fails within configured
limits.

#### Scenario: Huge decompressed bundle

Given bundle exceeds decompressed byte budget

When processed

Then ingestion aborts without unbounded allocation.

---

### Requirement: Cancellation Conformance

Conformance SHALL prove cancellation before commit leaves accepted state
unchanged.

#### Scenario: Cancel during validation

Given transaction has staged data

When cancelled

Then staged content is cleaned and accepted cache unchanged.

---

### Requirement: Commit Cancellation Race Conformance

Conformance SHALL prove concurrent cancel and commit produce exactly one
terminal result.

#### Scenario: Race

Given cancellation races atomic commit

When both complete

Then state is either committed or cancelled, never partial/ambiguous.

---

### Requirement: Active Inference Isolation Conformance

Conformance SHALL prove failed ingestion does not invalidate active
PreparedKernel.

#### Scenario: Broken N+1 bundle

Given N is executing

When N+1 fails validation

Then N remains valid.

---

### Requirement: Ingestion Redaction Conformance

Conformance SHALL prove audit/observability contain no raw source, binary,
credential, native handle, prompt, weight or KV payload by default.

#### Scenario: Artifact rejected

Given detailed failure exists

When exported

Then sensitive payloads remain redacted.