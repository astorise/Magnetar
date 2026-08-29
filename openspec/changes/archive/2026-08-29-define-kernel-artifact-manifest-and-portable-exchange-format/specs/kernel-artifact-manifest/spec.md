## ADDED Requirements

### Requirement: Versioned Kernel Manifest

Kernel Exchange Bundle SHALL contain an explicitly versioned Kernel Artifact
Manifest.

#### Scenario: v1 manifest

Given producer creates Kernel bundle

When manifest is parsed

Then schema identifies `magnetar:kernel-manifest@1.x`.

---

### Requirement: Canonical JSON Manifest

Version 1 portable manifest SHALL use canonical UTF-8 JSON for identity.

#### Scenario: Whitespace differs

Given two semantically identical manifests differ only in formatting

When canonicalized

Then they produce identical canonical bytes and manifest digest.

---

### Requirement: Duplicate Keys Rejected

Manifest parser SHALL reject duplicate JSON object keys.

#### Scenario: Two digest fields

Given artifact descriptor contains two `digest` properties

When parsed

Then manifest validation fails rather than choosing one ambiguously.

---

### Requirement: Content-Addressed Artifact References

Kernel payloads SHALL use immutable digest references.

#### Scenario: PTX blob

Given manifest references PTX

When blob is resolved

Then SHA-256 digest determines expected content identity.

---

### Requirement: Filename Does Not Determine Format

Artifact format SHALL be declared independently of filename/path.

#### Scenario: Misnamed blob

Given blob has no `.ptx` extension

But manifest declares `nvidia:ptx@9`

When compatibility is evaluated

Then format comes from validated descriptor rather than filename.

---

### Requirement: Extensible Format Identity

Manifest SHALL support unknown future Kernel source/compiled formats.

#### Scenario: Future DSL

Given format `vendor:new-ir@1`

When reader does not support execution of it

Then manifest can still be parsed structurally and compatibility may report
unsupported format.

---

### Requirement: Portable Operator Semantic Binding

Manifest SHALL identify portable Operator semantics implemented by Kernel.

#### Scenario: MatMul

Given Kernel implements MatMul

When descriptor is validated

Then Operator ID and semantic version are present.

---

### Requirement: Fused Semantic Binding

Fused Kernel SHALL declare the Operator composition it preserves.

#### Scenario: RMSNorm followed by MatMul

Given generated fused implementation exists

When manifest is validated

Then binding identifies RMSNorm -> MatMul rather than generic opaque fusion.

---

### Requirement: Explicit Specialization

Manifest SHALL make runtime-relevant specialization explicit.

#### Scenario: sm90 FP16 head-dim-128 Kernel

Given Kernel only supports those constraints

When manifest is parsed

Then architecture, dtype and head dimension constraints are explicit.

---

### Requirement: Manifest Trust Metadata Is Non-Authoritative

Publisher/source/signature metadata SHALL be evaluated by Runtime trust policy.

#### Scenario: Manifest claims trusted publisher

Given manifest says publisher is `trusted-company`

But no authenticated trust evidence exists

When trust is evaluated

Then publisher string alone cannot produce trusted status.

---

### Requirement: Recommendation Is Advisory

Manifest recommendation metadata SHALL not activate Kernel.

#### Scenario: Optimizer marks candidate best-latency

Given recommendation exists

When bundle is imported

Then Kernel is not promoted automatically.

---

### Requirement: Qualification Reference Is Revalidated

Manifest qualification reference SHALL be revalidated against current policy.

#### Scenario: Evidence revoked

Given manifest references formerly valid evidence

But evidence is revoked

When candidate is evaluated

Then candidate is not considered qualified.

---

### Requirement: Prepared Kernel State Is Forbidden

Portable manifest SHALL NOT transport process-local Prepared Kernel native state.

#### Scenario: Manifest contains CUDA function pointer

Given producer attempts to serialize native execution address

When validation runs

Then portable manifest is rejected or field is outside valid schema.

---

### Requirement: Manifest Parsing Has No Execution Side Effects

Parsing SHALL not compile, prepare, benchmark, promote or execute Kernel.

#### Scenario: Untrusted manifest parsed

Given malicious bundle is inspected

When parser runs

Then no Provider executable operation is triggered.