## ADDED Requirements

### Requirement: Canonical Manifest Conformance

Conformance SHALL prove equivalent supported manifests canonicalize
deterministically.

#### Scenario: Object key order differs

Given same fields appear in different JSON order

When canonicalized

Then digest is identical.

---

### Requirement: Duplicate-Key Conformance

Conformance SHALL reject duplicate JSON object keys.

#### Scenario: Duplicate source format

Given manifest supplies conflicting duplicate field

When parsed

Then failure is structured and fail-closed.

---

### Requirement: Blob Integrity Conformance

Conformance SHALL reject payload digest mismatch.

#### Scenario: One byte modified

Given compiled blob differs by one byte

When bundle is verified

Then preparation does not occur.

---

### Requirement: Filename Independence Conformance

Conformance SHALL prove filename/extension is not artifact format authority.

#### Scenario: CUBIN stored as digest path

Given no extension exists

When descriptor declares compatible CUBIN

Then format is resolved from metadata.

---

### Requirement: Optional Extension Forward Compatibility

Unknown optional extension SHALL not invalidate otherwise valid manifest.

#### Scenario: Vendor optional extension

Given Runtime does not understand extension

When extension marked optional

Then core manifest may still validate.

---

### Requirement: Required Extension Fail-Closed

Unknown required extension SHALL reject manifest.

#### Scenario: Required future semantic extension

Given Runtime does not support it

When manifest is loaded

Then manifest is unsupported.

---

### Requirement: Provenance Does Not Grant Trust

Conformance SHALL prove publisher/source/generator claims alone cannot grant
trusted status.

#### Scenario: Fake known publisher

Given malicious manifest writes known publisher name

When trust policy evaluates it

Then claim alone is insufficient.

---

### Requirement: Recommendation Does Not Promote

Conformance SHALL prove imported recommendation does not mutate active Kernel
Registry.

#### Scenario: Manifest says best-latency

Given candidate is imported

When no promotion occurs

Then currently active Kernel remains unchanged.

---

### Requirement: Qualification Evidence Revalidation

Conformance SHALL prove evidence reference is checked against current policy.

#### Scenario: Evidence revoked

Given valid digest references revoked evidence

When candidate is evaluated

Then it is not qualified.

---

### Requirement: External Reference Does Not Grant Network Authority

Conformance SHALL prove arbitrary external locator cannot trigger unrestricted
network access.

#### Scenario: Manifest references attacker URL

Given Runtime source policy denies it

When imported

Then no network access occurs.

---

### Requirement: Path Traversal Conformance

Conformance SHALL reject malicious bundle paths.

#### Scenario: `../../x`

Given archive contains traversal entry

When bundle is loaded

Then loading fails.

---

### Requirement: Symlink Conformance

Conformance SHALL reject symlink escape.

#### Scenario: digest path symlink

Given archive points blob path outside bundle

When loaded

Then bundle fails validation.

---

### Requirement: Repack Identity Conformance

Conformance SHALL prove archive metadata does not alter logical artifact
identity.

#### Scenario: Different ZIP compression

Given logical bytes are same

When bundle is repacked

Then manifest/blob logical digests remain identical.

---

### Requirement: Native Handle Exclusion Conformance

Conformance SHALL prove portable manifest contains no process-local native
execution authority.

#### Scenario: PreparedKernelId serialized

Given producer attempts to treat prepared ID as portable Kernel artifact

When validation runs

Then it is not accepted as executable artifact identity.

---

### Requirement: Parsing Side-Effect Conformance

Conformance SHALL prove parsing malicious manifest invokes no Provider compile,
prepare, execute or promotion operation.

#### Scenario: Parse-only validation

Given bundle is inspected

When validation fails

Then active Runtime execution state is unchanged.

---

### Requirement: Malformed Bundle Failure Atomicity

Conformance SHALL prove malformed imported bundle cannot disturb active
known-good Kernel.

#### Scenario: Replacement bundle corrupt

Given Kernel generation N is active

When invalid N+1 bundle is imported

Then N remains active.