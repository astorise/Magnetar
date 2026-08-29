# Define Kernel Artifact Ingestion And Policy Gateway

## Why

Magnetar now defines:

```text
Kernel Artifact lifecycle
Provider Kernel Compilation
Generated Kernel Qualification
Kernel Cache and Hot Swap
Kernel Optimization and Selection
Optimization Orchestration
Kernel Exchange Bundle
Kernel Artifact Manifest
```

External systems can therefore produce portable Kernel artifacts and evidence.

What remains undefined is the authoritative boundary through which these
artifacts enter a Magnetar deployment.

An artifact arriving from:

```text
local file
CI
Tachyon
object storage
artifact registry
vendor tooling
AI optimization system
developer tooling
```

must not automatically become executable Runtime state.

Import is a security-sensitive state transition.

Magnetar therefore requires an explicit Kernel Artifact Ingestion Gateway that:

- stages artifacts immutably
- applies defensive limits
- validates structure
- validates digests
- validates semantic bindings
- evaluates trust
- validates qualification evidence
- evaluates ingestion policy
- quarantines uncertain artifacts
- commits accepted content atomically
- remains idempotent
- prevents partial imports
- prevents TOCTOU substitution
- prevents direct preparation or promotion
- records auditable evidence

## What Changes

This change defines:

- Kernel Artifact Ingestion Gateway
- Ingestion Transaction
- ingestion source metadata
- immutable staging
- ingestion states
- validation pipeline
- policy evaluation
- quarantine
- rejection
- acceptance
- atomic cache commit
- idempotence
- duplicate handling
- quotas
- resource accounting
- transaction deadlines
- cancellation
- TOCTOU resistance
- temporary artifact handling
- external artifact resolution
- trust/evidence revalidation
- manual approval hooks
- revocation interaction
- cleanup
- observability
- audit evidence
- conformance

## Core Rule

Importing an artifact SHALL NOT make it executable.

The following states SHALL remain distinct:

```text
received
staged
validated
accepted
cached
qualified
prepared
promoted
selected
executing
```

## Ingestion Gateway

Kernel Artifact Ingestion Gateway is the authoritative boundary responsible for
turning external Kernel Exchange Bundle data into validated internal artifact
content.

The gateway SHALL NOT be an inference execution API.

The gateway SHALL not directly:

- execute Kernel
- call Provider Kernel execution
- promote Kernel
- mutate active Kernel Registry selection
- grant trust
- grant qualification
- expose native handles

## Ingestion Transaction

Each import SHALL occur in an explicit Ingestion Transaction.

Conceptually:

```text
KernelIngestionTransaction
    id
    source
    policy
    limits
    state
    received_artifacts
    validation_results
    decision
```

A transaction SHALL isolate incomplete import state from the committed Kernel
Cache namespace.

## Transaction Identity

Every ingestion transaction SHALL have a stable opaque identifier.

The identifier MAY be used for:

- observability
- cancellation
- audit
- cleanup
- retry correlation

It SHALL NOT encode:

- native pointer
- filesystem handle
- process ID
- secret
- Provider handle

## Ingestion Source

The transaction SHALL record an Ingestion Source descriptor.

Source types MAY include:

```text
local-tooling
deployment-package
ci
external-artifact-source
optimization-campaign
tachyon-distributed
vendor-package
test-fixture
```

The vocabulary SHOULD remain extensible.

## Source Is Not Trust

Ingestion source metadata SHALL be descriptive.

```text
source == ci
```

SHALL NOT by itself imply:

```text
trusted
```

unless an independent authenticated source policy explicitly establishes such
trust.

A manifest's own `source` claim SHALL not override gateway-observed source
metadata.

## Observed Versus Claimed Source

Gateway SHOULD distinguish:

```text
observed ingestion source
manifest-declared provenance/source
```

They SHALL not be conflated.

Example:

```text
observed:
    local-import

manifest claims:
    vendor-registry
```

Trust evaluation sees both but SHALL not treat the manifest claim as observed
fact.

## Transaction States

Suggested transaction states are:

```text
created
receiving
staged
validating
policy-evaluating
quarantined
accepted
committing
committed
rejected
cancelled
timed-out
failed
cleaning
closed
```

State transitions SHALL be explicit and validated.

## Immutable Staging

Incoming bytes SHALL first enter an isolated staging area.

Staged data SHALL not be considered part of the trusted/accepted Kernel Cache.

After a staged blob has been integrity-addressed for the transaction, the bytes
used for validation SHALL remain immutable for the remainder of that
validation transaction.

## Immutable Snapshot

Validation SHALL operate against an immutable logical snapshot.

The source from which data was originally read MAY change afterward without
changing the bytes being validated.

This prevents:

```text
validate file A
source replaces file A
prepare replaced file B
```

TOCTOU substitution.

## Content Copy Or Stable Snapshot

Gateway MAY obtain immutable staging through:

- copying bytes
- immutable object-store version
- content-addressed local temporary object
- filesystem snapshot
- another mechanism providing equivalent stability

The mechanism is implementation-specific.

The property is mandatory:

```text
validated bytes == committed bytes
```

## Validation Pipeline

The logical ingestion pipeline SHALL be ordered approximately as:

```text
receive
  -> enforce transport/input limits
  -> immutable staging
  -> parse manifest
  -> structural validation
  -> schema/version validation
  -> path/archive safety
  -> canonicalization
  -> manifest identity
  -> blob existence
  -> blob sizes
  -> blob digests
  -> semantic validation
  -> artifact relationship validation
  -> trust evaluation
  -> qualification/evidence validation
  -> ingestion policy evaluation
  -> decision
  -> atomic commit
```

Implementations MAY combine safe adjacent operations but SHALL preserve the
security invariants implied by the ordering.

## Cheap Checks Before Expensive Checks

Gateway SHOULD perform cheap defensive checks before expensive operations.

For example:

```text
size limits
entry counts
schema/version
path safety
```

SHOULD precede:

```text
large digest computation
cryptographic verification
evidence processing
```

where possible.

This reduces denial-of-service exposure.

## Parsing Has No Execution Side Effects

Manifest parsing and structural validation SHALL NOT:

- invoke compiler
- invoke Provider.prepare
- execute Kernel
- run benchmarks
- run AI generator
- promote Registry candidate
- modify active Model Instance

## Integrity Before Preparation

No staged Kernel Artifact SHALL become eligible for Provider preparation until
required content integrity has passed.

## Semantic Validation

Gateway SHALL validate portable semantics before acceptance.

Validation SHOULD include:

- Operator IDs
- Operator versions
- fused semantic structure
- target constraints
- specialization
- dtype/layout metadata
- precision metadata structure
- dependency relationships
- artifact role consistency
- source/compiled relationships

Semantic validation SHALL NOT prove numerical correctness.

Numerical correctness belongs to qualification.

## Evidence Validation

Gateway MAY validate referenced qualification and benchmark evidence.

Validation SHALL distinguish:

```text
evidence structurally valid
evidence integrity valid
evidence currently accepted
```

A syntactically valid evidence record MAY still be:

```text
expired
revoked
incompatible
insufficient
```

## Trust Evaluation

Gateway SHALL invoke the configured artifact trust policy before granting an
ingestion decision that requires trust.

Trust SHALL not come automatically from:

- manifest publisher string
- manifest source kind
- local path
- cache presence
- CI label
- recognized format
- generator name
- optimization recommendation
- successful compilation

## Fail-Closed Production Trust

Production policy SHOULD fail closed when required artifact trust cannot be
established.

Development/test policy MAY explicitly allow weaker trust modes.

Weakened policy SHALL be explicit and observable.

## Trust Policy Result

Trust evaluation SHOULD result in structured states such as:

```text
trusted
untrusted
unsigned
unknown
denied
development-allowed
```

Exact representation MAY reuse existing artifact trust contracts.

## Ingestion Policy

Gateway SHALL apply an explicit Kernel Ingestion Policy.

Policy MAY define:

- accepted schema versions
- accepted artifact roles
- accepted formats
- maximum bundle size
- maximum blob size
- maximum artifact count
- trust requirements
- qualification requirements
- allowed sources
- external-reference policy
- required signatures
- target restrictions
- compiler/toolchain restrictions
- required/forbidden extensions
- quarantine behavior
- duplicate behavior

## Policy Versioning

Ingestion policy SHALL have identifiable version or fingerprint.

Audit results SHOULD record which policy produced an acceptance/quarantine/
rejection decision.

## Policy Precedence

Deployment/security constraints SHALL take precedence over artifact-provided
metadata.

The manifest SHALL not weaken ingestion policy.

## Ingestion Decision

The gateway SHALL produce an explicit decision.

Baseline decision classes SHOULD include:

```text
accept
quarantine
reject
```

## Accept

`accept` means:

- required structural validation passed
- required integrity passed
- required policy conditions passed
- artifact may be committed to accepted Kernel Cache namespace

It does NOT mean:

```text
prepared
promoted
selected
executing
```

## Quarantine

`quarantine` means artifact is retained in an isolated non-executable state for
later review or additional evidence.

Possible reasons include:

```text
trust unresolved
signature unavailable
qualification missing
evidence expired
unsupported optional policy state
manual review required
unknown generator provenance
future compatibility pending
```

Quarantine SHALL NOT imply acceptance for execution.

## Rejection

`reject` means the transaction SHALL NOT publish artifact into accepted Kernel
Cache namespace.

Rejected content MAY be retained temporarily for audit according to policy.

## Quarantine Namespace

Quarantined artifacts SHALL be logically separated from accepted Kernel Cache
content.

Kernel Registry SHALL not discover quarantined candidates as normal executable
candidates.

## Quarantine Does Not Prepare

Provider.prepare SHALL not be invoked for quarantined Kernel Artifact by the
normal Runtime path.

Explicit diagnostic tooling MAY inspect metadata without preparing it.

## Quarantine Review

A future or authorized management operation MAY re-evaluate a quarantined
transaction/artifact after:

- trust evidence arrives
- qualification completes
- policy changes
- operator approves it
- missing dependency becomes available

Re-evaluation SHALL use current policy.

## Manual Approval

Ingestion policy MAY include manual approval as one input.

Manual approval SHALL not silently bypass mandatory cryptographic, integrity,
semantic, or safety requirements unless policy explicitly defines the weaker
mode.

An operator clicking "approve" SHALL NOT repair digest mismatch.

## Approval Identity

Where manual approval is recorded, audit metadata SHOULD identify:

- approval event ID
- approver identity from external authenticated management context
- timestamp
- policy version
- approved artifact digest

The portable artifact itself SHALL not self-declare approval.

## Atomic Cache Commit

Accepted content SHALL be committed atomically from the perspective of Kernel
Cache readers.

Readers SHALL observe:

```text
old cache state
```

or:

```text
complete new committed state
```

They SHALL NOT observe partially imported required artifact sets.

## Transaction Commit Set

A transaction SHOULD determine the complete set of content-addressed blobs and
metadata to commit before publication.

If any mandatory commit operation fails, the transaction SHALL not publish a
partial logical Kernel Artifact.

## Content-Addressed Deduplication

Already-existing validated blobs MAY be reused by digest.

Deduplication SHALL preserve logical transaction atomicity.

A transaction may therefore commit metadata referencing content already
present in cache without copying duplicate bytes.

## Idempotent Import

Importing the same valid logical bundle repeatedly SHOULD be idempotent.

Repeated import SHALL NOT create distinct artifact identity solely because a
new transaction occurred.

Transaction history MAY still record each attempt.

## Duplicate Artifact

When identical digest content already exists, gateway MAY return:

```text
already-present
```

or equivalent accepted/idempotent status.

It SHALL re-evaluate policy where current policy requires it.

## Existing Content Does Not Bypass Current Policy

Presence of a digest in Kernel Cache SHALL NOT skip current ingestion/trust
checks when importing a new logical manifest that references it.

## Digest Collision Handling

If content exists under a digest but bytes do not match expected identity, the
cache entry SHALL be treated as corrupt/security failure.

Gateway SHALL fail closed.

## Transaction Failure Atomicity

If transaction fails before commit:

- accepted cache state remains unchanged
- active Kernel Registry remains unchanged
- active Prepared Kernels remain unchanged
- Model Instances remain unchanged

## Promotion Separation

Successful ingestion SHALL NOT directly invoke Kernel promotion.

The following is forbidden:

```text
ingest(bundle)
    -> automatically replace active Kernel
```

Promotion remains governed by qualification, selection, and promotion policy.

## Preparation Separation

Accepted ingestion MAY make an artifact eligible for later preparation.

It SHALL NOT necessarily prepare it during commit.

If deployment policy allows preparation immediately after commit, preparation
SHALL be a logically distinct post-ingestion operation.

A preparation failure SHALL NOT roll back a previously valid content-addressed
cache import unless policy explicitly requires coupled deployment semantics.

## Kernel Registry Separation

Kernel Registry MAY become aware of accepted candidates only after successful
transaction commit.

It SHALL NOT index staged or quarantined candidates into normal eligible
candidate discovery.

## Active Runtime Independence

Importing, rejecting, or quarantining a candidate SHALL not affect currently
active Kernel execution.

## Revocation Interaction

A newly imported artifact whose digest is already revoked SHALL be rejected or
quarantined according to policy.

Import SHALL NOT clear revocation.

## Revocation Persistence

Revocation metadata SHOULD be independent from cache presence.

Deleting and re-importing an artifact SHALL not erase known revocation merely
because its cache entry was recreated.

## Quarantine Promotion After Revocation

A revoked artifact SHALL not leave quarantine/denied state solely through
re-import.

A new authoritative revocation/trust decision is required.

## External Artifact Resolution

Bundles may contain external artifact references as defined by the exchange
format.

Gateway SHALL resolve such references only through an authorized Artifact
Source.

It SHALL NOT execute arbitrary URLs embedded in a manifest.

## Artifact Source Authority

Artifact Source capability SHOULD define allowed:

- source schemes
- registries
- domains/endpoints
- credentials
- namespaces
- size limits

These authorities belong to ingestion/management infrastructure, not inference
requests.

## Source Fetch Integrity

Externally fetched bytes SHALL be staged and validated against declared digest
before use.

TLS or authenticated transport MAY provide additional protection but SHALL NOT
replace content digest validation.

## Source Mutation

If an external locator now returns bytes different from declared digest,
ingestion SHALL fail.

Locator identity SHALL not override content identity.

## No Ambient Network Authority

If no Artifact Source authorizes a locator, gateway SHALL not fetch it.

## Download Limits

External artifact retrieval SHALL enforce:

- response byte limits
- timeout
- redirect policy
- content count
- total transaction budget

as applicable.

## Redirect Policy

Artifact Source SHOULD explicitly govern redirects.

A trusted host redirecting to an unauthorized destination SHALL not
automatically expand network authority.

## Credentials

Artifact-source credentials SHALL remain outside portable manifest.

Credentials SHALL not be persisted into Kernel Artifact metadata.

Diagnostics SHALL redact them.

## Ingestion Resource Quotas

Gateway SHALL support defensive quotas.

Quotas MAY include:

```text
maximum transaction bytes
maximum manifest bytes
maximum blob bytes
maximum artifact count
maximum evidence count
maximum external fetches
maximum decompressed bytes
maximum staging storage
maximum concurrent transactions
maximum validation time
```

## Per-Source Quotas

Policy MAY impose different quotas by authenticated ingestion source.

Such source identity SHALL come from the management boundary, not manifest
self-assertion.

## Global Quotas

Gateway MAY enforce global resource limits to protect Runtime host.

A flood of ingestion requests SHALL not be allowed to exhaust inference
resources without admission policy.

## Inference Priority

Production deployments SHOULD be able to prioritize inference over ingestion
work.

Ingestion activity SHALL not silently consume unbounded:

- CPU
- host memory
- disk
- network
- Provider compiler resources

needed by active inference.

## Concurrency

Gateway MAY process multiple transactions concurrently.

Transactions SHALL remain isolated.

One failed transaction SHALL not corrupt another.

## Transaction Deadline

Ingestion transactions MAY have deadlines.

Timeout SHALL transition transaction to a terminal/non-committed state unless
commit has already completed atomically.

## Cancellation

Authorized caller MAY cancel an uncommitted transaction.

Cancellation SHALL:

- stop additional external fetches
- stop additional validation work where possible
- remove temporary staging according to policy
- leave committed cache unchanged

A committed transaction cannot be undone by cancellation; revocation/eviction
are separate operations.

## Cancellation Race With Commit

Commit/cancel race SHALL have a deterministic outcome.

Transaction SHALL become either:

```text
committed
```

or:

```text
cancelled
```

not an ambiguous partial state.

## Staging Cleanup

Temporary staging content SHALL be cleaned after terminal transaction according
to retention policy.

Cleanup failure SHALL be observable.

Cleanup SHALL not delete deduplicated committed blobs.

## Rejected Artifact Retention

Policy MAY retain rejected/quarantined evidence for:

- diagnostics
- incident response
- audit
- debugging

Retention SHALL have explicit limits.

## Sensitive Source Retention

Raw Kernel source MAY be sensitive intellectual property.

Retention and diagnostics SHALL respect artifact confidentiality policy.

## Audit Record

Every ingestion transaction SHOULD produce an audit record.

Audit record MAY contain:

- transaction ID
- observed source
- manifest digest
- bundle logical identity
- policy version
- structural validation result
- integrity result
- trust result
- qualification result summary
- decision
- quarantine/rejection reasons
- committed artifact digests
- timestamps/durations

## Audit Record Redaction

Audit record SHALL not contain by default:

- raw Kernel source
- raw binaries
- native handles
- secrets
- credentials
- raw signature private material
- raw inference data
- model weights
- KV cache contents

## Decision Explainability

Gateway SHOULD expose structured reasons for:

```text
accept
quarantine
reject
```

## Error Model

Structured errors SHOULD include:

```text
kernel-ingestion-transaction-invalid
kernel-ingestion-transaction-not-found
kernel-ingestion-state-invalid
kernel-ingestion-policy-invalid
kernel-ingestion-policy-denied
kernel-ingestion-source-denied
kernel-ingestion-source-unauthenticated
kernel-ingestion-quota-exceeded
kernel-ingestion-concurrency-limit
kernel-ingestion-timeout
kernel-ingestion-cancelled

kernel-ingestion-staging-failed
kernel-ingestion-staging-limit-exceeded
kernel-ingestion-staging-corrupt
kernel-ingestion-snapshot-unavailable
kernel-ingestion-toctou-detected

kernel-ingestion-manifest-invalid
kernel-ingestion-bundle-invalid
kernel-ingestion-integrity-failed
kernel-ingestion-semantic-validation-failed
kernel-ingestion-trust-denied
kernel-ingestion-trust-unresolved
kernel-ingestion-qualification-required
kernel-ingestion-evidence-invalid
kernel-ingestion-evidence-expired
kernel-ingestion-evidence-revoked
kernel-ingestion-artifact-revoked

kernel-ingestion-external-reference-denied
kernel-ingestion-external-fetch-failed
kernel-ingestion-external-fetch-timeout
kernel-ingestion-external-redirect-denied
kernel-ingestion-external-digest-mismatch

kernel-ingestion-quarantined
kernel-ingestion-manual-approval-required

kernel-ingestion-commit-failed
kernel-ingestion-commit-conflict
kernel-ingestion-cache-corrupt
kernel-ingestion-cleanup-failed

internal-kernel-ingestion-error
```

## Observability

Ingestion observability SHOULD include:

```text
ingestion-created
ingestion-receiving
ingestion-staged
ingestion-validation-started
ingestion-integrity-valid
ingestion-integrity-failed
ingestion-trust-evaluated
ingestion-evidence-evaluated
ingestion-quarantined
ingestion-rejected
ingestion-accepted
ingestion-commit-started
ingestion-committed
ingestion-cancelled
ingestion-timed-out
ingestion-cleanup-failed
```

Observations MAY include:

- transaction ID
- observed source kind
- manifest digest
- artifact digest
- policy version
- byte counts
- artifact counts
- decision reason
- duration

## Observability Redaction

Observability SHALL redact:

- raw source
- compiled binary bytes
- credentials
- secrets
- sensitive URLs
- local temporary paths
- native handles
- raw evidence tensors
- model weights
- raw prompts
- KV cache contents

## Conformance

Conformance SHALL validate:

- import does not imply acceptance
- acceptance does not imply preparation
- acceptance does not imply promotion
- staged bytes are isolated from accepted cache
- validated bytes equal committed bytes
- malformed bundle never reaches Provider preparation
- digest mismatch fails before commit
- self-declared source/publisher does not grant trust
- quarantine is invisible to normal Registry selection
- rejected transaction cannot partially populate logical artifact
- commit is atomic
- repeated import is idempotent
- cache deduplication preserves policy evaluation
- current policy is re-evaluated on import
- revoked artifact remains revoked after re-import
- arbitrary external URL cannot expand network authority
- external bytes are digest-validated
- transaction quotas are enforced
- cancellation leaves accepted cache unchanged
- commit/cancel race is deterministic
- failed candidate import leaves active Kernel unchanged
- ingestion observability is redacted

## Non-Goals

This change does not:

- define cryptographic signature algorithm
- make publisher metadata authenticated
- implement Provider compilation
- qualify numerical correctness itself
- benchmark candidates
- promote Kernel
- select Kernel
- execute Kernel
- define one artifact registry
- expose ingestion through normal Runtime generation API
- define fleet-wide artifact replication
- define distributed consensus for cache commits
- make quarantine a production execution state

## Impact

Magnetar gains a strong security/control boundary:

```text
UNTRUSTED EXTERNAL WORLD
          |
          v
Kernel Exchange Bundle
          |
          v
      staging
          |
          v
  Ingestion Gateway
          |
   +------+------+
   |      |      |
reject quarantine accept
                  |
                  v
             atomic cache
                  |
                  v
          later Runtime policy
```

This ensures that generated Kernel ecosystems can feed Magnetar continuously
without turning artifact arrival into executable authority.