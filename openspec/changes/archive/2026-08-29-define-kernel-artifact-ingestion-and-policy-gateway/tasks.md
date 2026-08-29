# Tasks

Implemented in `magnetar-runtime/src/kernel_artifact_ingestion.rs` (module
registered in `lib.rs`), reusing the existing manifest/bundle validation
pipeline in `kernel_artifact_manifest.rs` and the cache in `kernel_cache.rs`
rather than duplicating them. Conformance evidence:
`run_kernel_artifact_ingestion_conformance` (a single report covering the
bulk of the checklist below) plus the dedicated `ingestion_*` tests in
`tests.rs` for the filesystem-backed properties (TOCTOU/immutable snapshot,
failure-atomicity, concurrent-transaction isolation). All groups below are
implemented; see the module doc comment for which type/function backs each
bullet. Genuine simplifications inherent to this crate's pure-data style
(consistent with every sibling `kernel_*` contract module): no real wall
clock (deadlines/timeouts are caller-supplied ticks), no real network I/O
(external fetch is modeled as authorization + digest verification over
caller-supplied bytes), and no real OS-level concurrency (isolation is
demonstrated via independent transaction values rather than threads).

## 1. Ingestion Gateway

- [x] Define Kernel Artifact Ingestion Gateway.
- [x] Keep it outside normal inference API.
- [x] Define gateway policy ownership.
- [x] Prevent execution authority.
- [x] Add gateway boundary tests.

## 2. Ingestion Transaction

- [x] Define KernelIngestionTransactionId.
- [x] Define KernelIngestionTransaction.
- [x] Add source metadata.
- [x] Add policy version.
- [x] Add limits.
- [x] Add transaction result.
- [x] Add audit metadata.
- [x] Add lifecycle tests.

## 3. Transaction States

- [x] Add created.
- [x] Add receiving.
- [x] Add staged.
- [x] Add validating.
- [x] Add policy-evaluating.
- [x] Add quarantined.
- [x] Add accepted.
- [x] Add committing.
- [x] Add committed.
- [x] Add rejected.
- [x] Add cancelled.
- [x] Add timed-out.
- [x] Add failed.
- [x] Add cleaning.
- [x] Add closed.
- [x] Validate legal transitions.

## 4. Observed Ingestion Source

- [x] Define observed source descriptor.
- [x] Add local tooling.
- [x] Add deployment package.
- [x] Add CI.
- [x] Add external Artifact Source.
- [x] Add optimization campaign.
- [x] Add Tachyon-distributed.
- [x] Add vendor package.
- [x] Keep source vocabulary extensible.
- [x] Distinguish observed source from manifest claim.

## 5. Source Trust Separation

- [x] Ensure observed source does not imply trust by default.
- [x] Ensure manifest source claim does not become observed source.
- [x] Ensure publisher claim does not imply trust.
- [x] Add forged-source tests.

## 6. Immutable Staging

- [x] Define staging namespace.
- [x] Keep staging outside accepted Kernel Cache.
- [x] Freeze logical bytes before validation.
- [x] Prevent source mutation affecting validated snapshot.
- [x] Add immutable snapshot tests.

## 7. TOCTOU Protection

- [x] Guarantee validated bytes equal committed bytes.
- [x] Add source mutation regression test.
- [x] Prevent path-based re-open after validation where unsafe.
- [x] Detect snapshot inconsistency.
- [x] Add TOCTOU error.

## 8. Validation Pipeline

- [x] Enforce preliminary limits.
- [x] Parse manifest.
- [x] Validate schema.
- [x] Validate path/archive safety.
- [x] Canonicalize manifest.
- [x] Validate manifest identity.
- [x] Validate blob existence.
- [x] Validate blob size.
- [x] Validate blob digest.
- [x] Validate semantics.
- [x] Validate relationships.
- [x] Evaluate trust.
- [x] Validate evidence.
- [x] Apply policy.
- [x] Add validation-order tests.

## 9. Cheap Defensive Checks

- [x] Apply size limits before expensive parsing where possible.
- [x] Apply entry-count limits.
- [x] Apply nesting limits.
- [x] Apply schema compatibility before expensive downstream work.
- [x] Add DoS-resistance tests.

## 10. Side-Effect-Free Validation

- [x] Parsing does not compile.
- [x] Parsing does not prepare.
- [x] Parsing does not execute.
- [x] Parsing does not benchmark.
- [x] Validation does not promote.
- [x] Validation does not mutate active Model Instance.
- [x] Add side-effect tests.

## 11. Semantic Validation

- [x] Validate Operator binding.
- [x] Validate Operator versions.
- [x] Validate fused structure.
- [x] Validate artifact roles.
- [x] Validate source/compiled relationships.
- [x] Validate target constraints.
- [x] Validate specialization structure.
- [x] Validate dtype/layout structure.
- [x] Validate dependency graph.
- [x] Add semantic tests.

## 12. Trust Evaluation

- [x] Integrate artifact trust policy.
- [x] Support trusted.
- [x] Support untrusted.
- [x] Support unknown/unsigned.
- [x] Support denied.
- [x] Support explicit development policy.
- [x] Fail closed where production policy requires.
- [x] Add trust tests.

## 13. Qualification Evidence Validation

- [x] Validate evidence digest.
- [x] Validate profile.
- [x] Validate suite version.
- [x] Validate oracle.
- [x] Validate target compatibility.
- [x] Check expiration.
- [x] Check revocation.
- [x] Distinguish missing evidence.
- [x] Add evidence tests.

## 14. Kernel Ingestion Policy

- [x] Define KernelIngestionPolicy.
- [x] Add schema rules.
- [x] Add source rules.
- [x] Add format rules.
- [x] Add trust rules.
- [x] Add qualification rules.
- [x] Add target restrictions.
- [x] Add external-reference rules.
- [x] Add extension rules.
- [x] Add quarantine rules.
- [x] Add duplicate behavior.
- [x] Add policy validation.

## 15. Policy Versioning

- [x] Add policy version/fingerprint.
- [x] Record in transaction.
- [x] Record in audit.
- [x] Add policy-change tests.

## 16. Decisions

- [x] Define Accept.
- [x] Define Quarantine.
- [x] Define Reject.
- [x] Define structured decision reason.
- [x] Add decision tests.

## 17. Quarantine

- [x] Define quarantine namespace.
- [x] Keep separate from accepted cache.
- [x] Prevent Registry discovery.
- [x] Prevent normal preparation.
- [x] Retain validation evidence.
- [x] Add quarantine tests.

## 18. Quarantine Reasons

- [x] Add trust-unresolved.
- [x] Add signature-required.
- [x] Add qualification-missing.
- [x] Add evidence-expired.
- [x] Add manual-review-required.
- [x] Add compatibility-pending.
- [x] Add policy-specific reason.
- [x] Keep reasons extensible.

## 19. Quarantine Re-Evaluation

- [x] Re-evaluate under current policy.
- [x] Re-evaluate trust.
- [x] Re-evaluate evidence.
- [x] Support new qualification evidence.
- [x] Support authorized approval input.
- [x] Prevent automatic executable transition.
- [x] Add re-evaluation tests.

## 20. Manual Approval

- [x] Define management approval record.
- [x] Bind approval to artifact digest.
- [x] Bind approval to policy.
- [x] Keep approval outside portable manifest.
- [x] Prevent approval bypassing digest failures.
- [x] Add approval tests.

## 21. Atomic Cache Commit

- [x] Define transaction commit set.
- [x] Stage all required metadata.
- [x] Atomically publish logical artifact.
- [x] Prevent partially visible required content.
- [x] Roll back publication metadata on failure.
- [x] Add commit failure tests.

## 22. Content Deduplication

- [x] Reuse existing digest blobs.
- [x] Verify reused blob integrity according to policy.
- [x] Preserve transaction atomicity.
- [x] Add deduplication tests.

## 23. Idempotence

- [x] Detect same logical bundle.
- [x] Preserve content identity.
- [x] Allow already-present result.
- [x] Record new transaction audit separately.
- [x] Re-evaluate current policy.
- [x] Add repeated-import tests.

## 24. Existing Cache Policy

- [x] Do not infer trust from existing cache.
- [x] Do not infer qualification from blob presence.
- [x] Re-evaluate revocation.
- [x] Detect corrupt existing digest entry.
- [x] Add cache-policy tests.

## 25. Failure Atomicity

- [x] Keep accepted cache unchanged before commit.
- [x] Keep active Registry unchanged.
- [x] Keep Prepared Kernels unchanged.
- [x] Keep Model Instances unchanged.
- [x] Add all-stage failure injection tests.

## 26. Preparation Boundary

- [x] Accepted artifact may become preparation-eligible.
- [x] Do not prepare during validation.
- [x] Keep post-commit preparation logically separate.
- [x] Add boundary tests.

## 27. Promotion Boundary

- [x] Never promote from ingestion commit directly.
- [x] Reuse existing promotion contract.
- [x] Add no-auto-promotion test.

## 28. Registry Boundary

- [x] Prevent staged candidates from Registry.
- [x] Prevent quarantined candidates from Registry.
- [x] Publish only accepted candidate metadata after commit.
- [x] Add Registry visibility tests.

## 29. Revocation

- [x] Check artifact revocation during ingestion.
- [x] Preserve revocation across cache deletion/re-import.
- [x] Prevent re-import from clearing revocation.
- [x] Add revoked-reimport tests.

## 30. External Artifact Source

- [x] Define authorized resolver interface.
- [x] Restrict supported schemes/sources.
- [x] Add source-specific limits.
- [x] Keep source authority outside manifest.
- [x] Add unauthorized locator tests.

## 31. External Fetch Integrity

- [x] Stage fetched bytes.
- [x] Validate digest.
- [x] Validate size.
- [x] Reject changed remote object.
- [x] Add remote mutation tests.

## 32. Network Authority

- [x] Deny arbitrary URL fetch.
- [x] Require authorized Artifact Source.
- [x] Define redirect policy.
- [x] Prevent redirect authority expansion.
- [x] Add network-denial tests.

## 33. External Fetch Limits

- [x] Add response size limit.
- [x] Add request timeout.
- [x] Add redirect limit.
- [x] Add total transaction download limit.
- [x] Add external artifact count limit.
- [x] Add limit tests.

## 34. Credential Boundary

- [x] Keep credentials outside manifest.
- [x] Keep credentials outside Kernel Artifact metadata.
- [x] Pass only through authorized Artifact Source implementation.
- [x] Redact credentials from diagnostics.
- [x] Add credential tests.

## 35. Transaction Quotas

- [x] Limit manifest bytes.
- [x] Limit blob bytes.
- [x] Limit total transaction bytes.
- [x] Limit artifact count.
- [x] Limit evidence count.
- [x] Limit external fetches.
- [x] Limit decompressed bytes.
- [x] Limit staging storage.
- [x] Limit validation time.
- [x] Add quota tests.

## 36. Concurrent Transactions

- [x] Define max concurrent transactions.
- [x] Isolate transaction staging.
- [x] Handle same-digest concurrent imports.
- [x] Prevent cross-transaction corruption.
- [x] Add concurrency tests.

## 37. Inference Resource Protection

- [x] Allow admission throttling for ingestion.
- [x] Bound CPU use.
- [x] Bound host memory.
- [x] Bound storage.
- [x] Bound network.
- [x] Keep inference scheduling priority policy available.
- [x] Add pressure tests.

## 38. Transaction Deadline

- [x] Add optional deadline.
- [x] Stop uncommitted work after timeout.
- [x] Preserve committed cache.
- [x] Add timeout tests.

## 39. Cancellation

- [x] Add authorized cancellation.
- [x] Stop new fetches.
- [x] Stop validation where feasible.
- [x] Clean staging.
- [x] Preserve committed state.
- [x] Add cancellation tests.

## 40. Commit/Cancel Race

- [x] Define atomic commit point.
- [x] Make final state deterministic.
- [x] Prevent partial commit/cancel combination.
- [x] Add race tests.

## 41. Staging Cleanup

- [x] Remove temporary staged artifacts after terminal state.
- [x] Preserve committed/deduplicated blobs.
- [x] Observe cleanup errors.
- [x] Add cleanup tests.

## 42. Retention Policy

- [x] Define rejected artifact retention.
- [x] Define quarantine retention.
- [x] Define audit retention.
- [x] Define storage limit.
- [x] Handle confidential source.
- [x] Add retention tests.

## 43. Audit Record

- [x] Define KernelIngestionAuditRecord.
- [x] Record transaction ID.
- [x] Record observed source.
- [x] Record manifest digest.
- [x] Record logical bundle identity.
- [x] Record policy version.
- [x] Record integrity result.
- [x] Record trust result.
- [x] Record qualification summary.
- [x] Record decision.
- [x] Record reasons.
- [x] Record committed digests.
- [x] Record timing metadata.

## 44. Audit Redaction

- [x] Redact raw Kernel source.
- [x] Redact binary payload.
- [x] Redact credentials.
- [x] Redact secrets.
- [x] Redact native handles.
- [x] Redact sensitive paths.
- [x] Redact raw model/user data.
- [x] Add audit redaction tests.

## 45. Errors

- [x] Add transaction errors.
- [x] Add policy errors.
- [x] Add source errors.
- [x] Add quota errors.
- [x] Add staging errors.
- [x] Add TOCTOU error.
- [x] Add validation errors.
- [x] Add trust errors.
- [x] Add qualification errors.
- [x] Add revocation errors.
- [x] Add external fetch errors.
- [x] Add quarantine errors.
- [x] Add commit errors.
- [x] Add cleanup errors.
- [x] Add internal ingestion error.

## 46. Observability

- [x] Observe transaction creation.
- [x] Observe staging.
- [x] Observe validation.
- [x] Observe integrity.
- [x] Observe trust evaluation.
- [x] Observe evidence evaluation.
- [x] Observe quarantine.
- [x] Observe rejection.
- [x] Observe acceptance.
- [x] Observe commit.
- [x] Observe cancellation.
- [x] Observe timeout.
- [x] Observe cleanup failure.
- [x] Include policy/source metadata safely.

## 47. Conformance

- [x] Prove imported != accepted.
- [x] Prove accepted != prepared.
- [x] Prove accepted != promoted.
- [x] Prove staging isolation.
- [x] Prove immutable snapshot.
- [x] Prove validated bytes == committed bytes.
- [x] Prove integrity before preparation.
- [x] Prove forged source does not grant trust.
- [x] Prove quarantine Registry isolation.
- [x] Prove atomic commit.
- [x] Prove idempotent repeated import.
- [x] Prove dedup does not bypass current policy.
- [x] Prove revocation survives re-import.
- [x] Prove no arbitrary network authority.
- [x] Prove external digest verification.
- [x] Prove quotas.
- [x] Prove cancellation atomicity.
- [x] Prove active Kernel unaffected by failed import.
- [x] Prove redaction.

## 48. Documentation

- [x] Document Ingestion Gateway.
- [x] Document transaction lifecycle.
- [x] Document validation order.
- [x] Document quarantine.
- [x] Document atomic commit.
- [x] Document idempotence.
- [x] Document TOCTOU protection.
- [x] Document external Artifact Source.
- [x] Document trust separation.
- [x] Document promotion/preparation separation.

## 49. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify ingestion cannot execute a Kernel.
- [x] Verify ingestion cannot grant trust implicitly.
- [x] Verify ingestion cannot grant qualification implicitly.
- [x] Verify ingestion cannot promote a Kernel.
- [x] Verify malformed input cannot affect active inference.
- [x] Verify cache publication is atomic.
