# Tasks

Implementation lives in `magnetar-runtime/src/kernel_artifact_manifest.rs`,
wired into the crate via `lib.rs`, with tests in `magnetar-runtime/src/tests.rs`
(`kernel_artifact_manifest_*` / `kernel_manifest_*` / `kernel_exchange_*` /
`kernel_bundle_*`) and a conformance report
(`run_kernel_artifact_manifest_conformance`). A checked box means the
capability has real enforcing code exercised by a test (unit test or the
conformance report), not just a defined struct field.

## 1. Manifest Domain

- [x] Define KernelManifestV1.
- [x] Define schema identifier.
- [x] Define schema major/minor version.
- [x] Define manifest digest.
- [x] Define canonical representation.
- [x] Add manifest validation tests.

## 2. JSON Parsing

- [x] Require UTF-8 JSON.
- [x] Reject duplicate keys.
- [x] Enforce nesting depth.
- [x] Enforce manifest byte-size limit.
- [x] Enforce string-size limits.
- [x] Reject invalid numeric values.
- [x] Add adversarial parser tests.

## 3. Canonicalization

- [x] Define deterministic key ordering.
- [x] Define deterministic escaping.
- [x] Define deterministic integer encoding.
- [x] Remove insignificant whitespace.
- [x] Reject non-canonical unsupported numeric values.
- [x] Compute manifest digest.
- [x] Add equivalent-input canonicalization tests.

## 4. Kernel Exchange Bundle

- [x] Define logical bundle.
- [x] Require `kernel.manifest.json`.
- [x] Define `blobs/sha256/<digest>` layout.
- [x] Keep physical transport independent. (directory and tar/tar.gz both validate to the identical manifest digest for equivalent content -- `kernel_exchange_tar_archive_validates_to_identical_digest_as_directory_bundle`)
- [x] Add bundle validation tests.

## 5. Bundle Identity

- [x] Define logical bundle identity.
- [x] Keep archive checksum separate. (`archive_diagnostic_checksum` hashes raw archive bytes as an explicitly non-identity diagnostic, distinct from `KernelManifestV1::digest`)
- [x] Ensure repacking does not change logical artifact identity.
- [x] Add deterministic repack tests.

## 6. Blob Descriptor

- [x] Add digest.
- [x] Add size.
- [x] Add role.
- [x] Add media type.
- [x] Add format identity.
- [x] Add storage mode.
- [x] Add required/optional status.
- [x] Add descriptor validation.

## 7. Blob Roles

- [x] Add kernel-source.
- [x] Add compiled-kernel.
- [x] Add qualification-evidence.
- [x] Add benchmark-evidence.
- [x] Add auxiliary.
- [x] Keep role vocabulary extensible.

## 8. Blob Integrity

- [x] Validate SHA-256.
- [x] Validate byte size.
- [x] Reject missing required embedded blob.
- [x] Reject digest mismatch.
- [x] Reject size mismatch.
- [x] Add corruption tests.

## 9. Format Identity

- [x] Define namespace/name/version format identity.
- [x] Avoid closed TargetLang enum.
- [x] Support unknown format serialization.
- [x] Add known sample formats.
- [x] Add future-format round-trip tests.

## 10. Source Artifact Descriptor

- [x] Bind Kernel Source blob.
- [x] Bind source format.
- [x] Bind semantics.
- [x] Bind specialization.
- [x] Bind provenance.
- [x] Add source descriptor tests.

## 11. Compiled Artifact Descriptor

- [x] Bind compiled blob.
- [x] Bind source digest where known.
- [x] Bind compiled format.
- [x] Bind compiler metadata.
- [x] Bind target constraints.
- [x] Bind specialization.
- [x] Bind precision metadata.
- [x] Bind semantic identity.
- [x] Add compiled descriptor tests.

## 12. Semantic Binding

- [x] Define single Operator binding.
- [x] Define Operator version compatibility. (`KernelSemanticBinding::primary_version_requirements` + `is_version_compatible`, reusing `crate::KernelOperatorVersionRange`)
- [x] Define fused Operator binding.
- [x] Preserve Operator order/structure.
- [x] Add semantic fingerprint.
- [x] Reject invalid Operator references.
- [x] Add fused semantic tests.

## 13. Target Constraints

- [x] Add Device type.
- [x] Add hardware vendor.
- [x] Add architecture.
- [x] Add Device features.
- [x] Add execution environment.
- [x] Add Provider compatibility.
- [x] Add runtime/driver compatibility.
- [x] Add memory class requirements.
- [x] Prevent native handles. (structural: no pointer-shaped field exists)
- [x] Add target compatibility tests.

## 14. Specialization

- [x] Add exact dimensions.
- [x] Add dimension ranges.
- [x] Add batch range.
- [x] Add sequence range.
- [x] Add head count/dimension.
- [x] Add alignment.
- [x] Add dtype.
- [x] Add layout.
- [x] Add quantization.
- [x] Add execution phase.
- [x] Add Device features.
- [x] Add specialization validation.

## 15. Precision Metadata

- [x] Add accumulation dtype.
- [x] Add approximate-math claim.
- [x] Add determinism claim.
- [x] Add tolerance reference.
- [x] Add quantization metadata reference.
- [x] Ensure claims remain qualification-dependent. (structural: nothing in this module treats a precision claim as proof of correctness)

## 16. Compiler Metadata

- [x] Add compiler identity.
- [x] Add compiler version.
- [x] Add backend version.
- [x] Add flags fingerprint.
- [x] Add build fingerprint.
- [x] Add source digest. (via the artifact's own `source_digest`, cross-referenced during normalization)
- [x] Add target architecture.
- [x] Add compiler metadata tests.

## 17. Provenance

- [x] Add human-authored.
- [x] Add ai-generated.
- [x] Add ci-generated.
- [x] Add vendor-provided.
- [x] Add optimizer-generated.
- [x] Add compiler-generated.
- [x] Add imported.
- [x] Add generator identity/version. (`KernelGeneratorMetadata`)
- [x] Ensure provenance does not imply trust.
- [x] Add trust regression tests.

## 18. Campaign Metadata

- [x] Support Optimization Campaign ID.
- [x] Support generator version.
- [x] Support source revision metadata.
- [x] Do not require raw AI prompts. (no such field exists anywhere in the schema)
- [x] Prevent credentials in locator fields. (`looks_like_embedded_credential_locator`, a defensive heuristic, not a full URL parser)
- [x] Add campaign metadata tests.

## 19. Qualification Evidence Reference

- [x] Add evidence digest.
- [x] Add qualification profile.
- [x] Add suite version.
- [x] Add oracle identity.
- [x] Add target compatibility.
- [x] Add status.
- [x] Support embedded evidence.
- [x] Support external evidence.
- [x] Add evidence validation tests.

## 20. Benchmark Evidence Reference

- [x] Add evidence digest. (shares `KernelEvidenceReference` with qualification evidence)
- [x] Add benchmark profile.
- [x] Add workload profile. (distinct `workload_profile` field)
- [x] Add Device context.
- [x] Add Provider context.
- [x] Add freshness metadata. (`evaluate_qualification_evidence_currency`)
- [x] Add benchmark reference tests.

## 21. Recommendation Metadata

- [x] Add recommended profile metadata.
- [x] Add experimental recommendation.
- [x] Add rejection recommendation.
- [x] Keep recommendation advisory.
- [x] Prevent recommendation -> active shortcut.
- [x] Add recommendation tests.

## 22. Trust Metadata

- [x] Add publisher assertion.
- [x] Add source assertion.
- [x] Add signature envelopes.
- [x] Add optional key identifier.
- [x] Add signed scope.
- [x] Prevent manifest `trusted=true` authority.
- [x] Add trust validation tests.

## 23. Signature Envelope

- [x] Define algorithm field.
- [x] Define key ID field.
- [x] Define signed digest.
- [x] Define signature material reference.
- [x] Define optional certificate-chain reference.
- [x] Define signed scope.
- [x] Add structural signature tests.
- [x] Do not claim cryptographic verification without implementation.

## 24. Embedded Storage

- [x] Define embedded mode.
- [x] Require digest path.
- [x] Verify embedded blob presence.
- [x] Add embedded bundle tests.

## 25. External Storage

- [x] Define external mode.
- [x] Add stable artifact/source locator hints.
- [x] Keep digest authoritative.
- [x] Prevent ambient Runtime fetch.
- [x] Add denied-network tests.

## 26. Distribution Neutrality

- [x] Support local directory.
- [x] Support archive representation. (`extract_kernel_exchange_archive`: tar and tar.gz, via the `tar`/`flate2` crates, gated `#[cfg(not(target_arch = "wasm32"))]`)
- [x] Reserve object-store transport. (`KernelBundleTransport::ObjectStore` -- named, type-checked reservation, not an implementation: `is_implemented()` returns `false` for it, tested by `kernel_bundle_transport_reservation_marks_only_directory_and_tar_as_implemented`. Building the object-store/S3 API itself is out of scope per the proposal's own "Non-Goals".)
- [x] Reserve OCI-like transport. (`KernelBundleTransport::OciLike`, same reservation pattern; an OCI profile is explicitly a "Non-Goal".)
- [x] Reserve registry transport. (`KernelBundleTransport::Registry`, same reservation pattern; one artifact registry is explicitly a "Non-Goal".)
- [x] Keep core manifest transport-neutral.

## 27. Path Safety

- [x] Reject `../`.
- [x] Reject absolute paths.
- [x] Reject Windows drive escape paths.
- [x] Reject symlinks.
- [x] Reject hard-link escape. (archive transport: `reject_unsafe_tar_entry_type` rejects `tar::EntryType::Link`)
- [x] Reject device files. (archive transport: rejects `Char`/`Block`)
- [x] Reject special entries. (archive transport: rejects `Fifo` and any other non-regular/non-directory entry type)
- [x] Add traversal tests.

The directory transport still has no equivalent hard-link/device-file check
(these concepts are not portably detectable via `std::fs` metadata alone
across Windows/macOS/Linux), but the archive transport's tar entry-type byte
makes every one of these explicit and checkable before any bytes are
extracted, so the underlying capability now exists and is tested.

## 28. Archive Normalization

- [x] Ignore ownership metadata for logical identity. (`kernel_exchange_archive_ignores_ownership_timestamp_and_mode_for_identity`: differing uid/gid across two archives still validates to the identical manifest digest)
- [x] Ignore timestamps. (same test: differing mtime does not affect identity)
- [x] Ignore executable mode for semantics. (same test: differing mode bits do not affect identity)
- [x] Keep archive checksum optional/separate. (`archive_diagnostic_checksum`)
- [x] Add repack tests.

## 29. Compression

- [x] Digest logical uncompressed blob bytes. (extraction fully decompresses to disk before any digest computation runs; digests never see compression framing)
- [x] Allow transport compression. (`extract_kernel_exchange_archive(..., gzip_compressed: true, ...)` via `flate2`)
- [x] Verify decompressed size limits. (`KernelExchangeArchiveLimits::max_entry_decompressed_bytes` / `max_total_decompressed_bytes`)
- [x] Prevent decompression bomb beyond configured limits. (bounded read via `Read::take(limit + 1)`, so an oversized entry is detected and rejected without fully materializing it)
- [x] Add compressed bundle tests.

## 30. Duplicate Protection

- [x] Reject duplicate JSON keys.
- [x] Reject duplicate bundle paths. (structural: a bundle path is derived purely from digest, so a path collision is exactly the "conflicting digest metadata" case below)
- [x] Reject conflicting artifact IDs. (structural: this format has no artifact ID distinct from content digest)
- [x] Reject conflicting digest metadata. (`detect_conflicting_digest_metadata`: same digest declared with different sizes fails closed; same digest/same size is legitimate dedup)
- [x] Add duplicate tests.

## 31. Dependencies

- [x] Define content-addressed auxiliary dependency.
- [x] Validate dependency digests.
- [x] Detect dependency cycles where prohibited.
- [x] Prevent arbitrary host library lookup.
- [x] Add dependency tests.

## 32. Multiple Variants

- [x] Allow multiple source variants.
- [x] Allow multiple compiled architectures.
- [x] Allow multiple Provider targets.
- [x] Allow multiple qualification records.
- [x] Allow multiple benchmark profiles.
- [x] Add multi-target bundle fixture. (`kernel_manifest_multi_target_bundle_validates_with_distinct_architectures`)

## 33. Completeness

- [x] Mark artifact required/optional.
- [x] Reject missing required embedded blobs.
- [x] Allow optional evidence absence according to policy.
- [x] Add incomplete bundle tests.

## 34. Defensive Limits

- [x] Limit manifest bytes.
- [x] Limit total bundle bytes.
- [x] Limit artifact count.
- [x] Limit evidence count.
- [x] Limit target count. (`KernelManifestLimits::max_target_entries`)
- [x] Limit extension count.
- [x] Limit annotation size.
- [x] Limit nesting.
- [x] Add limit-exhaustion tests.

## 35. Integer Safety

- [x] Use checked size arithmetic. (`saturating_add` for total embedded bytes)
- [x] Use checked dimension arithmetic. (`KernelManifestSpecialization::validate` rejects `min > max` ranges)
- [x] Reject overflow. (saturating accumulation; proven non-panicking near `u64::MAX`)
- [x] Reject impossible offsets. (impossible batch/sequence ranges rejected)
- [x] Add overflow tests.

## 36. Annotations

- [x] Define namespaced annotation key.
- [x] Treat annotations as non-authoritative.
- [x] Enforce size/count limits.
- [x] Prevent core field override.
- [x] Add annotation tests.

## 37. Extensions

- [x] Define optional extension.
- [x] Define required extension.
- [x] Ignore unknown optional extension safely.
- [x] Reject unknown required extension.
- [x] Prevent extension overriding core security semantics.
- [x] Add extension compatibility tests.

## 38. Normalization

- [x] Normalize manifest to KernelSourceArtifact. (`normalize_to_source_artifact`)
- [x] Normalize compiled descriptors. (`normalize_to_compiled_artifact`; known limitation -- fused bindings keep only the primary Operator in `CompiledKernelArtifact.operator_semantics`, since that type models a single `OperatorId`)
- [x] Normalize qualification references. (`normalize_qualification_profile` / `normalize_oracle_identity` -- identity data only, never a fabricated `QualificationRecord` status)
- [x] Normalize benchmark references. (`normalize_benchmark_profile`, backed by the new `KernelBenchmarkWorkloadMetadata` evidence field; `hardware_architecture` is an explicit caller-supplied parameter since the evidence reference itself does not carry artifact linkage)
- [x] Keep portable types separate from Provider native types.
- [x] Add normalization tests.

## 39. No Execution During Parsing

- [x] Parsing does not compile.
- [x] Parsing does not prepare.
- [x] Parsing does not execute.
- [x] Parsing does not benchmark.
- [x] Parsing does not promote.
- [x] Add side-effect-free parsing tests.

## 40. Validation Pipeline

- [x] Parse.
- [x] Validate structure.
- [x] Validate schema.
- [x] Canonicalize.
- [x] Validate manifest identity.
- [x] Validate blob integrity.
- [x] Validate semantic binding.
- [x] Evaluate trust. (`evaluate_manifest_trust`, a nameable pipeline-stage wrapper that still delegates unchanged to `crate::evaluate_artifact_trust` -- deliberately not called automatically by `validate_kernel_exchange_bundle`, since trust needs Runtime policy context that function does not have)
- [x] Validate evidence. (structural validation; currency revalidation is the caller's responsibility)
- [x] Evaluate compatibility. (`evaluate_target_compatibility`, run separately from `validate_kernel_exchange_bundle` since it needs Runtime-observed context)
- [x] Add ordering tests. (`kernel_manifest_validation_pipeline_orders_schema_before_blob_io`)

## 41. Kernel Cache Integration

- [x] Insert validated blobs by digest. (`normalize_to_cache_key` / `normalize_to_cache_entry`)
- [x] Preserve artifact identity.
- [x] Keep trust separate. (entry always starts `Untrusted`)
- [x] Keep qualification separate. (entry always starts unqualified)
- [x] Prevent corrupt entry insertion. (deferred to `crate::verify_cache_entry_integrity` / `crate::evaluate_cache_eligibility`, not bypassed)
- [x] Add cache integration tests.

## 42. Prepared State Exclusion

- [x] Reject PreparedKernelId as portable executable identity.
- [x] Reject Provider native pointer.
- [x] Reject Device native pointer.
- [x] Reject stream/event/context handle.
- [x] Add handle-leak tests.

## 43. Inference API Boundary

- [x] Keep bundle import outside normal generation requests.
- [x] Prevent per-request executable bundle injection.
- [x] Preserve management/loading boundary.
- [x] Add inference API tests.

## 44. CLI / Tooling

- [x] Reserve manifest inspect. (`KernelManifestCliOperation::Inspect`)
- [x] Reserve bundle validate. (`KernelManifestCliOperation::Validate`)
- [x] Reserve bundle import. (`KernelManifestCliOperation::Import`)
- [x] Reserve bundle export. (`KernelManifestCliOperation::Export`)
- [x] Use shared validation implementation. (`run_kernel_manifest_cli_operation` delegates every operation to `validate_kernel_exchange_bundle`)
- [x] Add CLI boundary tests.

Note: this reserves the operation surface and shared-validation guarantee at
the `magnetar-runtime` library level, matching how `cli_boundary.rs` already
formalizes CLI contracts in this crate. It does not add argv/subcommand
parsing to the `magnetar-cli` binary -- that is a distinct crate/UX change
with no corresponding spec delta in this change.

## 45. Error Model

- [x] Add manifest errors.
- [x] Add bundle errors.
- [x] Add blob integrity errors.
- [x] Add evidence errors.
- [x] Add extension errors.
- [x] Add external-reference errors.
- [x] Add trust/compatibility errors.
- [x] Add internal manifest error.

## 46. Observability

- [x] Observe discovery.
- [x] Observe schema version. (`KernelManifestObservation::with_schema_version`)
- [x] Observe manifest digest.
- [x] Observe artifact counts. (`KernelManifestObservation::with_artifact_count`)
- [x] Observe formats. (`KernelManifestObservation::with_formats`)
- [x] Observe integrity failures.
- [x] Observe semantic binding.
- [x] Observe evidence references.
- [x] Observe provenance summary.
- [x] Observe trust result.
- [x] Observe cache import.
- [x] Redact source/binaries/credentials/paths.

## 47. Conformance

- [x] Prove canonical identity.
- [x] Prove duplicate-key rejection.
- [x] Prove filename != format.
- [x] Prove blob digest validation.
- [x] Prove optional extension forward compatibility.
- [x] Prove required extension rejection.
- [x] Prove provenance != trust.
- [x] Prove source location != trust.
- [x] Prove recommendation != promotion.
- [x] Prove qualification evidence revalidation.
- [x] Prove external URL does not create network authority.
- [x] Prove path traversal rejection.
- [x] Prove symlink rejection.
- [x] Prove repacking identity stability.
- [x] Prove no native handles.
- [x] Prove parse has no execution side effects.
- [x] Prove malformed bundle cannot alter active Kernel.

## 48. Documentation

- [x] Document Kernel Manifest v1. (module-level rustdoc)
- [x] Document Kernel Exchange Bundle v1.
- [x] Document content addressing.
- [x] Document source/compiled descriptors.
- [x] Document semantic binding.
- [x] Document qualification/benchmark references.
- [x] Document trust semantics.
- [x] Document extension model.
- [x] Document path safety.
- [x] Provide minimal example manifest. (in the module doc comment)
- [x] Provide multi-target example manifest. (in the module doc comment, with an executable equivalent in the test suite)

Documentation lives as rustdoc in `kernel_artifact_manifest.rs` rather than
standalone Markdown files, consistent with how the rest of this crate
documents its contracts.

## 49. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify format is generator-neutral.
- [x] Verify format is Provider-neutral.
- [x] Verify format is transport-neutral. (directory and tar/tar.gz archives both validate to the identical manifest digest for equivalent content)
- [x] Verify manifest cannot grant trust.
- [x] Verify native handles cannot cross boundary.
- [x] Verify Runtime remains network-authority-free by default.
