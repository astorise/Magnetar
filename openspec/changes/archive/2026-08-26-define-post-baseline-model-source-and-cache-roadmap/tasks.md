# Tasks

## 1. Roadmap Scope

- [x] Define post-baseline model source and cache roadmap.
- [x] Document source versus cache.
- [x] Document Runtime boundary.
- [x] Document CLI/source manager boundary.
- [x] Document trust and integrity boundary.
- [x] Document non-goals.

## 2. Source Kinds

- [x] Define development-fixture source.
- [x] Define client-provided-artifact source.
- [x] Define local-cache source.
- [x] Define local-directory-source.
- [x] Define external-registry-source.
- [x] Define model-hub-source.
- [x] Define tachyon-provided-source.
- [x] Ensure source kind does not imply trust.
- [x] Add source kind tests.

## 3. Development Fixture Source

- [x] Provide deterministic fixture artifacts.
- [x] Require normal artifact validation.
- [x] Require explicit test trust policy.
- [x] Prevent fixture trust in production unless policy allows.
- [x] Add fixture source tests.

## 4. Client-Provided Artifact Source

- [x] Define explicit caller-provided artifact references.
- [x] Support local file reference via authorized contract.
- [x] Support in-memory data placeholder where applicable.
- [x] Validate artifact before loading.
- [x] Prevent ambient filesystem authority.
- [x] Add client-provided source tests.

## 5. Local Cache Source

- [x] Define local cache lookup.
- [x] Address entries by digest or normalized identity.
- [x] Return controlled artifact references.
- [x] Prevent cache hit from bypassing trust.
- [x] Prevent cache hit from bypassing integrity.
- [x] Add local cache tests.

## 6. Local Directory Source

- [x] Define explicit local directory source.
- [x] Require user selection or policy authorization.
- [x] Prevent arbitrary Runtime directory scanning.
- [x] Normalize directory into artifact candidate.
- [x] Validate before loading.
- [x] Add local directory tests.

## 7. External Registry Source

- [x] Define registry source placeholder.
- [x] Require explicit policy.
- [x] Prevent arbitrary registry access during inference.
- [x] Keep fetch outside core inference path.
- [x] Normalize fetched artifacts before loading.
- [x] Add registry source placeholder tests.

## 8. Model Hub Source

- [x] Define model hub source placeholder.
- [x] Keep model hub integration outside core inference path.
- [x] Normalize hub metadata into Model Artifact metadata.
- [x] Keep authentication outside Runtime inference.
- [x] Add model hub placeholder tests.

## 9. Tachyon-Provided Source

- [x] Define Tachyon-provided source placeholder.
- [x] Preserve Tachyon orchestration ownership.
- [x] Preserve Magnetar artifact validation.
- [x] Prevent Tachyon bypass of Model Loading.
- [x] Add Tachyon source boundary tests.

## 10. ModelRef Resolution

- [x] Define ModelRef resolution flow.
- [x] Resolve to existing Model Instance where applicable.
- [x] Resolve to cached Model Artifact.
- [x] Resolve to client-provided artifact.
- [x] Resolve to local directory source.
- [x] Resolve to development fixture.
- [x] Reserve future registry entry.
- [x] Reserve future hub entry.
- [x] Reserve future Tachyon source.
- [x] Reject ambiguous ModelRefs.
- [x] Add ModelRef tests.

## 11. Model Aliases

- [x] Keep user-facing aliases in CLI where possible.
- [x] Define Runtime alias policy if needed.
- [x] Ensure aliases do not bypass validation.
- [x] Resolve aliases to ModelRef or source candidate.
- [x] Reject missing aliases.
- [x] Reject ambiguous aliases.
- [x] Add alias tests.

## 12. Artifact Identity

- [x] Define content digest identity.
- [x] Define manifest digest.
- [x] Define part digests.
- [x] Define shard digests.
- [x] Define tokenizer digest.
- [x] Define adapter digest.
- [x] Define config digest.
- [x] Define normalized manifest digest.
- [x] Preserve source annotations.
- [x] Preserve version metadata.
- [x] Prevent human name as authoritative identity.
- [x] Add identity tests.

## 13. Cache Addressing

- [x] Address entries by digest or normalized artifact identity.
- [x] Define implementation-specific cache path layout.
- [x] Prevent raw cache path exposure by default.
- [x] Return stable metadata and controlled references.
- [x] Add cache addressing tests.

## 14. Cache Entry Metadata

- [x] Store artifact identity.
- [x] Store normalized manifest reference.
- [x] Store source kind.
- [x] Store source annotations.
- [x] Store trust status.
- [x] Store integrity status.
- [x] Store format metadata.
- [x] Store size estimate.
- [x] Store part list.
- [x] Store shard list.
- [x] Store tokenizer references.
- [x] Store adapter references.
- [x] Store last used timestamp.
- [x] Store created timestamp.
- [x] Store eviction eligibility.
- [x] Store validation status.
- [x] Add metadata tests.

## 15. Cache Trust Model

- [x] Prevent cache presence from implying trust.
- [x] Re-check trust before loading.
- [x] Cache trust metadata only as metadata.
- [x] Apply current policy to cached trust.
- [x] Support revocation invalidating cached trust.
- [x] Add cache trust tests.

## 16. Cache Integrity

- [x] Validate content digest.
- [x] Validate shard digest.
- [x] Validate manifest consistency.
- [x] Validate file size consistency.
- [x] Validate normalized manifest consistency.
- [x] Validate tokenizer/artifact compatibility.
- [x] Validate adapter/base model compatibility.
- [x] Detect corruption.
- [x] Prevent corrupt entry loading.
- [x] Add integrity tests.

## 17. Cache Mutation

- [x] Define insert mutation.
- [x] Define update metadata mutation.
- [x] Define mark validated mutation.
- [x] Define mark untrusted mutation.
- [x] Define mark revoked mutation.
- [x] Define evict mutation.
- [x] Define prune mutation.
- [x] Define pin mutation.
- [x] Define unpin mutation.
- [x] Define repair placeholder.
- [x] Require policy for mutation.
- [x] Add mutation tests.

## 18. Cache Eviction

- [x] Define size-based eviction input.
- [x] Define age-based eviction input.
- [x] Define last-used eviction input.
- [x] Define pin protection.
- [x] Define trust-state input.
- [x] Define source-kind input.
- [x] Define validation-state input.
- [x] Define artifact-type input.
- [x] Protect active Model Instance references.
- [x] Add eviction tests.

## 19. Cache Pinning

- [x] Define pin operation.
- [x] Define unpin operation.
- [x] Protect pinned entries from automatic eviction.
- [x] Ensure pinning does not bypass validation.
- [x] Add pinning tests.

## 20. Partial Cache Entries

- [x] Define partial entry lifecycle.
- [x] Prevent partial entries from Model Loading unless explicitly supported.
- [x] Track download/import state.
- [x] Add partial entry tests.

## 21. Cache Lifecycle

- [x] Define discovered state.
- [x] Define resolving state.
- [x] Define fetching state.
- [x] Define partial state.
- [x] Define normalizing state.
- [x] Define validating state.
- [x] Define ready state.
- [x] Define untrusted state.
- [x] Define revoked state.
- [x] Define corrupt state.
- [x] Define evicting state.
- [x] Define evicted state.
- [x] Define failed state.
- [x] Add lifecycle tests.

## 22. Offline Mode

- [x] Define offline mode.
- [x] Use local cache in offline mode.
- [x] Use client-provided artifacts in offline mode.
- [x] Use development fixtures in offline mode where policy allows.
- [x] Prevent network access in offline mode.
- [x] Return structured offline errors.
- [x] Add offline tests.

## 23. Authentication Boundary

- [x] Keep remote authentication outside core Runtime inference.
- [x] Allow CLI/source manager credential handling.
- [x] Prevent credentials in Runtime cache metadata by default.
- [x] Prevent credentials in observability.
- [x] Add authentication boundary tests.

## 24. Source Policy

- [x] Define allowed source kinds.
- [x] Restrict remote sources where policy requires.
- [x] Restrict local directory sources where policy requires.
- [x] Restrict unsigned artifacts where policy requires.
- [x] Restrict untrusted cache entries where policy requires.
- [x] Restrict large artifacts where policy requires.
- [x] Restrict quantized artifacts where policy requires.
- [x] Restrict license-restricted artifacts where policy requires.
- [x] Restrict development fixtures in production.
- [x] Restrict Tachyon-provided sources where policy requires.
- [x] Add source policy tests.

## 25. License And Provenance

- [x] Preserve license metadata.
- [x] Preserve provenance metadata.
- [x] Apply license policy.
- [x] Apply provenance policy.
- [x] Avoid treating license metadata as verified unless policy validates it.
- [x] Add license/provenance tests.

## 26. Model Format Compatibility

- [x] Integrate with format normalization.
- [x] Cache raw source files where policy allows.
- [x] Cache normalized manifests.
- [x] Cache derived metadata.
- [x] Ensure Model Loading consumes normalized artifacts.
- [x] Prevent loading from arbitrary raw directories.
- [x] Add format/cache integration tests.

## 27. Adapter And Tokenizer Cache

- [x] Support Model Artifact cache entries.
- [x] Support Tokenizer Artifact cache entries.
- [x] Support Adapter Artifact cache entries.
- [x] Preserve adapter/base model compatibility metadata.
- [x] Preserve tokenizer/model compatibility metadata.
- [x] Add adapter/tokenizer cache tests.

## 28. Memory Manager Boundary

- [x] Document cache bytes versus memory residency.
- [x] Ensure cache presence does not imply loaded memory.
- [x] Ensure Model Loading controls memory materialization.
- [x] Ensure Memory Manager owns loaded resources.
- [x] Add memory/cache boundary tests.

## 29. Diagnostics

- [x] Report source kind.
- [x] Report cache hit/miss.
- [x] Report artifact digest prefix.
- [x] Report validation status.
- [x] Report trust status.
- [x] Report integrity status.
- [x] Report size estimate.
- [x] Report missing parts.
- [x] Report revoked status.
- [x] Report policy denial reason.
- [x] Redact credentials.
- [x] Redact raw file contents.
- [x] Redact raw weights.
- [x] Redact raw cache paths by default.
- [x] Add diagnostics tests.

## 30. Error Model

- [x] Define model-source-unsupported error.
- [x] Define model-source-invalid error.
- [x] Define model-source-ambiguous error.
- [x] Define model-source-policy-denied error.
- [x] Define model-source-network-denied error.
- [x] Define model-source-authentication-failed error.
- [x] Define model-source-not-found error.
- [x] Define model-source-offline-unavailable error.
- [x] Define model-cache-unavailable error.
- [x] Define model-cache-miss error.
- [x] Define model-cache-entry-invalid error.
- [x] Define model-cache-entry-corrupt error.
- [x] Define model-cache-entry-untrusted error.
- [x] Define model-cache-entry-revoked error.
- [x] Define model-cache-integrity-failed error.
- [x] Define model-cache-insert-denied error.
- [x] Define model-cache-eviction-denied error.
- [x] Define model-cache-active-reference error.
- [x] Define model-cache-partial-entry error.
- [x] Define model-cache-path-redacted status.
- [x] Define model-alias-not-found error.
- [x] Define model-alias-ambiguous error.
- [x] Define internal-model-source-cache error.

## 31. Observability

- [x] Emit model source resolved observation.
- [x] Emit model source rejected observation.
- [x] Emit model source ambiguous observation.
- [x] Emit cache lookup started observation.
- [x] Emit cache hit observation.
- [x] Emit cache miss observation.
- [x] Emit cache entry validating observation.
- [x] Emit cache entry ready observation.
- [x] Emit cache entry corrupt observation.
- [x] Emit cache entry untrusted observation.
- [x] Emit cache entry revoked observation.
- [x] Emit cache entry evicted observation.
- [x] Emit cache insertion started observation.
- [x] Emit cache insertion completed observation.
- [x] Emit cache pruning started observation.
- [x] Emit cache pruning completed observation.
- [x] Emit offline mode active observation.
- [x] Emit source policy denied observation.
- [x] Verify default redaction.

## 32. Tests

- [x] Test source kind does not imply trust.
- [x] Test fixture source validation.
- [x] Test client-provided artifact validation.
- [x] Test cache hit still validates trust.
- [x] Test cache hit still validates integrity.
- [x] Test local directory source explicitness.
- [x] Test arbitrary Runtime directory scan denied.
- [x] Test registry source placeholder denied unless policy allows.
- [x] Test model hub source placeholder denied unless policy allows.
- [x] Test Tachyon source cannot bypass loading.
- [x] Test ambiguous ModelRef rejection.
- [x] Test alias missing and ambiguous errors.
- [x] Test digest-based identity.
- [x] Test corrupt cache entry not loaded.
- [x] Test partial cache entry not loaded.
- [x] Test pinned cache entry not auto-evicted.
- [x] Test active Model Instance protects cache entry.
- [x] Test offline mode network denial.
- [x] Test credentials redacted.
- [x] Test raw cache path redacted by default.

## 33. Documentation

- [x] Document model source/cache roadmap.
- [x] Document source kinds.
- [x] Document source resolution.
- [x] Document artifact identity.
- [x] Document cache addressing.
- [x] Document cache metadata.
- [x] Document trust model.
- [x] Document integrity validation.
- [x] Document cache mutation.
- [x] Document eviction.
- [x] Document pinning.
- [x] Document partial entries.
- [x] Document offline mode.
- [x] Document authentication boundary.
- [x] Document source policy.
- [x] Document license/provenance.
- [x] Document format compatibility.
- [x] Document adapter/tokenizer cache.
- [x] Document Memory Manager boundary.
- [x] Document diagnostics.
- [x] Document non-goals.

## 34. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify source kind does not imply trust.
- [x] Verify cache hit does not bypass validation.
- [x] Verify Runtime does not gain arbitrary network authority.
- [x] Verify Runtime does not gain arbitrary filesystem scanning authority.
- [x] Verify cache presence does not imply memory residency.
- [x] Verify CLI/Runtime boundary remains intact.