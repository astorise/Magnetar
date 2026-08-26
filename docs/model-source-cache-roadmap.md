# Post-Baseline Model Source And Cache Roadmap

The [model format roadmap](model-format-roadmap.md) defines how external
model files normalize into Model Artifact, Tokenizer Artifact, Adapter
Artifact, tensor, quantization, and manifest metadata. This roadmap defines
where those artifacts come from and how they are cached: controlled source
kinds, digest-based cache addressing, cache entry metadata, trust and
integrity re-validation, mutation/eviction/pinning policy, offline mode, and
the CLI/Runtime boundary -- without giving Magnetar Runtime arbitrary
filesystem or network authority.

This document, and the `magnetar-runtime::model_source_cache_roadmap` module
it describes, do **not** implement real model downloads, a
`magnetar model pull` UX, a registry protocol, a model hub API, a Tachyon
distribution protocol, a cache directory layout, or production credential
storage. They define the roadmap **contract** -- source kinds, cache
semantics, identity rules, structured errors, observability categories, and
conformance checks -- that any future source/cache implementation must
satisfy.

## Source And Cache Principle

```text
ModelRef
    |
    v
Source Resolution
    |
    v
Artifact Candidate
    |
    v
Format Normalization
    |
    v
Artifact Validation
    |
    v
Trust / Integrity Validation
    |
    v
Digest-addressed Cache
    |
    v
Model Loading
```

A source provides artifact bytes and metadata; a cache stores validated
artifact bytes and metadata. Neither grants trust by itself.

## Source Kinds

`ModelSourceKind` enumerates the seven post-baseline source kinds from the
proposal (development-fixture, client-provided-artifact, local-cache,
local-directory-source, external-registry-source, model-hub-source,
tachyon-provided-source). `ModelSourceKind::grants_trust` is always `false`
-- "source kind SHALL not imply trust" is the roadmap's central invariant.

Rather than introducing a parallel source enum, `ModelSourceKind::from_artifact_source`
and `ModelSourceKind::from_resolution_source` normalize the *existing*
[`crate::model::ModelArtifactSource`] (already a closed, seven-variant enum:
local path, local cache, client-provided, registry, Hugging Face, OCI,
Tachyon) and [`crate::inference_api::ModelResolutionSource`] contracts onto
the roadmap's naming -- proving both already-Runtime-owned source contracts
cover the roadmap's seven kinds.

## Development Fixture Source

`validate_development_fixture_source` denies development-fixture sources in
production unless policy explicitly allows them (the "development fixtures
in production" restriction from "Source Policy"). Fixtures are never
special-cased for trust:
`development_fixture_requires_explicit_trust_evaluation` always re-evaluates
trust through the caller-supplied `ModelTrustStore`, the same store every
other source kind uses.

## Client-Provided Artifact Source

`validate_client_provided_source` requires an explicit `authorized`
attestation -- a client-provided reference is never assumed authorized by
default.

## Local Cache Source And Cache Addressing

`CacheKey` is a digest-based addressing key built on the existing
`ModelDigest` rather than a parallel identity type. `CacheEntryRef` pairs a
`CacheKey` with a raw on-disk path; `CacheEntryRef::redacted_path` is the
*only* accessor for that path, and it returns `None` unless the caller
explicitly attests disclosure is policy-allowed -- "public APIs SHALL not
expose raw cache paths by default" holds structurally, not by convention.

## Local Directory Source

`validate_local_directory_source` composes the existing
[`model_format_roadmap::validate_local_file_boundary`](model-format-roadmap.md)
rather than duplicating the local-file authorization boundary: Runtime never
scans arbitrary local directories, whether the caller is a format parser or
a source/cache lookup.

## External Registry / Model Hub / Tachyon-Provided Sources

`validate_remote_source_policy` is one shared policy gate for every
remote-shaped source kind, delegating to `SourcePolicy::validate_kind`.
Tachyon-provided sources require an *additional* explicit
`allow_tachyon_provided_sources` flag even when the kind is otherwise listed
in `allowed_kinds` -- Tachyon orchestration authority never implies source
authority. Tachyon-provided artifacts still flow through the same
`reject_non_ready_cache_entry_for_loading` / `evaluate_cache_trust` /
`validate_cache_integrity` checks as every other source: "Tachyon SHALL not
bypass Model Loading."

## ModelRef Resolution And Model Aliases

`resolve_model_ref_candidates` takes every candidate a `ModelRef` matched
against and fails **not-found** on zero candidates and **ambiguous** on more
than one -- resolution can never silently pick one of several matches.
`ModelAliasTable` mirrors this for user-facing aliases: `ModelAliasTable::resolve`
fails `model-alias-not-found` or `model-alias-ambiguous` and, on success,
returns only an owned reference string for the caller to resolve and
validate as a `ModelRef` -- an alias can never itself become a loaded model.

## Artifact Identity

`ArtifactIdentityCoverage::from_manifest` reports which roadmap-named
identity fields an already-validated `ModelManifest` carries (content
digest, part digests, shard digests, tokenizer reference, source annotation,
version metadata), mirroring
[`model_format_roadmap::NormalizedManifestCoverage`](model-format-roadmap.md)
-- proving the *existing* `ModelArtifactId`/`ModelManifest` contract already
carries the roadmap's identity fields. `artifacts_are_distinct_despite_same_name`
is the roadmap's explicit, testable statement that two artifacts sharing a
name but declaring different digests remain distinct identities.

## Cache Entry Metadata And Lifecycle

`CacheEntryMetadata` carries the proposal's cache entry metadata fields,
built on `ModelArtifactId`, `ModelShardId`, `TokenizerArtifactId`, and
`AdapterArtifactId` rather than parallel identity types.
`CacheLifecycleState` enumerates the thirteen lifecycle states from the
proposal (discovered, resolving, fetching, partial, normalizing, validating,
ready, untrusted, revoked, corrupt, evicting, evicted, failed).
`reject_non_ready_cache_entry_for_loading` denies every non-`Ready` state a
structured, state-specific error (`model-cache-partial-entry`,
`model-cache-entry-corrupt`, `model-cache-entry-untrusted`,
`model-cache-entry-revoked`, or `model-cache-entry-invalid`).

## Cache Trust Model And Integrity

`evaluate_cache_trust` always re-evaluates trust through `ModelTrustStore`;
an explicit `revoked` attestation always wins over any previously cached
trust status -- "cache presence SHALL not imply trust" and "revocation
checks MAY invalidate cached trust" both hold structurally.
`validate_cache_integrity` compares a declared digest against a recomputed
one; `validate_cache_shard_integrity` composes the existing
`ModelShard::verify_bytes` for shard-level integrity rather than duplicating
digest comparison. Corrupt entries never load.

## Cache Mutation, Eviction, And Pinning

`authorize_cache_mutation` requires an explicit policy-allowed flag for
every `CacheMutationKind` (insert, update metadata, mark validated/
untrusted/revoked, evict, prune, pin, unpin, repair placeholder) and denies
evict/prune outright whenever `active_instance_refs > 0`, before the policy
flag is even consulted -- "eviction SHALL not remove artifacts required by
active Model Instances" cannot be overridden by policy.

`is_evictable` combines pin status, active references, and lifecycle
in-flight state (`CacheLifecycleState::is_in_flight`); `select_eviction_candidates`
orders evictable entries oldest-`last_used`-first. `pin_entry`/`unpin_entry`
only affect `is_evictable` -- pinning never bypasses
`reject_non_ready_cache_entry_for_loading` or `evaluate_cache_trust`.

## Partial Cache Entries And Offline Mode

Partial entries are one of the thirteen lifecycle states and are rejected
for loading by `reject_non_ready_cache_entry_for_loading` like every other
non-`Ready` state. `validate_offline_source` restricts offline mode to
exactly `ModelSourceKind::available_offline`'s three kinds (development
fixture, client-provided artifact, local cache) -- every other kind fails
`model-source-offline-unavailable` while offline.

## Authentication Boundary

`reject_credential_in_metadata` rejects a cache metadata annotation map
whose keys look credential-shaped (`token`, `password`, `secret`, `apikey`,
`credential`, `bearer`, ...) -- authentication for remote sources stays
outside core Runtime cache metadata by construction.

## Source Policy

`SourcePolicy` names the allowed source kinds plus the proposal's
restriction flags (unsigned artifacts, untrusted cache entries, artifact
size, quantized artifacts, license-restricted artifacts, development
fixtures in production, Tachyon-provided sources). Its `Default` is
deny-by-default: only the three offline-available kinds are allowed.

## License And Provenance

`validate_license_policy` mirrors
[`model_format_roadmap::reject_silent_tokenizer_config_override`](model-format-roadmap.md)'s
shape: an unvalidated license is denied even when policy would otherwise
allow it -- "license metadata SHALL not be treated as verified unless
validated by policy."

## Compatibility With Model Formats, Adapter, And Tokenizer Artifacts

`cache_entry_ready_for_format_normalization` requires both declared format
metadata and a `Ready`/`Normalizing` lifecycle state before a cache entry
feeds format normalization. `validate_adapter_cache_entry` and
`validate_tokenizer_cache_entry` require an actual adapter/tokenizer
reference on the cache entry plus explicit base-model/tokenizer
compatibility -- a cache hit never bypasses compatibility validation.

## Compatibility With Memory Manager

`cache_presence_implies_memory_residency` always returns `false`.
`CacheEntryMetadata` has no field through which a `TensorResource` or
`MemoryAllocation` could be represented -- cache storage is bytes and
metadata only; residency remains owned exclusively by Model Loading and
Memory Manager.

## Diagnostics, Error Model, And Observability

`ModelSourceCacheDiagnostic` has no field through which a raw cache path,
credential, raw file content, raw model weight, or Provider/Device/Kernel
handle could be represented; `digest_prefix_from` returns a short,
non-reversible digest prefix, never the full digest, and
`redact_policy_denial_reason` reuses the existing `redact_backend_diagnostic`
path.

`ModelSourceCacheRoadmapError` covers all 22 error categories from the
proposal's "Error Model" section. `ModelSourceCacheRoadmapObservationKind`
covers all 18 categories from "Observability";
`ModelSourceCacheRoadmapObservation` carries only an observation kind, an
optional artifact identity string, and a `redacted_metadata` map whose
values always pass through `redact_backend_diagnostic`.

## Conformance Report

`run_model_source_cache_roadmap_conformance` produces a
`ModelSourceCacheRoadmapConformanceReport` (mirroring
`ModelFormatRoadmapConformanceReport`) asserting: no source kind grants
trust; local-directory sources deny unauthorized local file access; ModelRef
resolution rejects zero and multiple candidates; aliases reject
missing/ambiguous resolution; two same-named artifacts with different
digests remain distinct; a non-`Ready` cache entry is never loadable; cached
trust is always re-evaluated and revocation always wins; a digest mismatch
fails cache integrity validation; eviction/pruning is denied while a Model
Instance references the entry; a pinned entry is never evictable; offline
mode denies non-offline source kinds; credential-shaped cache metadata is
rejected; unvalidated license metadata is denied; and cache presence never
implies memory residency.

## Local Commands

Run the model source/cache roadmap tests:

```powershell
cargo test -p magnetar-runtime model_source_cache_roadmap -- --nocapture
```

Run the full Runtime suite:

```powershell
cargo test --workspace --all-targets
```

Validate the OpenSpec change:

```powershell
openspec validate define-post-baseline-model-source-and-cache-roadmap --strict
```

## Compatibility Versioning

The current roadmap contract version is `0.1.0`, exposed as
`MODEL_SOURCE_CACHE_ROADMAP_VERSION`. Passing this contract's conformance
checks does not imply any real source implementation (downloads, registry
client, model hub client, Tachyon distribution) has been built -- it only
confirms the roadmap's structural guarantees (no source kind grants trust,
cache hits never bypass trust/integrity, active Model Instances block
eviction, offline mode is source-kind-restricted, cache presence never
implies memory residency) hold in this Runtime revision.
