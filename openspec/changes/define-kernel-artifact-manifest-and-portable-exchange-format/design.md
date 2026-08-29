## Context

Magnetar's Kernel lifecycle (source -> compiled -> qualified -> benchmarked
-> cached -> selected -> promoted -> executed) is fully modeled internally,
but nothing describes how an external producer (an AI kernel generator, a
human engineer, CI, vendor tooling, an optimization service, an offline
build farm) hands a Kernel and its evidence to Magnetar. Every existing
contract assumes the artifact is already Runtime-native data.

This change adds that missing portable boundary: a versioned Kernel
Artifact Manifest and a content-addressed Kernel Exchange Bundle, so
Magnetar never has to trust a specific producer, language, registry, or
transport to accept a Kernel.

## Goals / Non-Goals

**Goals:**

- A versioned (`magnetar:kernel-manifest@1.x`), canonical, content-addressed
  manifest format with deterministic identity.
- A directory-based Kernel Exchange Bundle (`kernel.manifest.json` +
  `blobs/sha256/<digest>`) and at least one real alternate physical
  transport (tar / tar.gz), proving transport neutrality rather than just
  asserting it.
- Defensive parsing: duplicate-key rejection, nesting/size limits, and path
  safety (traversal, absolute paths, symlinks, hard links, device files)
  enforced before any bytes are trusted or extracted.
- Non-authoritative trust, provenance, recommendation, and signature
  metadata -- none of it can itself produce trusted/qualified/promoted
  status.
- Bridges into the existing `KernelSourceArtifact` / `CompiledKernelArtifact`
  / qualification / cache contracts, without fabricating state those
  contracts don't actually have evidence for.

**Non-Goals:**

- Defining one artifact registry, an OCI profile, or S3/HTTP APIs.
- Choosing a mandatory cryptographic signature algorithm.
- Implementing Kernel compilation, qualification, benchmarking, or
  promotion themselves (this is an exchange format, not a pipeline).
- Making arbitrary URLs fetchable by Runtime.

## Decisions

1. Canonical identity rides on `serde_json`'s default (non-`preserve_order`)
   `Map`, which is `BTreeMap`-backed.

   Deterministic key ordering, escaping, and integer representation all
   fall out of using `serde_json::to_vec` on a `Map` built without the
   `preserve_order` feature -- no hand-rolled canonicalizer needed.

   Alternative considered: write a custom canonical JSON serializer. Would
   duplicate what `serde_json` already guarantees deterministically, for no
   real benefit.

2. Duplicate-key detection needs a hand-written scanner over raw text,
   because `serde_json::Map` insertion is last-write-wins.

   `scan_manifest_structure` is a small recursive-descent JSON grammar
   walker that rejects duplicate object keys and enforces nesting depth
   *before* any `serde_json::Value` tree is built. Strict JSON grammar
   (no bare `NaN`/`Infinity`) falls out of the same walker.

   Alternative considered: parse twice (once with a duplicate-detecting
   deserializer, once normally). Rejected: no such detector ships in
   `serde_json` without `preserve_order`, and writing one is no simpler
   than the direct text scan.

3. An archive checksum is deliberately a different concept from manifest
   identity.

   `archive_diagnostic_checksum` hashes raw (possibly compressed) archive
   bytes purely as an optional transport-level diagnostic.
   `KernelManifestV1::digest` is the only identity that matters, and is
   computed after full decompression/extraction, so compression or
   archive metadata (ownership, timestamps, mode bits) never affects it --
   proven directly by tests that build two byte-for-byte-different archives
   around identical logical content and assert identical manifest digests.

4. Path/entry-type safety uses the archive format's own explicit structure
   rather than trying to infer it from `std::fs` metadata.

   For the directory transport, `std::fs::symlink_metadata` reliably
   detects symlinks but has no portable way to detect hard links or device
   files across Windows/macOS/Linux. For the tar transport, the entry-type
   byte makes every one of these explicit (`Symlink`, `Link`, `Char`,
   `Block`, `Fifo`), so `reject_unsafe_tar_entry_type` checks it directly,
   before a single byte is extracted.

   Alternative considered: shell out to `tar`/`unzip` and inspect the
   result on disk. Rejected: loses the "check before extracting" property
   this needs for defense in depth, and adds a process-execution
   dependency this crate otherwise avoids.

5. Trust and compatibility evaluation are pipeline stages, not automatic
   side effects of validation.

   `validate_kernel_exchange_bundle` never calls `evaluate_manifest_trust`
   or `evaluate_target_compatibility` itself, because both genuinely need
   Runtime-side context (policy approval; observed Provider/architecture/
   Device-feature state) that a pure manifest/bundle validation function
   does not have. Both exist as separate, directly callable, directly
   testable functions instead.

   Alternative considered: thread a `TrustPolicy`/`RuntimeContext` parameter
   through `validate_kernel_exchange_bundle`. Rejected for v1: conflates
   "is this bundle well-formed" with "is it currently trusted/compatible",
   two questions with different lifetimes (structural validation is cacheable
   forever; trust/compatibility are not).

6. Normalization into existing Runtime-native contracts stays honest about
   what the portable schema does not carry.

   `normalize_to_source_artifact` / `normalize_to_compiled_artifact` map
   cleanly onto existing rich types. `normalize_qualification_profile` /
   `normalize_oracle_identity` extract identity data only -- never a
   fabricated `QualificationRecord` status, since only real verification
   may call `start_qualifying`/`mark_qualified`. `normalize_benchmark_profile`
   required adding `KernelBenchmarkWorkloadMetadata` to the evidence schema
   first, rather than inventing plausible-looking `BenchmarkProfile` fields
   that were never actually declared by the producer.

   Alternative considered: skip benchmark normalization until a future
   change extends the schema. Rejected once the actual field gap was small
   (workload shape, warmup/measurement counts, sync policy) and could be
   added honestly in this pass.

7. Object-store, OCI-like, and registry transports are named, not built.

   `KernelBundleTransport` enumerates all five conceptual transports but
   `is_implemented()` is only `true` for `Directory` and `TarArchive`. This
   matches "Reserve X transport" in the task list (naming the concept) as
   distinct from "Support X transport" (building it) -- and matches the
   proposal's own Non-Goals, which explicitly exclude defining a registry,
   an OCI profile, or S3/HTTP APIs from this change.

## Risks / Trade-offs

- New dependencies (`tar`, `flate2`) -> Mitigation: `default-features =
  false`, `flate2` pinned to the pure-Rust `miniz_oxide` backend (no system
  zlib/C toolchain dependency), both gated `cfg(not(target_arch =
  "wasm32"))` so the wasm32 target build is unaffected; verified with
  `cargo check --target wasm32-unknown-unknown`.
- Decompression bombs -> Mitigation: per-entry and total decompressed-byte
  limits enforced through a bounded reader (`Read::take(limit + 1)`), so an
  oversized entry is detected without ever being fully materialized.
- Fused-binding fidelity loss on normalization -> `CompiledKernelArtifact`
  models a single `operator_semantics: OperatorId`, so
  `normalize_to_compiled_artifact` keeps only the primary Operator for a
  fused binding. The full fused sequence remains available from the
  portable `KernelSemanticBinding` itself; documented as a known limitation
  rather than silently dropped.
- CLI/tooling surface is reserved at the library level only
  (`KernelManifestCliOperation`), not wired into `magnetar-cli`'s actual
  argv parsing -> that is a distinct crate/UX change with no corresponding
  spec delta here.

## Migration Plan

1. Add `kernel_artifact_manifest.rs` with the manifest/bundle/error/
   observability/conformance types and the JSON parsing pipeline.
2. Wire the module into `lib.rs`; add unit tests and the conformance report
   in `tests.rs`.
3. Add the tar/tar.gz archive transport, gated off wasm32, plus the
   normalization/cache-key/CLI-operation bridges.
4. Keep existing `kernel_artifact.rs`, `kernel_qualification.rs`,
   `kernel_cache.rs` contracts unchanged; this change only adds bridges
   *into* them, never modifies their own semantics.
5. Future changes can add: object-store/OCI/registry transports (if ever
   needed, as their own dedicated changes per the Non-Goals boundary), real
   `magnetar-cli` subcommands, and a `BenchmarkProfile` schema extension if
   richer workload granularity turns out to be needed beyond what
   `KernelBenchmarkWorkloadMetadata` already carries.
