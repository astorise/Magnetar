## Context

Investigated directly against the code before designing anything, per this
session's standing practice:

- `ModelDigest` (`model.rs`) already exists with `sha256(bytes) -> Self`,
  `parse(value) -> Result<Self, _>` (`"sha256:<64 hex>"` only), and
  `verify_bytes(&self, bytes: &[u8]) -> Result<(), ModelArtifactError>`.
  Digest fields already exist at whole-artifact (`ModelArtifactId.digest`),
  part (`ModelArtifactPart.digest`), and shard (`ModelShard.digest`,
  with its own `verify_bytes`) granularity. **`ModelTensorMetadata` has no
  digest field at all** -- this is the gap.
- `model_format_roadmap.rs` (`SafetensorsTensorEntry`/`GgufTensorEntry`)
  is confirmed non-production: its own header doc states it "does not
  implement byte-level safetensors/GGUF/SentencePiece parsers... Instead
  it defines... the roadmap contract." No parser anywhere in this crate
  reads a real Safetensors/GGUF file into `ModelTensorMetadata` today.
- The one real artifact source that exists is the E2E fixture:
  `E2E_FIXTURE_SAFETENSORS_BYTES` (`first_native_runtime.rs:886`) is a
  real checked-in `.safetensors` file
  (`magnetar-runtime/fixtures/e2e-fixture-weights.safetensors`).
  `e2e_fixture_weight_inventory` builds `ModelTensorMetadata` entries with
  real, provably-correct `offset_bytes`/`size_bytes` into that file (by
  construction, not by parsing -- documented in its own comment).
  `e2e_fixture_weights_from_real_artifact` then slices real bytes out of
  the file per those offsets via `host_tensors_from_artifact_bytes`
  (`model_loading.rs`), which is itself artifact-source-agnostic (takes
  `&[ModelTensorMetadata]` + raw bytes + a data-section start offset, not
  fixture-specific).
- `bind_qwen_fixture_weights` already does *a* digest check today, but it
  is a single aggregate hash (`e2e_fixture_weight_digest`) computed over
  the **in-memory** `fixture.weights` map, compared against one hardcoded
  constant (`E2E_FIXTURE_WEIGHT_DIGEST`) -- then it **discards that map**
  and materializes `e2e_fixture_weights_from_real_artifact`'s separately-
  parsed result instead, relying on a parity *test* to prove the two are
  bit-identical rather than a structural guarantee. This is fixture-
  specific (its own comment says so) and is not what
  `materialize_model_instance_weights` itself checks -- that function
  performs no content verification of any kind on its `weights` parameter.
- `ModelLoadingCoordinator::load()` takes no raw artifact bytes at all --
  only `&ModelManifest` (metadata) and a pre-computed `&ModelTrustDecision`
  (which itself, via `ModelTrustStore::evaluate`, only string-compares the
  manifest's *self-declared* whole-artifact digest against an
  operator-pinned trusted-digest set -- it never recomputes a digest from
  real bytes). This whole-artifact trust question ("is this digest one an
  operator has pinned as trusted") is explicitly out of this Change's
  scope: it is the pre-existing, already-accepted v0.1 trust mechanism
  (`CHANGELOG.md`: "Digest pinning and explicit local development policy
  remain the accepted non-signature trust mechanisms"). This Change's
  concern is narrower and different: *given* a trusted manifest declaring
  per-tensor content, do the bytes a caller stages for a specific tensor
  name actually match what that manifest declares for it.

## Goals / Non-Goals

**Goals:**
- A manifest that declares a per-tensor content digest makes
  `materialize_model_instance_weights` reject any staged tensor whose
  actual bytes do not match that digest, before admission/write succeed.
- The check is generic (any artifact source that populates the new field),
  not fixture-specific, so it needs no rework when a real Safetensors/GGUF
  parser eventually lands.
- The E2E fixture's real, checked-in Safetensors file is the one manifest
  populated with real digests for this Change's own tests and to prove
  the happy path stays green.
- No signature changes to `materialize_model_instance_weights`,
  `create_model_instance`, or `ModelInstanceDefinition::from_loaded_context`.

**Non-Goals:**
- Whole-artifact trust establishment (operator digest pinning, revocation,
  publisher signatures) -- pre-existing, unrelated, unchanged by this
  Change.
- A real Safetensors/GGUF parser -- `model_format_roadmap.rs` stays
  roadmap-only; this Change only makes the *consuming* side (the
  materialization transaction) digest-aware, so a future parser has
  somewhere real to plug into.
- Per-tensor digests as *mandatory* on every manifest. Empty/absent means
  "unknown," matching `required_weight_names`'s existing precedent
  (round 5): a manifest that declares no digest for a tensor does not
  block materialization of that tensor, it just does not gain this
  Change's protection for it. Making digests mandatory everywhere would
  break every existing fixture/test manifest that predates this Change
  for no proportionate benefit, since the manifests that most need this
  protection (the real fixture, and future real parsers) are exactly the
  ones that will populate it.

## Decisions

**Digest granularity is per-tensor, not per-artifact or per-part.**
Whole-artifact/part/shard digests already exist and answer "is this file
the one an operator trusted" -- a different, coarser question than "does
*this specific tensor's* content match what the manifest declares for
it," which is what `materialize_model_instance_weights` needs to check
against a caller-supplied, per-tensor `HostTensor` map. Reusing
`ModelDigest`'s existing type (not inventing a new one) keeps this
consistent with the three other digest fields already on manifest types.

**The declared digest is computed from the *parsed* `HostTensor`
representation (canonical little-endian `f32` byte concatenation, the
same representation `e2e_fixture_weight_digest`'s existing per-tensor
hashing already uses), not from a raw byte slice of the artifact file.**
Considered hashing the raw file bytes directly (available for the
fixture, via `offset_bytes`/`size_bytes`): rejected because it is
*narrower* than what a real future parser could produce -- a GGUF
dequantization step, for instance, transforms stored bytes into `f32`
`HostTensor` data with no simple byte-for-byte relationship to the file's
raw region. Digesting the final `HostTensor` content is the artifact-
source-agnostic choice: "did the bytes actually staged match declared
content," decoupled from any one container format's on-disk layout. For
the fixture specifically, this digest is computed once from
`e2e_fixture_weights_from_real_artifact`'s already-real, already-parsed
`HostTensor`s (not from `e2e_fixture_weights`'s synthetic in-memory
values), so it is a real, non-circular check -- not the fixture hashing
its own assumptions about itself.

**Verification happens in `WeightMaterializationTransaction::stage_weight`,
per tensor, before Memory Manager admission** -- not as a separate
pre-flight pass over the whole `weights` map, and not deferred to
`commit`. This mirrors the existing admission-before-write ordering
principle in the same function (`transactional-weight-materialization`'s
"Memory Admission Precedes Provider Materialization") and gives the
earliest possible rejection point, consistent with how a memory-admission
failure already aborts the whole attempt via the existing
`transaction.abort(runtime)` path in `materialize_model_instance_weights`'s
caller loop -- a content mismatch becomes the same class of staging
failure, not a new special case.

**A content mismatch is reported through a new, specific
`InferenceApiError` variant** (not reused/conflated with
`MemoryAdmissionFailed`), so a caller can distinguish "the Provider/
Memory Manager rejected this" from "the bytes are wrong for this
tensor" -- these have different remediation (retry vs. re-source the
weights) and different security posture (the second is potentially an
active-forgery signal worth its own observability, not routine resource
pressure).

## Risks / Trade-offs

- [Risk] Digest verification adds a `sha256` pass over every tensor's
  bytes on every materialization call, including the hot fixture/test
  path. → Accepted: weight materialization is a one-time, not per-inference-
  step, operation (already true of the existing Memory Manager admission
  and Provider write it sits alongside); the added cost is proportional to
  tensor size once per load, not per generation step.
- [Risk] "Empty digest means unknown" (a Non-Goal decision) means a
  manifest that never populates per-tensor digests gets no protection
  from this Change at all -- a caller could still forge content for such
  a manifest exactly as before. → Accepted for the same reason
  `required_weight_names`'s identical fallback was accepted in round 5:
  the alternative (mandatory digests) breaks every existing fixture/test
  manifest for a protection those manifests do not need (they are not
  real, sourced-from-an-external-embedder Model Artifacts); real artifact
  sources are exactly the ones that will populate this field once they
  exist.
- [Risk] `ModelDigest::sha256` computed over canonical `f32` bytes ties
  this check to `HostTensor`'s specific `f32`-only representation; a
  future non-`f32` `HostTensor` variant (a quantized or bf16 native type)
  would need its own canonical byte representation defined at that point.
  → Accepted as a known, bounded extension point, not a blocker: `HostTensor`
  itself is already `f32`-only end to end today (`reference_cpu.rs`), so
  this Change does not narrow anything that is not already the case.

## Migration Plan

Single-PR change, additive schema field (no persisted format to migrate --
manifests are constructed fresh at load time, not stored). Implementation
order, each step buildable/testable before the next:

1. `ModelTensorMetadata` gains `pub digest: Option<ModelDigest>`; its YAML
   deserialization counterpart struct gains the matching optional field
   (`#[serde(default)]` so existing manifest YAML without a `digest:` key
   per tensor keeps parsing unchanged) and the conversion path threads it
   through. `cargo build` clean; existing manifest-parsing tests
   unaffected (new field defaults to `None`).
2. A canonical `HostTensor` content-digest helper (shared by fixture
   inventory construction and transaction verification) is added --
   likely in `reference_cpu.rs` alongside `HostTensor` itself, or
   `first_native_runtime.rs` near the existing `e2e_fixture_weight_digest`
   it parallels; exact placement decided at implementation time based on
   what avoids a new cross-module dependency.
3. `e2e_fixture_weight_inventory`'s digest-populated variant (or an
   in-place change, whichever proves simpler once the real signature
   constraints are visible) computes real digests from
   `e2e_fixture_weights_from_real_artifact`'s parsed `HostTensor`s;
   `e2e_fixture_manifest`'s YAML emits them per tensor.
4. `LoadedModelContext`/`ModelInstanceDefinition` gain
   `required_weight_digests`, populated in `ModelLoadingCoordinator::load()`
   from `manifest.tensors`, threaded through `from_loaded_context` (no
   signature changes, mirroring `required_weight_names`'s round-5
   precedent exactly).
5. `WeightMaterializationTransaction::stage_weight` verifies against the
   instance's `required_weight_digests` when a digest exists for the
   staged tensor name; new `InferenceApiError` variant for a mismatch.
6. Tests: content mismatch rejected (tampered fixture bytes, digest
   present); matching content accepted (real fixture bytes, digest
   present); no-digest-declared tensor still materializes (permissive
   fallback, proving this Change does not regress any manifest that
   predates it); live `qwen-test` stays green (real fixture materializes
   real bytes matching its own real digests).
7. Full verification suite; push; CI on the exact final HEAD.

Rollback: revert the commit(s); no persisted state to migrate back.

## Open Questions

- Exact shape of the YAML deserialization struct change (which
  intermediate `#[derive(Deserialize)]` struct in `model.rs` corresponds
  to a manifest's `tensors:` entries, and whether `digest` there is a raw
  `Option<String>` converted via `ModelDigest::parse` or something
  `ModelDigest` itself can derive `Deserialize` for directly) -- deferred
  to implementation; the investigation already done confirms serde-based
  parsing exists, just not which exact intermediate type needs the new
  field.
- Exact placement of the canonical `HostTensor` content-digest helper
  (`reference_cpu.rs` vs. `first_native_runtime.rs`) -- deferred to
  implementation, decided by whichever avoids introducing a new
  cross-module dependency neither file currently has.
