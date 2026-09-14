## Context

MAG-01 was the last open P0 from the Tachyon integration audit
(`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md`) after
MAG-02/MAG-03 closed (`wire-inference-component-to-generic-registry`).
Investigating before implementing (this session's established practice)
found the fix is smaller than it first appears: `ProductionModelArtifactIngestor`
(the trait both `HuggingFaceIngestor` and `GgufIngestor` already implement)
was *already* format-neutral by design -- `magnetar-runtime` itself never
imports either concrete crate. The gap was entirely in `inference-components`,
the one embedder that only ever constructed the Hugging Face one.

`ProductionIngestionResult { manifest, payload_source }` carries no
tokenizer -- neither ingestor's `.ingest()` builds one; a caller
separately constructs a tokenizer from the bundle's own files after
ingestion, for both formats alike. This meant tokenizer construction
needed its own format branch too, not just ingestor selection.

## Goals / Non-Goals

**Goals:**
- `LoadedInferenceComponent::load` accepts both Hugging Face-shaped and
  GGUF-shaped bundles, selecting the real ingestor/tokenizer construction
  path for each without hardcoding either as the only option.
- Reuse existing, already-tested logic on both sides (`GgufIngestor`,
  `GgufTokenizer::from_gguf_metadata`, `HuggingFaceTokenizer::from_bytes`,
  `HuggingFaceChatTemplateFormatter`) -- no new ingestion/tokenizer logic
  invented in `inference-components` itself.

**Non-Goals:**
- A third format. This change closes the audit's specific finding (only
  one format was ever reachable); it does not build a fully general
  format-registry/plugin mechanism for an arbitrary number of future
  formats -- if a third real ingestor exists in this codebase someday,
  extending the same `if is_gguf { .. } else { .. }` shape (or replacing
  it with a real registry, if the number of formats grows enough to
  justify one) is that future change's decision, not pre-built here
  speculatively.
- A dedicated `inference-components`-level integration test driving
  `LoadedInferenceComponent::load` end to end for either format. Already
  an open, documented gap before this change (zero such tests existed);
  this change extends the *scope* of what that future test would need to
  cover (both formats now, not one) without closing the gap itself.

## Decisions

- **Format detection is a single file-existence check** (`root.join(GGUF_FILE_NAME).is_file()`), not a content-sniffing/magic-byte check. `GgufIngestor` itself already only ever looks for one fixed file name (`model.gguf`) within a bundle root -- mirroring that exact expectation in the caller is simpler and cannot disagree with what the ingestor itself will do a moment later. `GGUF_FILE_NAME` was `const` (private); made `pub` so the caller does not duplicate the literal string.
- **Chat template formatting needed no format-specific code.** Investigated first: `loaders/gguf::load_gguf_chat_template`'s own doc comment already states its purpose is to return raw text for a caller to render "with whatever `magnetar_runtime::ChatTemplateFormatter` implementation it has available" -- a deliberate design decision from `wire-gguf-into-model-loading`, not something this change needed to build. `HuggingFaceChatTemplateFormatter::new(template: impl Into<String>)` takes raw Jinja text directly and has no Hugging-Face-specific behavior despite its name; both formats' raw chat template text flows through the identical call. Renaming the type to something more format-neutral was considered and rejected as out of scope -- it would touch `loaders/huggingface`'s own public API for a cosmetic reason unrelated to closing MAG-01.
- **A new `loaders/gguf::load_gguf_tokenizer` helper, not a raw-metadata-access API.** `GgufTokenizer::from_gguf_metadata` needs the GGUF file's raw parsed key-value metadata map, which `ProductionIngestionResult` does not carry. Rather than have `inference-components` depend on `magnetar-format-gguf` directly to re-parse the file itself, `loaders/gguf` gains one more "re-read the file, return a ready-to-use value" helper -- the exact same shape as the already-existing `load_gguf_chat_template`, kept in the crate that already owns GGUF-parsing knowledge.
- **Verified against the same real GGUF fixture bytes `GgufIngestor`'s own ingestion test already uses** (`tiny_qwen2_gguf()`), rather than inventing a new fixture -- proves the new helpers work against the identical real file shape ingestion itself already trusts.

## Risks / Trade-offs

- **A real byte-level-BPE encode/decode round-trip was not achievable in the new test** with this fixture's minimal vocabulary (no merges, no byte-level-prefixed tokens) -- encoding arbitrary text produced zero tokens, a fixture-realism limitation, not a defect in `load_gguf_tokenizer` itself (which only threads real metadata through to the already-independently-tested `GgufTokenizer::from_gguf_metadata`). The test instead asserts the tokenizer's real `vocabulary_size` matches the fixture's real embedded vocabulary -- a real, meaningful check that the metadata flows through correctly, just not as strong as an encode/decode proof would have been.
- **The two format branches remain hand-written `if`/`else`, not `magnetar-runtime`'s own `ProductionIngestionRegistry`.** Checked first, not assumed: that registry is a lookup-by-`ingestor_id` map (`register`/`get`/`ingest(ingestor_id, source)`) -- it does not itself inspect a bundle to decide which id to use, so a caller still needs exactly the same `is_gguf` decision before calling it. Routing through the registry would add a layer of indirection (register both ingestors under their ids, then look one up by a string this change would still have to compute) for a real embedder-side decision the registry was never built to make -- it exists for a Runtime holding an arbitrary, possibly not-compile-time-known set of externally-registered ingestors, not this crate's fixed, compile-time-known choice between exactly two.
