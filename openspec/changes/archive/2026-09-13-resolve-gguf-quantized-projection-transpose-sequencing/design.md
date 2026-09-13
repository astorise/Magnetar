## Context

`support-gguf-quantized-tensor-dequantization`'s own design.md identified this exact blocker and deliberately deferred it: `loaders/gguf`'s `weight_layout::TransposingPayloadSource` performs a real byte-level transpose of every 2D projection weight, computing row/column strides from `storage_dtype.descriptor().size_bytes()` -- a flat per-element byte width. This is correct for `F32`/`F16`/`BF16` and would silently scramble a block-quantized tensor's structure if applied to its raw bytes directly, since a block covers a fixed run of logical elements at a fixed byte size, not "N bytes per element."

## Goals / Non-Goals

**Goals:**
- Let `loaders/gguf` actually load a real quantized GGUF checkpoint end to end.
- Sequence correctly: dequantize to `F32` first, transpose second -- never the reverse.
- Verify against a real, genuinely quantized public checkpoint, not only a synthetic construction.

**Non-Goals:**
- Changing `magnetar-runtime`'s own generic `Q8_0`/`Q4_K`/`Q5_K` dequantization (`support-gguf-quantized-tensor-dequantization`) at all -- it remains available generically, simply never reached for a GGUF-sourced tensor once this loader has already resolved it to `F32`.
- GPTQ/AWQ/BitsAndBytes.

## Decisions

- **`loaders/gguf` gets its own copy of the dequantization algorithm, not a shared dependency on `magnetar-runtime`'s.** Consistent with this crate's existing "ported, not shared" convention for `weight_layout.rs`/`derived_lm_head.rs` (already duplicated from `loaders/huggingface` for the identical architectural reason: independent externalized modules cannot depend on each other). The alternative -- exposing `magnetar-runtime`'s dequantization functions publicly for a loader to call -- would work but creates an odd dependency shape (a `magnetar-runtime` implementation detail becoming public API surface specifically so an externalized module can call it), and this logic is small, stable, and already has its own dedicated test coverage in `magnetar-runtime`; porting a byte-for-byte-identical second copy carries the same acceptable duplication cost `weight_layout.rs` already established as the norm here.

- **Dequantization happens inside `GgufPayloadSource::read_payload`, at the exact point raw bytes are read from the file** -- before they are handed to `weight_layout::TransposingPayloadSource` (which wraps the whole payload source and is unaware whether a given tensor was ever quantized). This means the transpose logic itself needed *zero* changes: by the time it receives bytes for a tensor whose `storage_dtype` it reads as `F32`, those bytes genuinely are flat `F32`, exactly matching what it already handles correctly.

- **The discovered tensor's metadata is fully rewritten to reflect the dequantized reality** (`storage_dtype: F32`, `size_bytes` recomputed as `element_count * 4`, `quantization: None`, `digest: None`) at discovery time, before naming/transpose ever runs. This is what lets every later step -- including `magnetar-runtime`'s own Model Loading, which never learns this tensor's real GGUF origin -- treat it identically to a tensor that was never quantized in the first place. A `TensorLocation` internal to `weights.rs` separately tracks the *original* file offset/length (to know how many raw bytes to read and where) and which dequantization function to apply, decoupled from the *declared* (post-dequantization) size the manifest and Model Loading actually see.

- **A declared content digest, if any, is checked against the original quantized bytes, before dequantization** -- mirroring `weight_layout.rs`'s own established "verify before transform" precedent for its byte transpose. In practice this is moot today (`formats/gguf` never populates a digest for a quantized tensor, per its own documented "cannot be materialized into a HostTensor at all today, so no digest could ever be checked anyway" reasoning), but implementing it correctly now means a future `formats/gguf` version that does start populating quantized-tensor digests will not need this code touched again.

- **Verification escalated beyond the synthetic case**: a hand-constructed, non-square, genuinely `Q8_0`-quantized projection weight is verified end to end through the full `GgufIngestor::ingest` pipeline against hand-computed expected values (proving the sequencing itself, not just that dequantization or transposition individually work) -- and, going further, the real public `Qwen2.5-0.5B-Instruct-GGUF` checkpoint's actual `Q8_0` export was downloaded and run through real production generation, producing the *exact same output* as its unquantized F16 GGUF export and Safetensors counterpart for the same prompt. This is strong, real-world evidence: `Q8_0` quantization is real, lossy numerical approximation, so identical output is not guaranteed by construction -- getting it anyway is a meaningful confirmation, not merely "generation did not crash."

## Risks / Trade-offs

- **Two independent copies of the same dequantization algorithm now exist** (`magnetar-runtime` and `loaders/gguf`), each with their own test suite. A future correctness fix to one (e.g. a newly discovered edge case) must be manually mirrored to the other -- an accepted, already-established cost in this codebase's externalization architecture, not a new one this change introduces.
- **GPTQ/AWQ/BitsAndBytes remain entirely out of scope.** These fail earlier anyway, at `formats/gguf`'s own parse layer (their `ggml_type` values are never GGUF concepts to begin with), so this change does not change their behavior at all.
