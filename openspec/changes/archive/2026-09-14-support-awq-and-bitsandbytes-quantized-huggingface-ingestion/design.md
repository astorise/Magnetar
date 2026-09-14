## Context

GPTQ's own chantier established the real methodology for this repository's quantization support: dequantize to `F32` at ingestion time, verified against real bytes from a public checkpoint via targeted HTTP Range requests (not a full download), compared element-by-element against the same weight's own unquantized counterpart. AWQ and BitsAndBytes both needed the identical rigor -- each has its own real, independently-discoverable packing convention that a documentation-only port risks getting subtly wrong (the exact failure mode GPTQ's own "+1" zero-point offset investigation was built to catch).

## Goals / Non-Goals

**Goals:**
- Real AWQ 4-bit dequantization, verified against real bytes from a public checkpoint, with the packing convention ported directly from AutoAWQ's own real source (`packing_utils.py`) rather than reconstructed from a written description.
- Real BitsAndBytes 4-bit (NF4, with double quantization -- the real, common `transformers`/`bitsandbytes` default) dequantization, verified the same way.
- Both integrated into the existing GPTQ detection pipeline without breaking it -- a bundle using any one of the three schemes ingests correctly; `gptq.rs`'s own tests continue to pass unmodified except for the one deliberately-changed behavior (skip, not error, on a `.qweight` tensor with no `.g_idx`).

**Non-Goals:**
- FP4 BitsAndBytes specifically verified against a real checkpoint (only NF4 was; the dequantization algorithm is generic over whatever codebook `quant_map` declares, so FP4 support is a natural consequence of the same code path, but not independently checkpoint-verified here).
- A full, real, downloaded-checkpoint end-to-end *generation* proof for either scheme -- the same scope boundary `support-gptq-quantized-huggingface-ingestion` drew for GPTQ.
- 8-bit BitsAndBytes (`llm_int8`, a genuinely different, non-block-quantized scheme with its own real complexity -- outlier-channel handling) or non-GEMM AWQ kernel variants (`version: "marlin"`/`"exllama"`, which pack differently again).

## Decisions

- **Port AWQ's real interleave from actual reference source, not written description.** `AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]` and the two-step unpack-then-reorder algorithm were fetched directly from AutoAWQ's own `awq/utils/packing_utils.py` (a real, public, actively-maintained reference implementation) and independently confirmed empirically against real checkpoint bytes -- both steps matter: the source gives the intended algorithm, the real-byte comparison confirms this ingestor's own port of it is actually correct.
- **BitsAndBytes' extraction reads real file bytes, breaking the GPTQ/AWQ pattern deliberately, not by oversight.** `extract_gptq_projections`/`extract_awq_projections` derive everything they need from tensors' own declared `shape` metadata alone; BitsAndBytes' real logical shape and block sizes live only inside a JSON blob no tensor's `shape` field carries. `extract_bnb_projections` takes `&dyn ProductionArtifactPayloadSource` for exactly this reason, reading the (tiny, ~100-200 byte) `quant_state` blob during extraction -- a real, justified architectural difference, documented as such rather than forced into the other two extractors' uniform shape.
- **Mutate the packed weight tensor's metadata in place, rather than remove-and-resynthesize like GPTQ/AWQ.** A BitsAndBytes-quantized projection's packed codes already occupy the exact raw name `<prefix>.weight` a plain weight would use; `extract_bnb_projections` corrects that entry's declared `shape`/`storage_dtype`/`size_bytes` directly instead of removing it and pushing a differently-named placeholder. This surfaced a real subtlety the other two schemes never hit: the packed tensor's own raw name still flows through the standard renaming loop (unlike a sibling tensor, which is removed from `tensors` before that loop runs), so its *location* ends up under the *canonical* key, not the raw one -- `extract_bnb_projections`'s own `weight_range.identity = canonical_name` line exists specifically for this, found and fixed while writing this change's own end-to-end `ingest()`-level test (the low-level `dequantize_bnb_projection` test alone never exercises this wiring).
- **BitsAndBytes projections are not excluded from the existing transpose.** Confirmed empirically (dequantized values compared row-major `[out_features, in_features]` directly against the real unquantized checkpoint, no transpose applied, and the comparison matched) -- BitsAndBytes' own packed-byte layout reconstructs to `nn.Linear`'s real storage convention, unlike GPTQ/AWQ's `qweight` which packs the input dimension first.

## Risks / Trade-offs

- **FP4 (non-NF4) BitsAndBytes remains unverified against a real checkpoint**, though the same code path should handle it (the codebook itself, not the algorithm, differs). Documented as a Non-Goal, not silently assumed identical.
- **8-bit BitsAndBytes (`llm_int8`) and non-GEMM AWQ kernel variants remain entirely unaddressed** -- both are real, separately-shaped conventions this change does not attempt.
