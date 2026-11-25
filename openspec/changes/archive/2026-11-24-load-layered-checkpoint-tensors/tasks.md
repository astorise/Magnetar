## 1. Hugging Face loader

- [x] 1.1 Names for the packed tables, fused experts, router projection and scales.
- [x] 1.2 `layered_tensors`: per-layer slices, split and stacked experts, folded router; partial reads of a stored tensor.

## 2. GGUF loader

- [x] 2.1 `gemma4` names for the packed tables, fused and separate experts, router and `.scale` tensors.
- [x] 2.2 `layered_tensors`: the same derivations, cut in stored bytes on row or block boundaries.

## 3. Tests

- [x] 3.1 Hugging Face derivations against hand-computed values, and the refusals.
- [x] 3.2 A `gemma4` GGUF ingest with a `Q8_0` per-layer table; mutation checks.
- [x] 3.3 The Component's logit comparison with `transformers` on the `ple` and `moe` bundles through the Hugging Face loader.

## 4. Validation

- [x] 4.1 `cargo fmt`, `clippy -D warnings` and tests in both loaders; Magnetar pins them.
