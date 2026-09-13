## 1. Implementation

- [x] 1.1 Add `f32_to_f16(value: f32) -> u16` to `magnetar-runtime/src/model_loading.rs`, directly beside `f16_to_f32`: round-to-nearest-even, correct subnormal/overflow-to-infinity/infinity/NaN handling.
- [x] 1.2 Add `f32_to_bf16(value: f32) -> u16`, directly beside `bf16_to_f32`: round-to-nearest-even truncation, correct NaN handling.

## 2. Tests

- [x] 2.1 `f32_to_f16_round_trips_every_possible_f16_bit_pattern_exactly`: for all 65,536 `u16` values, decode via the already-trusted `f16_to_f32`, re-encode via `f32_to_f16`, assert the bit pattern matches exactly (NaN inputs: assert the re-encoded bits still decode to NaN, not bit-exact equality). Passed on the first run against the hand-derived bit-manipulation implementation -- strong correctness evidence.
- [x] 2.2 `f32_to_bf16_round_trips_every_possible_bf16_bit_pattern_exactly`: same, for all 65,536 `u16` values against `bf16_to_f32`/`f32_to_bf16`. Also passed on the first run.
- [x] 2.3 A handful of named-value spot checks for readability (`+0.0`, `-0.0`, `1.0`, smallest normal, smallest subnormal, largest finite, overflow-to-infinity, `+Inf`, `-Inf`) -- not load-bearing on their own (2.1/2.2 already cover them exhaustively) but documents intent for a reader who does not want to reason about the exhaustive test.
- [x] 2.4 Full regression: `cargo test -p magnetar-runtime --lib` (1258 passed), `cargo clippy -p magnetar-runtime --all-targets -- -D warnings` clean, `cargo fmt -p magnetar-runtime -- --check` clean. Both new functions are `#[allow(dead_code)]` (unused outside tests until Phase 2 consumes them) with a doc comment explaining why.

## 3. Documentation

- [x] 3.1 `openspec validate add-native-cuda-half-precision-compute --strict` passes.
- [x] 3.2 `design.md`'s Phase 2/Phase 3 sketch stands as the reference for the next two chantiers in this sequence; no README change in this phase (the README's "native CUDA F16/BF16 compute kernels do not exist yet" line stays accurate until Phase 2 lands).
