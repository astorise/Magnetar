## 1. Runtime and Reference CPU

- [x] 1.1 Add the optional `rotation_pairing` string rule to the `rope` Operator attribute schema.
- [x] 1.2 Implement `half_split` in `magnetar-runtime`'s Reference CPU `rope` (multi-head and partial RoPE), keep `interleaved` the default, reject unknown values; add known-value and equivalence tests (including against a numpy-style `rotate_half` formulation).
- [x] 1.3 Forward the graph node's `rotation_pairing` from the Component-graph dispatch to the Kernel. (No Rust-side graph builder emits `rope` nodes; the graph-equivalence oracle tests now rotate with `half_split` to match the Components.)

## 2. Providers

- [x] 2.1 `providers/cpu`: implement both pairings and rejection, with tests matching Reference CPU.
- [x] 2.2 `providers/cuda`: pass the pairing to `rope_kernel`, implement `half_split`, reject unknown values; CPU-vs-CUDA conformance for both pairings (GPU-gated). Implemented (astorise/Magnetar-provider-CUDA 589b5d9); GPU run 41008921749 (RTX 3060, driver 595.71.05): NVRTC compile and `rope_half_split_matches_reference_cpu` pass.
- [x] 2.3 `providers/wgpu`: not applicable -- WGPU implements only `add` today and has no `rope` kernel; the pairing must be part of its future `rope` implementation.

## 3. Components and fixtures

- [x] 3.1 `components/qwen` and `components/llama`: emit `rotation_pairing: half_split` on every `rope` node.
- [x] 3.2 Rebuild `qwen-real.component.wasm`, `llama-real.component.wasm` and `magnetar-cli/fixtures/qwen-real.component.wasm`; update their manifests and every pinned digest.
- [x] 3.3 Move the submodule pins after each repository's own validation.

## 4. Evidence and qualification

- [x] 4.1 Run `tests_real_checkpoint_numerics` on Reference CPU against the independent Qwen2.5-0.5B-Instruct corpus and record that tokens match and logits are within `f32-cross-provider-v1`. Result: all 16 tokens identical (" Paris. It is the largest city in Europe…"), max |logit difference| 5.4e-5 (was 10.48 with interleaved pairing); lifecycle checks pass.
- [x] 4.2 Run fmt/check/clippy/tests for the root and submodule workspaces and each changed Provider/Component repository.
- [x] 4.3 Confirm the CUDA comparison in `gpu-runner-smoke.yml` passes on real hardware. GPU run 41008921749: CUDA reproduces the independent corpus's 16 tokens, max |logit difference| 6.6e-5.
