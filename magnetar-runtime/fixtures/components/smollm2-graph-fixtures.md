# SmolLM2 test components

`smollm2-graph.component.wasm` is built from the pinned `components/smollm`
submodule with Rust 1.100.0 and `wasm-tools`. It reads activation and per-layer
RoPE selection from `model-config@1.9.0` and rejects unknown or unsupported activations. The
adjacent manifest pins its SHA-256. This is a test fixture, not a production
release.

Regenerate with `cargo build --locked --release --target
wasm32-unknown-unknown`, followed by `wasm-tools component new` on
`magnetar_component_smollm.wasm`. The component-integration workflow builds
a fresh artifact and passes its path to the tests.
