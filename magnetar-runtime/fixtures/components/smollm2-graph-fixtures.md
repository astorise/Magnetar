# SmolLM2 test components

These artifacts are built from `components/smollm` at commit
`cc9614315930436ae9392494333e0af28b0bba6c` with Rust 1.100.0 and
`wasm-tools` 1.245.1. They are test fixtures, not production releases.

- `smollm2-blocked.component.wasm`: default build. Its graph exports refuse
  execution until the host validates the checkpoint activation.
- `smollm2-graph.component.wasm`: built with `test-only-assume-silu`, used
  only with known SiLU configurations in engine and numerical tests. The
  adjacent manifest pins this artifact's SHA-256.

Regenerate each from the submodule with `cargo build --locked --release
--target wasm32-unknown-unknown`, adding `--features test-only-assume-silu`
for the second, followed by `wasm-tools component new` on the resulting
`magnetar_component_smollm.wasm`. The component-integration workflow instead
builds fresh versions of both artifacts and passes their paths to the tests.
