This test-only Component builds a graph with a single fixed-size recurrent
slot through `magnetar:model-component-graph/graph-builder@1.11.0`. It does
not execute the graph. Wasmtime tests exercise both prefill and decode and
confirm unsupported recurrent execution is rejected.

Rebuild the checked-in fixture from this directory with:

```sh
cargo build --locked --release --target wasm32-unknown-unknown
wasm-tools component new target/wasm32-unknown-unknown/release/magnetar_recurrent_state_fixture.wasm \
  -o ../../../magnetar-runtime/fixtures/components/recurrent-state.component.wasm
```
