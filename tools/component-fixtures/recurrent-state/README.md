This test-only Component builds a graph with a fixed-size recurrent slot and
a two-output split through `magnetar:model-component-graph/graph-builder@1.11.0`.
Wasmtime tests exercise both prefill and decode and confirm unsupported
recurrent execution is rejected. The prefill segment export additionally
produces a stateless multi-output graph that can be planned normally.

`recurrent-state-legacy.component.wasm` is the earlier 1.11 fixture from
Magnetar commit `196f42a6`, built before the multi-output addition. A separate
test verifies that it still links and builds the same state graph.

Rebuild the checked-in fixture from this directory with:

```sh
cargo build --locked --release --target wasm32-unknown-unknown
wasm-tools component new target/wasm32-unknown-unknown/release/magnetar_recurrent_state_fixture.wasm \
  -o ../../../magnetar-runtime/fixtures/components/recurrent-state.component.wasm
```
