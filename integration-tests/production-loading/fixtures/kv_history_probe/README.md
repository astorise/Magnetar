# KV history Component probe

This synthetic Component exercises graph-builder 1.16's copy-based KV history
ABI. It builds graphs only; it is not an Olala model or a numerical reference.
The API dependency is pinned to the contract change under review.

From this directory, with Rust 1.100.0, `wasm32-unknown-unknown`, and
`wasm-tools` installed, regenerate the checked-in Component with:

```sh
cargo +1.100.0 build --locked --release --target wasm32-unknown-unknown
wasm-tools component new target/wasm32-unknown-unknown/release/magnetar_kv_history_probe.wasm \
  -o ../../../../magnetar-runtime/fixtures/components/kv-history-probe.component.wasm
wasm-tools validate ../../../../magnetar-runtime/fixtures/components/kv-history-probe.component.wasm --features component-model
```

The generated Component imports `graph-builder@1.16.0` and
`model-config@1.16.0`, with no WASI capabilities. The Wasmtime integration test builds prefill and decode graphs
and verifies their paired Runtime-issued KV identities and descriptors.
With rustc 1.100.0 and wasm-tools 1.245.1, the Component's SHA-256 is
`0d970cfdc4d3ea9806ae6f22c1c3ab7223d820f29caf62284b4b4ef49dd5e143`.
