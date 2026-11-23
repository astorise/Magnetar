# ModernBERT encoder Component fixture

`modernbert-encoder.component.wasm` is built from the ModernBERT submodule
pinned by this checkout. It is a test fixture, not a checkpoint or a
production model. The test executes its graph with synthetic weights and
compares its outputs against the pinned Transformers reference in
`../modernbert-encoder-reference.json`.

Build with Rust 1.100.0, the `wasm32-unknown-unknown` target and
`wasm-tools` 1.245.1:

```sh
cargo build --locked --release --manifest-path components/modernbert/Cargo.toml --target wasm32-unknown-unknown
wasm-tools component new components/modernbert/target/wasm32-unknown-unknown/release/magnetar_component_modernbert.wasm -o magnetar-runtime/fixtures/components/modernbert-encoder.component.wasm
wasm-tools validate magnetar-runtime/fixtures/components/modernbert-encoder.component.wasm --features component-model
```

The integration workflow additionally runs the numerical test against a
fresh build via `MAGNETAR_MODERNBERT_TEST_COMPONENT`, so fixture behavior is
checked against current source even if toolchain metadata changes the bytes.
