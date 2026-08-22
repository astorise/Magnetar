# Quality Gates

Magnetar validates repository changes with the same commands locally and in CI.

## Rust Toolchain

The repository pins Rust in `rust-toolchain.toml`.

Install or update the pinned toolchain with:

```powershell
rustup show
rustup component add rustfmt clippy llvm-tools-preview
```

## Local Commands

Run formatting:

```powershell
cargo fmt --all -- --check
```

Run compilation checks:

```powershell
cargo check --workspace --all-targets --all-features
```

Run Clippy with repository policy:

```powershell
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Run the complete Rust test suite:

```powershell
cargo test --workspace --all-targets
```

Run concrete Wasmtime Component Engine tests:

```powershell
cargo test -p magnetar-runtime --features wasmtime-component-engine
```

Build Rust documentation with warnings denied:

```powershell
$env:RUSTDOCFLAGS="-D warnings"
cargo doc --workspace --all-features --no-deps
```

Validate WIT packages:

```powershell
wasm-tools component wit magnetar-runtime/wit/compute.wit
wasm-tools component wit magnetar-runtime/wit/observability.wit
```

Validate Component fixtures:

```powershell
Get-ChildItem magnetar-runtime/fixtures/components/*.component.wat | ForEach-Object {
  wasm-tools parse $_.FullName -o "$env:TEMP/component-fixture.wasm"
  wasm-tools validate "$env:TEMP/component-fixture.wasm" --features component-model
}
```

Validate OpenSpec artifacts:

```powershell
openspec validate --all --strict
```

Generate coverage JSON and LCOV:

```powershell
New-Item -ItemType Directory -Force target/llvm-cov | Out-Null
cargo llvm-cov --workspace --all-targets --all-features --ignore-filename-regex '(^|/)(target|tests?)/' --json --summary-only --output-path target/llvm-cov/coverage.json
cargo llvm-cov report --ignore-filename-regex '(^|/)(target|tests?)/' --lcov --output-path target/llvm-cov/lcov.info
```

Check the coverage ratchet:

```powershell
cargo run --manifest-path tools/coverage-ratchet/Cargo.toml -- target/llvm-cov/coverage.json quality/coverage-baseline.json
```

## Coverage Policy

The accepted baseline is stored in `quality/coverage-baseline.json`.

The baseline records measured production Runtime source coverage. Generated
build output and test source are excluded through the same coverage scope in
local and CI commands.

To raise the baseline, generate a fresh coverage report after tests improve and
update `line_coverage_percent` to the measured line coverage. Reductions are
intentional policy changes and must be reviewed as normal version-controlled
changes.

Branch coverage is recorded as `null` until the selected coverage tooling
supports it reliably in this repository.
