# Quality Gates

Magnetar validates repository changes with the same commands locally and in CI.

Commands are given in shell form first and PowerShell form where the two
differ. CI runs on Linux, macOS and Windows, so both are supported.

## Rust Toolchain

The repository pins Rust and its components in `rust-toolchain.toml`. Running
any cargo command in this directory installs that toolchain automatically:

```bash
rustup show
```

Do not install a different toolchain version to match CI. CI runs `rustup show`
for the same reason, so the pinned toolchain is the only one in use.

The declared minimum supported Rust version is the `rust-version` field in
`magnetar-runtime/Cargo.toml`, verified by the `quality / msrv` CI job.

## Local Commands

Run formatting:

```powershell
cargo fmt --all -- --check
```

Run compilation checks:

```bash
cargo check --locked --workspace --all-targets --all-features
```

Run Clippy with repository policy:

```bash
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
```

Run the complete Rust test suite:

```bash
cargo test --locked --workspace --all-targets
```

Run concrete Wasmtime Component Engine tests:

```bash
cargo test --locked -p magnetar-runtime --all-targets --features wasmtime-component-engine
```

Run hardware-independent Provider conformance tests:

```bash
cargo test --locked -p magnetar-runtime provider_conformance -- --nocapture
```

Run the End-to-End Local Inference Conformance suite (no GPU required):

```powershell
cargo test -p magnetar-runtime e2e_conformance -- --nocapture
```

Run the Post-Baseline Provider Roadmap contract tests:

```powershell
cargo test -p magnetar-runtime provider_roadmap -- --nocapture
```

Run the Post-Baseline Server API Roadmap contract tests:

```powershell
cargo test -p magnetar-runtime server_api_roadmap -- --nocapture
```

Build Rust documentation with warnings denied:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --all-features --no-deps
```

PowerShell:

```powershell
$env:RUSTDOCFLAGS="-D warnings"
cargo doc --locked --workspace --all-features --no-deps
```

Validate WIT packages:

```bash
wasm-tools component wit magnetar-runtime/wit/compute.wit
wasm-tools component wit magnetar-runtime/wit/observability.wit
```

Validate Component fixtures:

```bash
for wat in magnetar-runtime/fixtures/components/*.component.wat; do
  wasm-tools parse "$wat" -o /tmp/component-fixture.wasm
  wasm-tools validate /tmp/component-fixture.wasm --features component-model
done
```

PowerShell:

```powershell
Get-ChildItem magnetar-runtime/fixtures/components/*.component.wat | ForEach-Object {
  wasm-tools parse $_.FullName -o "$env:TEMP/component-fixture.wasm"
  wasm-tools validate "$env:TEMP/component-fixture.wasm" --features component-model
}
```

Validate OpenSpec artifacts:

```bash
openspec validate --all --strict
```

Generate coverage JSON and LCOV:

```bash
mkdir -p target/llvm-cov
cargo llvm-cov --locked --workspace --all-targets --all-features --ignore-filename-regex '(^|/)(target|tests?)/' --json --summary-only --output-path target/llvm-cov/coverage.json
cargo llvm-cov report --ignore-filename-regex '(^|/)(target|tests?)/' --lcov --output-path target/llvm-cov/lcov.info
```

Check the coverage ratchet:

```bash
cargo run --manifest-path tools/coverage-ratchet/Cargo.toml -- target/llvm-cov/coverage.json quality/coverage-baseline.json
```

Check dependency advisories, bans, licenses and sources:

```bash
cargo deny --all-features check
```

The policy lives in `deny.toml`. An exception belongs in that file with a
reason, never as a skipped CI step.

## Coverage Policy

The accepted baseline is stored in `quality/coverage-baseline.json`.

The baseline records measured production Runtime source coverage. Generated
build output and test source are excluded through the same coverage scope in
local and CI commands.

Exclusion works by filename, and cargo-llvm-cov appends its own patterns to the
one this repository passes -- including `tests.rs` and `*_tests.rs`. A filename
pattern cannot reach test code embedded in a production file, so an inline
`#[cfg(test)] mod tests { ... }` block would be measured as Runtime
implementation source and inflate the result, since test code is executed by
definition. Unit tests therefore live in a sibling `tests.rs` file
(`src/adapter.rs` declares `mod tests;`, whose body is `src/adapter/tests.rs`)
rather than inline.

Coverage is measured with `--all-features` so feature-gated modules
(`component_wasmtime`, `component_web`) are included.

To raise the baseline, generate a fresh coverage report after tests improve and
update `line_coverage_percent` to the measured line coverage. Reductions are
intentional policy changes and must be reviewed as normal version-controlled
changes.

Branch coverage is recorded as `null` until the selected coverage tooling
supports it reliably in this repository.
