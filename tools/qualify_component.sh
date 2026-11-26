#!/usr/bin/env bash
# Qualifies a Model Component against this Runtime.
#
# A Component repository tests its graph code on the host (golden graphs and
# refusals); it cannot tell whether the Runtime accepts the artifact it builds.
# This builds the Component found at `components/<name>` for the Component Model
# and runs the Runtime's tests that consume it, in place of the checked-in
# artifact, through the environment variable each test reads:
#
#   tools/qualify_component.sh list           the Components it knows and what it runs for each
#   tools/qualify_component.sh <name>         build components/<name>, validate it, run its tests
#
# It needs, in this checkout: the Component at its path, the Runtime's own
# dependencies (`tools/repositories.sh fetch` for every other pinned
# repository), the `wasm32-unknown-unknown` target and `wasm-tools`. The
# reusable workflow `component-qualification.yml` prepares all of that, for a
# Component repository's pull request or by hand.
#
# A Component is added by one `case` below: its crate path, the artifact its
# crate builds, the variable the tests read it through, and the tests.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

die() { echo "qualify_component: $*" >&2; exit 1; }

known="llama smollm"

# Sets dir, artifact and variable for `$1`, and defines run_tests.
describe() {
  case "$1" in
    llama)
      dir=components/llama
      artifact=magnetar_component_Llama.wasm
      variable=MAGNETAR_LLAMA_TEST_COMPONENT
      tests_text="the GGUF and Hugging Face ingestion of identical Llama weights generate the same tokens"
      run_tests() {
        cargo test --locked --manifest-path submodule-workspace/Cargo.toml \
          -p magnetar-integration-production-loading tests_gguf_loading_e2e
      }
      ;;
    smollm)
      dir=components/smollm
      artifact=magnetar_component_smollm.wasm
      variable=MAGNETAR_SMOLLM2_TEST_COMPONENT
      tests_text="the engine's graph tests, SmolLM2 and SmolLM3 (rotary and NoPE) against Transformers, block-scaled FP8"
      run_tests() {
        cargo test --locked -p magnetar-engine --features wasmtime-component-engine smollm
        cargo test --locked --manifest-path submodule-workspace/Cargo.toml \
          -p magnetar-integration-production-loading tests_smollm
        cargo test --locked --manifest-path submodule-workspace/Cargo.toml \
          -p magnetar-integration-production-loading tests_fp8_logits
      }
      ;;
    *) return 1 ;;
  esac
}

case "${1:-}" in
  list)
    for name in $known; do
      describe "$name"
      printf '%-8s %s\n         builds %s, read through %s\n         runs: %s\n' \
        "$name" "$dir" "$artifact" "$variable" "$tests_text"
    done
    exit 0
    ;;
  '' | -h | --help)
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac

name="$1"
describe "$name" || die "unknown Component '$name' (known: $known)"
[ -f "$dir/Cargo.toml" ] || die "no Component crate at $dir"
command -v wasm-tools > /dev/null || die "wasm-tools is not installed"

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
component="$scratch/$name.component.wasm"

echo "::group::Build $dir for the Component Model"
cargo build --locked --release --manifest-path "$dir/Cargo.toml" --target wasm32-unknown-unknown
wasm-tools component new "$dir/target/wasm32-unknown-unknown/release/$artifact" -o "$component"
wasm-tools validate "$component" --features component-model
echo "$name: $(wc -c < "$component") bytes, a valid Component Model artifact"
echo "::endgroup::"

echo "Running the Runtime's tests on it: $tests_text"
export "$variable=$component"
run_tests
