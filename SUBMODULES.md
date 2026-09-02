# External Modules

Magnetar's Model Components, Providers, and Formats live in separate
repositories, pinned into this repository as git submodules under
`components/`, `providers/`, and `formats/`. This document defines
versioning/release ownership per submodule (`reach-architecture-freeze-1`
task 15.3) and the current Magnetar-to-module compatibility matrix (task
15.4).

## Modules

| Path | Repository | Status |
| --- | --- | --- |
| `components/qwen` | [Magnetar-component-Qwen](https://github.com/astorise/Magnetar-component-Qwen) | Real: implements `magnetar:model-component-graph@1.0.0` for the fixed E2E fixture architecture (see its own README) |
| `components/llama` | [Magnetar-component-Llama](https://github.com/astorise/Magnetar-component-Llama) | Empty `cargo new --lib` template |
| `formats/gguf` | [Magnetar-format-GGUF](https://github.com/astorise/Magnetar-format-GGUF) | Empty `cargo new --lib` template |
| `formats/safetensors` | [Magnetar-format-safetensors](https://github.com/astorise/Magnetar-format-safetensors) | Empty `cargo new --lib` template |
| `providers/cpu` | [Magnetar-provider-CPU](https://github.com/astorise/Magnetar-provider-CPU) | Empty `cargo new --lib` template; the real Reference CPU Provider implementation still lives in `magnetar-runtime` pending extraction (`reach-architecture-freeze-1` task group 14) |
| `providers/cuda` | [Magnetar-provider-CUDA](https://github.com/astorise/Magnetar-provider-CUDA) | Empty `cargo new --lib` template |

## Versioning and release ownership

- **Each module versions itself independently.** A module's `Cargo.toml`
  `version` is that module repository's own concern; nothing in this
  repository requires it to track Magnetar's own version number, and a
  module's maintainer (today: whoever authors its real implementation) is
  the one who decides when to cut a release and what that release's
  version means for that module.
- **Magnetar pins, not floats.** This repository never depends on a
  module's floating branch tip; every module is a `160000` gitlink pinned
  to one exact commit (see `reach-architecture-freeze-1` task 15.1, and
  its own note about the `.gitmodules`-alone failure mode this pinning
  discipline exists to prevent). Advancing a module's pin in this
  repository is a change to *this* repository (a normal commit touching
  the gitlink), reviewed the same way any other Magnetar change is,
  independent of whether the module repository itself calls the commit
  being pinned to a "release."
- **This repository owns compatibility, not the module.** Whether a given
  module commit actually works with the current `magnetar-runtime` is
  determined by this repository's own CI (`submodule-integration` in
  `.github/workflows/quality.yml`) at the pinned commit, not by any
  version-number contract the module declares. A module bumping its own
  `Cargo.toml` version is not itself a compatibility claim.
- **No compatibility range pinning yet.** Because every module except
  `components/qwen` is still a template, there is nothing yet to define a
  meaningful version *range* against (a `^0.1` style constraint would be
  vacuous). Once a module has more than one real, meaningfully different
  release, this section should be extended with actual compatible-version
  ranges per module rather than the current one-pinned-commit-at-a-time
  model.

## Compatibility matrix

| `magnetar-runtime` | `components/qwen` | Notes |
| --- | --- | --- |
| This branch (`make-first-native-datapath-authoritative`) | `1a9bca7` or later | Requires `magnetar:model-component-graph@1.0.0`; not yet wired into the production generation path (`reach-architecture-freeze-1` task 11.5) |

Every other module has no real content yet, so no compatibility claim is
meaningful for it beyond "the empty template builds" (verified by
`submodule-integration`). This table should gain a row per module once
that module has real content to be compatible (or not) with.
