# External Modules

Magnetar's Model Components, Providers, and Formats live in separate
repositories, pinned into this repository as git submodules under
`components/`, `providers/`, and `formats/`. This is a normative
requirement, not only a packaging convention -- see `project-architecture`'s
"Model Components, Providers, and Formats Are Externalized" requirement
(`externalize-runtime-extension-modules`), which also names the one
deliberate exception (a minimal, generic in-crate Reference CPU double for
the Core's own test suite) and is enforced by a CI dependency guard
(`.github/workflows/quality.yml`'s `submodule-integration` job). This
document defines versioning/release ownership per submodule
(`reach-architecture-freeze-1` task 15.3) and the current
Magnetar-to-module compatibility matrix (task 15.4).

## Modules

| Path | Repository | Status |
| --- | --- | --- |
| `components/qwen` | [Magnetar-component-Qwen](https://github.com/astorise/Magnetar-component-Qwen) | Real: implements `magnetar:model-component-graph@1.0.0` for the fixed E2E fixture architecture, and is the exclusive production graph source under the strict path (`reach-architecture-freeze-1` task 11.5) -- see its own README |
| `components/llama` | [Magnetar-component-Llama](https://github.com/astorise/Magnetar-component-Llama) | Empty `cargo new --lib` template |
| `formats/gguf` | [Magnetar-format-GGUF](https://github.com/astorise/Magnetar-format-GGUF) | Real: parses the full GGUF container into generic `ModelTensorMetadata`, for the `F32`/`F16`/`BF16`/`I8`/`I16`/`I32`/`I64`/`F64`/`Q4_K`/`Q5_K`/`Q8_0` subset (`implement-model-format-parsers`) -- see its own README |
| `formats/safetensors` | [Magnetar-format-safetensors](https://github.com/astorise/Magnetar-format-safetensors) | Real: parses Safetensors files into generic `ModelTensorMetadata`, covering every standard Safetensors dtype (`implement-model-format-parsers`) -- see its own README |
| `providers/cpu` | [Magnetar-provider-CPU](https://github.com/astorise/Magnetar-provider-CPU) | Real: independent extraction of `ReferenceCpuExecutor`/kernels/SIMD detection (`reach-architecture-freeze-1` task group 14), plus the `TensorValue` structured-error-channel fix (`generalize-first-native-provider-dispatch`) -- see its own README |
| `providers/cuda` | [Magnetar-provider-CUDA](https://github.com/astorise/Magnetar-provider-CUDA) | Real: `CudaProvider` baseline -- device discovery, all `operator-scope` required-now kernels via NVRTC, full `ProviderExecutionApi`, passing `provider-core`/`provider-compute` conformance on real hardware (`implement-cuda-provider-baseline`) -- see its own README |

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
- **No compatibility range pinning yet.** `components/llama` is still a
  template; the other five modules are real but each still has exactly one
  or two real releases, so there is nothing yet to define a meaningful
  version *range* against (a `^0.1` style constraint would be vacuous). Once
  a module has more than one real, meaningfully different release, this
  section should be extended with actual compatible-version ranges per
  module rather than the current one-pinned-commit-at-a-time model.

## Compatibility matrix

| `magnetar-runtime` | Module | Commit | Notes |
| --- | --- | --- | --- |
| This branch (`make-first-native-datapath-authoritative`) | `components/qwen` | `e5422c7` or later | Requires `magnetar:model-component-graph@1.0.0`; is the exclusive production graph source under the strict path (`reach-architecture-freeze-1` task 11.5) |
| This branch | `formats/gguf` | `88ae910` or later | Requires `magnetar-runtime`'s `model` module (`ModelTensorMetadata`, `ModelDType`, `ModelQuantization`/`ModelQuantizationFormat`, including the `GgufQ8` variant this parser's development added); not yet wired into `Model Loading` (`implement-model-format-parsers`'s explicit non-goal) |
| This branch | `formats/safetensors` | `7492a04` or later | Requires `magnetar-runtime`'s `model` module; not yet wired into `Model Loading` (same non-goal as above) |
| This branch | `providers/cpu` | `ed9c58c` or later | Requires `magnetar-runtime`'s Provider/Device/Kernel/Tensor contracts, plus (uniquely among these modules) `HostTensor`/`ReferenceCpuError`/`ReferenceCpuErrorCode` specifically, imported rather than redefined (`reach-architecture-freeze-1` task group 14); never referenced by `magnetar-runtime`. `a855224` requires `write_tensor_value`/`write_tensor_value_admitted`'s structured error channel (`generalize-first-native-provider-dispatch`) -- earlier commits do not compile against this branch's `magnetar-runtime`. `42608e1` additionally fixes `execute_invocation_with_memory_manager`'s Memory Manager admission leak (`enable-device-resident-kernel-chaining` task group 9): every Kernel dispatch previously admitted a fresh, never-released allocation for its output resource id, unbounded over a long-running session. `560895b` further makes that same function honor a caller-pre-admitted output resource (via the new `MemoryManager::admit_kernel_output`) instead of always self-admitting -- required for `unify-provider-output-admission-and-residency`'s Runtime-owned output pre-admission to actually take effect through this Provider; falls back to self-admission unchanged when no pre-admission is found. `ed9c58c` adds `publish = false` (documentation only, no code change) |
| This branch | `providers/cuda` | `8eb0160` or later | Requires `magnetar-runtime`'s Provider/Device/Kernel/Tensor/Memory contracts and the same `write_tensor_value`/`write_tensor_value_admitted` error-channel shape as `providers/cpu` above; never referenced by `magnetar-runtime`. Its own kernel/Memory Manager/conformance behavior was verified against real GPU hardware (RTX 3070 Ti Laptop GPU), not just compiled -- see `implement-cuda-provider-baseline` and `enable-device-resident-kernel-chaining`. `557ceaa` replaces host-resident storage with a real, persistent device allocation table and explicit-only host/device data movement -- `7ac13b1` and earlier commits are correct but do not satisfy the CUDA Provider baseline's own "Explicit Data Movement"/"Memory Manager Integration" spec requirements literally (device residency was claimed to the Memory Manager but not physically real). `aea1915` additionally fixes the same Memory Manager admission leak as `providers/cpu`'s `42608e1` above (identical bug pattern, task group 9). `7997a2f` further makes that same function honor a caller-pre-admitted output resource, identical to `providers/cpu`'s `560895b`, and adds `benches/kernel_chaining.rs` (measurement-only, `cargo bench` opt-in, not part of `cargo test`/CI) -- on this workstation's RTX 3070 Ti, chaining kernels via resident device buffers measured ~2.65x faster than a forced host round-trip between them, moving 5x less data across 5x fewer H2D/D2H crossings, for `unify-provider-output-admission-and-residency`. `8eb0160` closes three `docs/audits/cuda-provider-full-audit-2026-09-05.md` findings: a deterministic (real `cuMemAlloc`-failure, not hand-constructed) out-of-device-memory test, an `#[ignore]`-by-default `hardware_conformance_actually_ran_not_silently_skipped` test that `gpu-runner-smoke.yml` now runs via `--include-ignored` so a green CI run can't mean "silently skipped", and a new `deny.toml` (paired with `publish = false`) making `cudarc`'s dependency graph a real, passing `cargo deny` check for the first time |

`components/llama` has no real content yet, so no compatibility claim is
meaningful for it beyond "the empty template builds" (verified by
`submodule-integration`). This table should gain a row for it once it has
real content to be compatible (or not) with.
