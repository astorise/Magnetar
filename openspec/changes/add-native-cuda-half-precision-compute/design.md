## Context

The README names the remaining gap plainly: native CUDA `F16`/`BF16` compute kernels do not exist. Investigating the real blast radius before committing to a scope (matching this session's established practice of investigating before implementing) found two hardcoded-`f32` walls, not one:

- `providers/cuda`'s `CudaDeviceBuffer { slice: CudaSlice<f32>, .. }` -- every kernel wrapper (`add`, `mul`, `matmul`, `rmsnorm`, ...), the device allocation table, and `upload`/`download` are written against `f32` specifically.
- `magnetar-runtime`'s `HostTensor { shape: Vec<u64>, data: Vec<f32> }` -- the Provider-agnostic host/device transport type `TensorValue::Host` carries, which `upload`/`download` read/write directly (`clone_htod(&tensor.data)`/`clone_dtoh` typed against `CudaSlice<f32>`).

Unlike every prior chantier this session (each bounded to one crate/subsystem with a contained blast radius), a literal "just add F16/BF16 support" touches the shared Runtime-Provider transport contract itself. This was surfaced to the user, who chose to proceed in explicit, separately-committed-and-verified phases, starting with `magnetar-runtime`.

## Goals / Non-Goals

**Goals (this change, Phase 1 only):**
- Real, exhaustively-verified `f32_to_f16`/`f32_to_bf16` encoders in `magnetar-runtime`, completing the existing decoders.
- Document the full multi-phase plan so later phases have a stable reference and this change's narrow scope is legible in isolation.

**Non-Goals (deferred to later phases, explicitly not attempted here):**
- Retyping `HostTensor` or `CudaDeviceBuffer`.
- Any actual on-device native half-precision kernel.
- Wiring compute-dtype selection into the Runtime's planner.
- Touching `providers/cpu` / Reference CPU at all -- see Decisions below for why this is believed to be a non-goal *permanently*, not just for this phase.

## Decisions

- **`HostTensor` stays `Vec<f32>` -- it is not retyped, in this change or (per the current plan) in Phase 2 either.** Retyping the Provider-agnostic transport type that `magnetar-runtime`, `providers/cpu`, and `providers/cuda` all depend on would ripple through the entire compute surface (every Reference CPU kernel indexes `.data` as `f32` directly) for a benefit Phase 2 does not need: a real native-half-precision *device* buffer only needs half-precision bytes to exist between upload and download, not for the host-side transport type to carry them. Phase 2's plan converts `F32` host bytes to `F16`/`BF16` device bytes at the `CudaKernels::upload` boundary (using this change's `f32_to_f16`/`f32_to_bf16`), and back at `download` (using the pre-existing `f16_to_f32`/`bf16_to_f32`) -- `HostTensor` never sees a non-`f32` byte.
- **Reference CPU's `dtype_conversion` (`reference_cpu.rs`, explicitly `Float32 -> Float32` only) is not touched, and there is no plan to touch it.** It is a deliberate simplicity decision for a reference/conformance-oracle implementation that does not need performance, not a temporary gap parallel to the CUDA one -- `openspec/specs/compute/spec.md`'s own forward-looking "BF16 compute from INT8 stored weights" scenario is about Memory-Manager accounting, not a commitment for Reference CPU to ever compute in anything but `f32`. This matters for scoping: it means the *only* real target for "native half-precision compute" is `providers/cuda`, not a second parallel effort in `providers/cpu`.
- **Encoders are placed beside their decoder counterparts in `model_loading.rs`, not in a new module.** Consistent with how the file is already organized (the existing `f16_to_f32`/`bf16_to_f32` pair lives there, used by `tensor_from_raw_bytes`); no reason to split the pair across files.
- **Verification is exhaustive round-trip testing over all 65,536 `u16` bit patterns per format, not hand-picked cases.** `f16_to_f32`/`bf16_to_f32` are already trusted (covered by `f16_to_f32_handles_every_numeric_class_exactly`/`bf16_to_f32_handles_every_numeric_class_exactly`); decoding every possible bit pattern through them and re-encoding through the new functions must reproduce the original bit pattern exactly, since every such value is by construction exactly representable in its own format -- no rounding should ever be observable in a round trip that starts from an exact value. This is strictly stronger than hand-checked cases and cheap (131,072 iterations total). NaN is the one exception: many bit patterns decode to NaN and a NaN's exact payload is not required to round-trip bit-for-bit by IEEE 754, only "is still NaN" is checked for those inputs -- this mirrors how `f16_to_f32_handles_every_numeric_class_exactly` already treats NaN class-wise rather than bit-wise.
- **Rounding mode is round-to-nearest-even**, matching every other production numeric-conversion path in this codebase (dequantization, etc.) and standard practice (e.g. TensorFlow's/PyTorch's own `f32->bf16` truncate-with-rounding-bias trick, which this borrows for `f32_to_bf16`).

## Phase 2 (future chantier, sketched here for continuity, not implemented in this change)

- `providers/cuda`: `CudaDeviceBuffer` gains a dtype tag (`Float32` default, `Float16`, `BrainFloat16`), `upload`/`download` convert at the host/device boundary using this change's encoders/decoders, and at least one real native half-precision elementwise kernel (`add`/`mul`, the two simplest existing kernels) is added to `kernels.cu`. NVRTC's minimal preprocessor environment (documented at the top of `kernels.cu`) does not reliably expose `cuda_fp16.h`/`cuda_bf16.h` without further investigation; the lower-risk path is a manual IEEE 754 bit-manipulation device function mirroring the file's own existing `__int_as_float`-based `neg_inf()` helper, exactly like this change's Rust encoders -- avoiding any new header/include-path dependency. Verified against Reference CPU's `f32` output within tolerance, on real hardware (the RTX 3070 Ti workstation and/or the `arc-gpu-magnetar` self-hosted CI runner already used for `providers/cuda` verification).

## Phase 3 (future, not yet scoped in detail)

- Wiring an actual compute-dtype *choice* into the Runtime's planner (today, `KernelDescriptor` already separates `storage_dtype`/`compute_dtype`, but nothing selects `Float16`/`BrainFloat16` at dispatch time) -- deliberately left undesigned until Phase 2 exists to inform what the real selection contract should look like.

## Risks / Trade-offs

- **Two-function change with no visible behavior difference yet.** Everything this change adds is inert until Phase 2 consumes it. Accepted: matches this session's established pattern of landing a verified prerequisite before the change that uses it (e.g. `support-gguf-quantized-tensor-dequantization` landing dequantization before `resolve-gguf-quantized-projection-transpose-sequencing` wired it into the loader).
- **The full plan may still change shape once Phase 2 is actually attempted** (e.g. if NVRTC does expose `cuda_fp16.h` cleanly, that may be preferable to the manual bit-manipulation fallback sketched above). This design.md's Phase 2 section is a sketch for continuity, not a binding commitment.
