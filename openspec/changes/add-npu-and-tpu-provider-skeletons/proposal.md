## Why

`add-rocm-metal-provider-skeletons-and-real-wgpu-provider` closed the Metal/ROCm/WGPU portion of the original scope charter's additional-Providers item, but left NPU and TPU untouched. The user confirmed this gap directly and asked for both to be closed next, then multi-device execution. As with Metal/ROCm, this repository's tooling has no Intel NPU or Coral Edge TPU hardware anywhere, so real hardware verification is not possible; this change draws the same honest line established for ROCm: real device-discovery skeletons, built against real, publicly-documented vendor APIs (fetched directly from their real headers, not recalled from memory), gracefully reporting unavailable everywhere this repository can test.

## What Changes

- New submodule `providers/npu` (`Magnetar-provider-NPU`): `NpuProvider` dynamically loads the real Intel oneAPI Level Zero loader library (`ze_loader.dll`/`libze_loader.so.1`) and calls the real `zeInit(ZE_INIT_FLAG_VPU_ONLY)`/`zeDriverGet` entry points -- `ZE_INIT_FLAG_VPU_ONLY` is Level Zero's own real flag scoping discovery to NPU/VPU-class devices specifically, distinct from its GPU discovery path. Gracefully reports `ProviderHealth::Unavailable` when no such driver is found -- true everywhere in this repository's tooling. No compute Kernels.
- New submodule `providers/tpu` (`Magnetar-provider-TPU`): `TpuProvider` dynamically loads the real Google Coral `libedgetpu` runtime library (`edgetpu.dll`/`libedgetpu.so.1`/`libedgetpu.1.dylib`) and calls the real `edgetpu_list_devices`/`edgetpu_free_devices` C API. Targets the real, purchasable local Coral Edge TPU accelerator specifically, not Google Cloud TPU (a cloud service with no local device to discover). Gracefully reports unavailable everywhere this repository's tooling can test. No compute Kernels.
- `.github/workflows/quality.yml`: `provider-integration` job builds/tests both new Providers and runs `cargo deny` for their `libloading` dependency; `submodule-integration` job's full test sweep includes both.
- **BREAKING**: none. Both are new, independent submodules; no existing crate's public contract changes.

## Capabilities

### New Capabilities
- `npu-provider`: real device-discovery-only Provider baseline for Intel NPU/VPU-class devices via Level Zero, unavailable everywhere this repository's tooling can verify.
- `tpu-provider`: real device-discovery-only Provider baseline for Google Coral Edge TPU devices via libedgetpu, unavailable everywhere this repository's tooling can verify.

### Modified Capabilities
(none)

## Impact

- New repositories: `Magnetar-provider-NPU`, `Magnetar-provider-TPU`, each pinned into this repository as a git submodule (`providers/npu`, `providers/tpu`) via `.gitmodules`.
- `.github/workflows/quality.yml`: `provider-integration` and `submodule-integration` jobs extended.
- `SUBMODULES.md`: Modules table and Compatibility matrix updated for both.
