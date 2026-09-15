## 1. Infrastructure investigation and fix

- [x] 1.1 Verified the CI runner's real GPU visibility from the latest `gpu-runner-smoke.yml` log: only `GPU 0` (RTX 3060) shown, contradicting the user's claim of multi-GPU hardware.
- [x] 1.2 Investigated via `mcp__homelab__wsl_get_host_capabilities` (this development workstation, a different single GPU -- not the CI runner) and real `kubectl` access to the Talos cluster (`C:\Users\astor\Git\talos\kubeconfig`): `kubectl describe node talos-4rh-gm0` confirmed real `nvidia.com/gpu: 2` capacity/allocatable.
- [x] 1.3 Found the real cause in `C:\Users\astor\Git\talos\arc-magnetar-gpu-job-hook.yaml`: `nvidia.com/gpu: 1` in both `requests`/`limits`, while sibling job hooks (`arc-candle-*`) on the same cluster already requested 2.
- [x] 1.4 Raised both to `2`, applied via `kubectl apply` against the live cluster, and re-verified via a real `gpu-runner-smoke.yml` dispatch (`34890018266`): `nvidia-smi` genuinely listed two real GPUs (`GPU 0`, `GPU 1`, both RTX 3060).
- [x] 1.5 Found and fixed a real, separate, pre-existing gap along the way (unrelated to the GPU-count fix): the workflow never checked out `loaders/gguf`/`formats/gguf`, both real dependencies of its own `integration-tests/production-loading` step -- fixed and re-verified with a second dispatch.

## 2. `providers/cuda`

- [x] 2.1 `CudaProvider::for_device(ordinal, provider_name)`: real constructor selecting a specific device ordinal under a caller-chosen Provider name; `new()` redefined as exactly `for_device(0, CUDA_PROVIDER_NAME)`.
- [x] 2.2 Threaded `provider_name` through `device::cuda_device_descriptor`, `advertisements::cuda_kernel_advertisements` (and its two private helpers), and `CudaExecutor::new`/`provider_binding()`/execution-id generation -- every place `CUDA_PROVIDER_NAME` was previously read as a constant.
- [x] 2.3 New tests: `for_device_zero_matches_new_exactly` (proves the refactor changed nothing about ordinal-0 behavior), `for_device_one_reports_against_the_real_device_count` (checked against `cudarc`'s own real device count, not a hardcoded assumption), `two_distinct_ordinals_register_into_one_runtime_without_name_collision`.
- [x] 2.4 Found and fixed a real bug running 2.3's tests on CI: `for_device_one_reports_against_the_real_device_count`'s direct `CudaContext::device_count()` call panics (not `Err`) when the CUDA library is completely absent -- wrapped in the same `catch_unwind` guard `CudaProvider` itself already uses internally, matching `provider.rs`'s own established pattern.
- [x] 2.5 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean; all 48 tests (1 intentionally ignored) pass on this development workstation's single real GPU.

## 3. `integration-tests/multi-device-cpu-cuda`

- [x] 3.1 `stage_from_provider` helper: builds a `Stage` from any already-registered, available Provider's own discovered Device, reused for GPU0 and GPU1 (the existing CPU+CUDA test's own inline construction left untouched).
- [x] 3.2 `two_real_cuda_gpus_execute_one_chained_add_across_two_real_devices_in_one_runtime`: two `CudaProvider`s (ordinal 0, ordinal 1) registered into one `Runtime`, a chained `add` across both with an explicit host round trip, verified against a hand-computed expected value; a real `DeviceSet`/`MultiDevicePlacementPlan` built from both real GPUs' own metadata afterward, same pattern as the CPU+CUDA test.
- [x] 3.3 Gracefully skips (both GPU-availability checks) on any host with fewer than two real CUDA devices -- confirmed genuinely skipping on this development workstation (one real GPU).
- [x] 3.4 Updated the module's own top-of-file doc comment: the prior "no second GPU exists anywhere in this repository's tooling" claim no longer held for CI once 1.4 landed.
- [x] 3.5 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean.

## 4. Verification

- [x] 4.1 Both `integration-tests/multi-device-cpu-cuda` tests verified genuinely executing (not skipped) on the real two-GPU `arc-gpu-magnetar` CI node via `gpu-runner-smoke.yml` (`34930015637`): no "skipping" message for either test in the real job log.
- [x] 4.2 Real CI green (`quality.yml`, after fixing the `catch_unwind` bug found by that same CI run) on GitHub's GPU-less runners, confirming graceful behavior there too.

## 5. Documentation

- [x] 5.1 `openspec validate add-real-second-gpu-cuda-provider --strict` passes.
- [x] 5.2 `SUBMODULES.md` updated: `providers/cuda` pin and compatibility-matrix row.
- [x] 5.3 README.md's top-level scope-charter status updated once archived.
