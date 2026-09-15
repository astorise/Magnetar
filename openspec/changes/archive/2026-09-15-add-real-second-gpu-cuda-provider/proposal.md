## Why

`add-multi-device-cpu-cuda-execution-proof` closed a real CPU+CUDA foundational proof but explicitly deferred real multi-GPU work, stated as unverifiable: "this repository has only one real GPU". That premise turned out to be wrong for the CI environment specifically -- the user identified that `arc-gpu-magnetar`'s host node has two real GPUs, and investigation found the real cause: its per-job Kubernetes pod spec (`arc-magnetar-gpu-job-hook` ConfigMap, in the separate `talos` infrastructure repository) requested only `nvidia.com/gpu: 1`, while sibling job hooks on the same cluster already requested 2. Raising that request and verifying it against the live cluster (`kubectl describe node` confirmed `nvidia.com/gpu: 2` capacity) unlocked genuine two-real-GPU CI verification for the first time. This change uses it.

## What Changes

- `providers/cuda`: `CudaProvider::for_device(ordinal, provider_name)` binds to a specific real GPU ordinal under a caller-chosen Provider name, instead of always ordinal 0 under the fixed `CUDA_PROVIDER_NAME` constant `new()` uses. Required, not just convenient: `Runtime`'s `ProviderLoader` rejects a second `register_provider` call under an already-registered name outright, so two `CudaProvider`s bound to two different real GPUs cannot coexist in one `Runtime` unless each carries its own distinct name. `new()` remains exactly `for_device(0, CUDA_PROVIDER_NAME)` -- verified unchanged by a new test. The Provider name is threaded through `device::cuda_device_descriptor`, `advertisements::cuda_kernel_advertisements`, and `CudaExecutor` (`provider_binding()`, execution-id generation) -- everywhere `CUDA_PROVIDER_NAME` was previously read as a constant.
- `integration-tests/multi-device-cpu-cuda`: a new test, `two_real_cuda_gpus_execute_one_chained_add_across_two_real_devices_in_one_runtime`, registers two `CudaProvider`s (ordinal 0 and ordinal 1, distinct names) into one `Runtime` and dispatches a chained `add` across both real GPUs with an explicit host round trip between them -- the real two-GPU counterpart to the existing CPU+CUDA proof, reusing its `Stage`/`dispatch_add` helpers via a new `stage_from_provider` factory.
- Infrastructure (outside this repository, in `C:\Users\astor\Git\talos`): `arc-magnetar-gpu-job-hook.yaml`'s `nvidia.com/gpu` resource request/limit raised from 1 to 2, applied directly to the live Talos/Kubernetes cluster and re-verified via a real `gpu-runner-smoke.yml` dispatch showing both GPUs visible (`nvidia-smi` listing GPU 0 and GPU 1, both real RTX 3060s) before this change's own code was written.
- **BREAKING**: none. `CudaProvider::new()` is behaviorally identical to before (verified by a new test); `for_device` is new, additive API surface.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `cuda-provider`: gains a requirement for binding to a specific real GPU ordinal under a distinct Provider name.
- `multi-device-placement`: gains a requirement for genuine two-real-GPU concurrent registration and dispatch, extending the CPU+CUDA proof `add-multi-device-cpu-cuda-execution-proof` already added.

## Impact

- `providers/cuda/src/{provider,device,advertisements,executor,tests}.rs`: `for_device` constructor and provider-name threading.
- `integration-tests/multi-device-cpu-cuda/src/tests_multi_device_cpu_cuda.rs`: new test, new `stage_from_provider` helper, updated module doc comment (the prior "no second GPU exists" claim no longer held for CI).
- `SUBMODULES.md`: `providers/cuda` pin and compatibility-matrix row updated.
- Outside this repository: `talos/arc-magnetar-gpu-job-hook.yaml` (applied to the live cluster, not just committed to a repo this change tracks).
