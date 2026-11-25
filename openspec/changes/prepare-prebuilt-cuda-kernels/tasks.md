Status: **Proposed (RFC lot 1).** No task is implemented yet. Requires lot 0.

## 1. Contracts

- [ ] 1.1 `magnetar-kernel`: `KernelPreparationCapability`, `KernelPreparationRequest`, `PreparedKernelHandle`, `KernelPreparationError` (incompatible target, missing symbol, invalid launch descriptor, resource limit, in use, Provider failure).
- [ ] 1.2 `magnetar-provider-api`: `Provider::kernel_preparation_api()` defaulting to `None`; `ProviderKernelPreparationApi` (`capability`, `prepare`, `release`).
- [ ] 1.3 `magnetar:cuda-launch@1`: parsing into `LaunchDescriptor` and validation against the semantic binding, the specialization and Device limits (design D2), with structured errors.

## 2. Runtime

- [ ] 2.1 After lot 0 eligibility, request preparation on the Runtime-selected Device. Register the returned advertisement under Provider authority, bound to the bundle identity, as generation 1. Record the stages prepared and ready.
- [ ] 2.2 The explicit pin `(operator, bundle identity)` from policy configuration: refused when ineligible, recorded in dispatch observations.
- [ ] 2.3 Preparation and release stay off the token path (asserted by a test that runs a decode while preparation is pending).

## 3. CUDA Provider

- [ ] 3.1 `nvidia:cubin` preparation: architecture check against the compute capability, in-memory module load, entry resolution, launch-descriptor checks against Device limits.
- [ ] 3.2 Generic launch from a descriptor on the Provider stream, with device-resident inputs and outputs.
- [ ] 3.3 Step capture references to prepared kernels; `release` returns in use while referenced; modules unloaded on release or shutdown.
- [ ] 3.4 Built-in advertisements unchanged; a revoked or failed prepared kernel leaves the built-in one selected.

## 4. Fixture and CI

- [ ] 4.1 `tools/kernel-bundle-fixture`: compile checked-in `silu_mul.cu` and `rmsnorm.cu` with NVRTC to an `sm_86` CUBIN; write a Kernel Exchange Bundle with the launch extension; sign the manifest digest with an ephemeral job key.
- [ ] 4.2 A GPU job (`gpu-runner-smoke.yml`):
  - build the fixtures;
  - configure a test trust store and permission set limited to the job key and `nvidia:cubin` on the CUDA Provider;
  - ingest, qualify, prepare, pin, run.

## 5. Tests and qualification

- [ ] 5.1 Contract tests: no capability; ineligible artifact; missing symbol; an unknown parameter kind; an index past the binding; a block over the limit; a mismatched architecture; release while captured.
- [ ] 5.2 Differential qualification on the GPU against the Reference CPU kernel, over the specialization's shapes with tails, before preparation is offered to selection.
- [ ] 5.3 Real checkpoint (Qwen2.5-0.5B, CUDA Provider, greedy):
  - built-in versus pinned external `silu-mul`, then `rmsnorm`, on the same job;
  - identical tokens;
  - TTFT, TPOT, p50/p95, peak device memory, preparation time and the first launch recorded in this file.
- [ ] 5.4 `cargo fmt`, `clippy -D warnings`, tests of both workspaces and the CUDA Provider's.

## 6. Documentation

- [ ] 6.1 `docs/architecture/kernel-artifact-sources.md`: preparation, the launch descriptor, the experiment's evidence.
- [ ] 6.2 The CUDA Provider README: precompiled kernel preparation, its limits (CUBIN only, declarative launches) and the execution permission it requires.
- [ ] 6.3 RFC acceptance criteria 4, 5 and 9 (section 13), checked against the recorded evidence.

## 7. Inherited items

- [ ] 7.1 The absorbed latent decode's TPOT at least the expanded form's after a short prompt (inherited from `cache-latent-attention-history` 5.3, closed there on 2026-11-25 with the gap accepted). Run 44195880740: 13.02 ms with the f32 latent history and 13.45 ms with the tq4 one, against 12.25 ms expanded; of the 13 ms per step, the replay takes 7.1 ms and the logits' materialization 4.9 ms. A candidate workload for the first kernel this change prepares: measure it before and after on the GPU runner.
