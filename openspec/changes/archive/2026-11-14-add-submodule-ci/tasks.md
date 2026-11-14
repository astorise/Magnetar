## 1. Reusable workflow in Magnetar

- [x] 1.1 Add `.github/workflows/submodule-ci.yml` (`workflow_call`) with inputs: module path, sibling modules, extra system packages, Clippy and `cargo-deny` switches, GPU job on or off, GPU runner label, GPU container image. Permissions: `contents: read`. No secret.
- [x] 1.2 Checkout: Magnetar at the same-name branch if it exists, else `main`; sibling modules at the commits that revision pins; the module under test at its submodule path. A step prints the Magnetar branch and both commits.
- [x] 1.3 Hosted job: `cargo fmt --check`, `cargo clippy --locked --all-targets -- -D warnings`, `cargo test --locked`, `cargo deny check`, with the toolchain from Magnetar's `rust-toolchain.toml` and a Cargo cache.
- [x] 1.4 Components use the same layout and the same job, with Clippy off (design D3).
- [x] 1.5 GPU job: self-hosted runner from the input label, `nvidia/cuda` container as in `gpu-runner-smoke.yml`, `nvidia-smi` and `nvcc` precondition that fails without a GPU, `cargo test --locked -- --include-ignored`, concurrency group, condition excluding fork pull requests.
- [x] 1.6 Third-party actions pinned by commit SHA, as in `quality.yml`. Branch names and inputs reach shell steps through the environment, not through expression interpolation.

## 2. First adopter: CPU Provider

- [x] 2.1 Caller workflow in the CPU Provider repository (`pull_request`, `push` to `main`): astorise/Magnetar-provider-CPU#17, a draft that references the workflow at the Magnetar branch adding it.
- [x] 2.2 Coordinated case validated in CI: the run log names the Magnetar branch of the same name, and formatting, Clippy, the 39 tests and `cargo-deny` pass.
- [x] 2.3 Stale lockfile reproduced locally with the pre-split lockfile: `cargo test --locked` fails with "cannot update the lock file ... because --locked was passed".
- [x] 2.4 After astorise/Magnetar#169 was merged: reference switched to `@main` in astorise/Magnetar-provider-CPU#17, whose run passed with the workflow loaded from Magnetar's `main`; merged.

## 3. Other module repositories

- [x] 3.1 Providers CUDA, Metal, NPU, ROCm, TPU, WGPU: caller workflow as draft pull requests (CUDA with `providers/cpu` as sibling, WGPU with the software Vulkan driver package).
- [x] 3.2 Formats GGUF and safetensors, Loaders GGUF and Hugging Face: caller workflow as draft pull requests; each Loader checks out its Format sibling.
- [x] 3.3 Components Qwen and Llama: caller workflow as draft pull requests, Clippy off.
- [x] 3.4 First runs: all thirteen repositories pass on their current head. Two findings fixed on the way: the WGPU Provider also depends on `providers/cpu` by path (sibling added), and the two Formats locked `wasmtime` 50.0.0, affected by RUSTSEC-2026-0422, -0423 and -0424 (lockfiles updated to 50.0.1, which Magnetar and the other modules already used).
- [x] 3.5 The twelve other pull requests had been merged while still referencing the workflow at the Magnetar branch that added it. After astorise/Magnetar#169 was merged, one follow-up pull request per repository switched the reference to `@main`; all twelve passed and are merged. Magnetar's submodule pins now point at those commits.

## 4. CUDA Provider GPU job

- [x] 4.1 Runner label recorded in `SUBMODULES.md`: `arc-gpu-magnetar-provider-cuda`. The runner is online for the CUDA Provider repository and runs the `nvidia/cuda` container image with `--gpus all` (`nvidia-smi` and `nvcc` steps pass).
- [x] 4.2 GPU job enabled in the CUDA Provider's caller workflow (astorise/Magnetar-provider-CUDA#35) with that label.
- [x] 4.3 Validated on astorise/Magnetar-provider-CUDA#35 and on the push to `main` that merged it: the GPU job reports 96 passed, 0 failed, 0 ignored, including the `tests_hardware_hot_path` tests and the Provider conformance profiles.
- [x] 4.5 The GPU job's condition runs it only for non-pull-request events and for pull requests whose head repository is the module repository itself. Read from the workflow; not exercised with a real fork.

## 5. Documentation

- [x] 5.1 `SUBMODULES.md`: the two levels of checks (module CI, Magnetar integration), the same-name branch rule for coordinated changes, and what to do when a module check fails on a stale lockfile.
- [x] 5.2 `docs/quality.md` and `CONTRIBUTING.md`: how to reproduce a module's checks locally inside a Magnetar checkout.
- [x] 5.3 Each module repository's `README.md` has a "Continuous integration" section: what runs, which Magnetar revision, what a lockfile failure means, how to reproduce locally. Merged in the thirteen repositories; Magnetar's pins include those commits.
