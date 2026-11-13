## 1. Reusable workflow in Magnetar

- [x] 1.1 Add `.github/workflows/submodule-ci.yml` (`workflow_call`) with inputs: module path, sibling modules, extra system packages, Clippy and `cargo-deny` switches, GPU job on or off, GPU runner label, GPU container image. Permissions: `contents: read`. No secret.
- [x] 1.2 Checkout: Magnetar at the same-name branch if it exists, else `main`; sibling modules at the commits that revision pins; the module under test at its submodule path. A step prints the Magnetar branch and both commits.
- [x] 1.3 Hosted job: `cargo fmt --check`, `cargo clippy --locked --all-targets -- -D warnings`, `cargo test --locked`, `cargo deny check`, with the toolchain from Magnetar's `rust-toolchain.toml` and a Cargo cache.
- [x] 1.4 Components use the same layout and the same job, with Clippy off (design D3).
- [x] 1.5 GPU job: self-hosted runner from the input label, `nvidia/cuda` container as in `gpu-runner-smoke.yml`, `nvidia-smi` and `nvcc` precondition that fails without a GPU, `cargo test --locked -- --include-ignored`, concurrency group, condition excluding fork pull requests.
- [x] 1.6 Third-party actions pinned by commit SHA, as in `quality.yml`. Branch names and inputs reach shell steps through the environment, not through expression interpolation.

## 2. First adopter: CPU Provider

- [ ] 2.1 Add the caller workflow to the CPU Provider repository (`pull_request`, `push` to `main`).
- [ ] 2.2 Validate on a pull request with no same-name Magnetar branch: the four checks run against Magnetar's `main` and pass.
- [ ] 2.3 Validate the coordinated case: a Magnetar branch and a CPU Provider branch of the same name; the run log names the Magnetar branch.
- [ ] 2.4 Validate failure reporting: a deliberately failing test on a scratch branch produces a failed check; a stale lockfile produces a failure that names the lockfile. Delete the scratch branches.

## 3. Other module repositories

- [ ] 3.1 Providers CUDA, Metal, NPU, ROCm, TPU, WGPU: caller workflow (WGPU with the software Vulkan driver package).
- [ ] 3.2 Formats GGUF and safetensors, Loaders GGUF and Hugging Face: caller workflow; Loaders check out their Format sibling.
- [ ] 3.3 Components Qwen and Llama: caller workflow, Component variant.
- [ ] 3.4 Confirm each repository's first run passes on its `main`.

## 4. CUDA Provider GPU job

- [ ] 4.1 Confirm the GPU runner is registered for the CUDA Provider repository; record its label and container image in `SUBMODULES.md`.
- [ ] 4.2 Enable the GPU job in the CUDA Provider's caller workflow.
- [ ] 4.3 Validate: a pull request from a repository branch runs the Provider's tests on real hardware; the hardware-gated tests are executed, not skipped.
- [ ] 4.4 Validate: with the GPU made unavailable to the container, the job fails.
- [ ] 4.5 Confirm from the workflow condition that a fork pull request does not start the GPU job.

## 5. Documentation

- [ ] 5.1 `SUBMODULES.md`: the two levels of checks (module CI, Magnetar integration), the same-name branch rule for coordinated changes, and what to do when a module check fails on a stale lockfile.
- [ ] 5.2 `docs/quality.md` and `CONTRIBUTING.md`: how to reproduce a module's checks locally inside a Magnetar checkout.
- [ ] 5.3 Each module repository's `README.md`: a short section on its CI and on the Magnetar revision it is tested against.
