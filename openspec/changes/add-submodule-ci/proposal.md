## Why

The Component, Provider, Format and Loader repositories carry their own tests (96 in the CUDA Provider, 70 in the Hugging Face Loader, 40 in the CPU Provider, 38 in the GGUF Loader, 19 in each Format), and none of them runs those tests in its own CI. Their only workflow is release publication, triggered by tags.

Their tests run in one place: Magnetar's `quality.yml`, at the commit Magnetar pins (`component integration`, `format integration`, `provider integration`, `submodule integration`), and for the CUDA Provider on real hardware in Magnetar's `gpu-runner-smoke.yml`, daily or on demand.

So a pull request in one of these repositories gets no test signal before it merges. That happened during `layer-and-split-runtime-crate`: eleven lockfile pull requests were merged with no check run on any of them. A regression introduced in a Provider repository is found only when Magnetar bumps its pin, away from the change that caused it.

This belongs in Magnetar because Magnetar defines what "compatible" means for these modules: `SUBMODULES.md` states that compatibility is verified by CI at the pinned commit. This change keeps that rule and adds an earlier check, in the repository where the change is made.

## What Changes

- Add a reusable workflow in Magnetar, `submodule-ci.yml`, that builds and tests one module repository inside a Magnetar checkout: formatting, Clippy, `cargo test --locked`, and `cargo-deny` on the module's own dependency graph.
- Each module repository calls it from a small workflow of its own, on pull requests and on pushes to `main`.
- The module is tested against Magnetar's branch of the same name as the pull request's branch when that branch exists, otherwise against Magnetar's `main`. A change that spans Magnetar and a module can then be green on both sides before either merges.
- The CUDA Provider repository adds a GPU job on a self-hosted GPU runner: the Provider's tests including the hardware-gated ones, on real hardware, for pull requests from branches of the repository itself and for pushes to `main`.
- Magnetar's integration jobs stay as they are: they verify the pinned combination. The cross-repository integration test crates stay in Magnetar.

## Capabilities

### New Capabilities

- `submodule-ci`: what each module repository's own CI runs, against which Magnetar revision, on which runners, and how it relates to Magnetar's integration jobs.

### Modified Capabilities

None. `quality` keeps its requirements; Magnetar's integration jobs are unchanged.

## Impact

- **Magnetar**: one new workflow file (`workflow_call`), documentation in `SUBMODULES.md` and `docs/quality.md`. No Rust code.
- **Thirteen module repositories** (Components Qwen and Llama; Providers CPU, CUDA, Metal, NPU, ROCm, TPU, WGPU; Formats GGUF and safetensors; Loaders GGUF and Hugging Face): one workflow file each. The five template Component repositories get the same file when they gain content.
- **CUDA Provider repository**: a self-hosted GPU runner registered for that repository, and a GPU job that uses it.
- **Owning layer**: none of Component, Runtime, Provider or Device semantics changes. No WIT change, no Capability change, no Resource Affinity change.
- **External access**: the workflows read the public Magnetar repository and the module's own repository with the default read-only token. They use no secret. The GPU job runs code from the CUDA Provider repository on a self-hosted machine; it does not run for pull requests from forks.
- **Cost**: every module pull request compiles the Runtime crates it depends on. A Cargo cache keyed on Magnetar's lockfile limits that.

## Non-goals

- Moving tests between repositories. Module tests are already in the module repositories; the integration test crates need the Runtime and several modules at once and stay in Magnetar.
- Removing or weakening Magnetar's integration jobs or the daily GPU run.
- Running long hardware campaigns (long context, multi-GPU, MoE timing) on every CUDA Provider pull request. They stay in Magnetar's GPU workflow.
- Making module repositories buildable without a Magnetar checkout. The path dependency on the Runtime stays; `extract-provider-api-crate` narrows what is compiled, not how it is found.
- Branch protection settings of the module repositories. This change provides the checks; requiring them is a repository setting.
