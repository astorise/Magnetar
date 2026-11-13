## Context

Module repositories are pinned in Magnetar as submodules. Those that depend on the Runtime do so by relative path (`../../magnetar-runtime`), and some depend on a sibling module the same way (the Loaders on a Format, the CUDA Provider on the CPU Provider for its conformance oracle). A module therefore builds only inside a Magnetar checkout, at its submodule path. The release workflows of the module repositories already assemble that layout: check out Magnetar, then check out the module at its path.

The Component repositories do not depend on the Runtime; they build on their own.

Magnetar's CI pins every action by commit SHA and installs the toolchain from `rust-toolchain.toml`.

The GPU runner used today, `arc-gpu-magnetar`, belongs to the Magnetar repository and runs `gpu-runner-smoke.yml` daily and on demand, in an `nvidia/cuda` container. A second GPU runner is being added for the CUDA Provider repository.

## Goals / Non-Goals

**Goals:**

- A pull request in a module repository runs that module's tests before it merges.
- A change that spans Magnetar and a module can be validated on both sides before either merges.
- One definition of the checks, maintained in one place.
- The CUDA Provider's hardware paths are tested on its own pull requests.

**Non-Goals:**

- Moving tests, replacing Magnetar's integration jobs, or changing how modules find the Runtime.

## Decisions

### D1. One reusable workflow, hosted in Magnetar

`.github/workflows/submodule-ci.yml` in Magnetar is triggered by `workflow_call`. Inputs: the module's path in Magnetar (`providers/cuda`), the sibling modules it depends on by path, extra system packages (the WGPU Provider's software Vulkan driver), whether to run Clippy and `cargo-deny`, and whether to run the GPU job, with its runner label and container image. Each module repository has a workflow of about fifteen lines that calls it on `pull_request` and on `push` to `main`.

Callers reference the workflow at Magnetar's `main`, not at a commit SHA. The workflow must match Magnetar's layout and toolchain, which move with `main`; a SHA pin would need thirteen updates for every change to it. Third-party actions inside the workflow stay pinned by SHA, as in `quality.yml`.

*Alternative: a full workflow copied into each repository.* Rejected: thirteen copies drift.

*Alternative: Magnetar triggers module tests by `repository_dispatch`.* Rejected: results would not appear as checks on the module's pull request.

### D2. Which Magnetar revision

The workflow checks out Magnetar at the branch named like the pull request's head branch if Magnetar has one, otherwise at `main`. For a push to the module's `main`, it uses Magnetar's `main`.

Sibling modules that the module depends on are initialised at the commits that Magnetar revision pins. The module under test replaces its own submodule directory with the commit being tested.

A coordinated change then works as follows: push the Magnetar branch first, then the module branch of the same name; the module's checks run against the Magnetar branch. After both are merged, Magnetar's pin is updated as today.

*Alternative: always test against Magnetar's `main`.* Rejected: a module change that needs a Runtime change could never be green before merging, which is the situation the lockfile pull requests were in.

*Alternative: test against the commit Magnetar pins for the Runtime.* There is no such commit: Magnetar pins modules, not the reverse.

### D3. What the workflow runs

From the Magnetar checkout, on the module's manifest:

1. `cargo fmt --check`
2. `cargo clippy --locked --all-targets -- -D warnings`
3. `cargo test --locked`
4. `cargo deny check`

Magnetar's integration jobs run `cargo test --locked` on every pinned module and `cargo-deny` on five of them; formatting and Clippy are new for modules. Checked before writing the workflow: formatting passes in all thirteen modules, and Clippy passes in the eleven that depend on the Runtime.

Clippy and `cargo-deny` are inputs, on by default. The two Components call the workflow with Clippy off: their package names (`magnetar-component-Qwen`, `magnetar-component-Llama`) fail the `non_snake_case` lint on the crate name. Fixing that means renaming the crates or allowing the lint in them, which is their repositories' decision and not part of this change.

Components do not depend on the Runtime and could build alone. They use the same layout as the other modules anyway (inside a Magnetar checkout, at their submodule path): one code path in the workflow, and the toolchain Magnetar pins. Magnetar's `component integration` job builds them the same way.

`--locked` is kept: the module's `Cargo.lock` is part of what is tested. When Magnetar's `main` changes the Runtime's dependency set, a module's next run fails on the stale lockfile; that is the signal to refresh it, and it now appears in the module repository instead of at the next pin update.

### D4. GPU job for the CUDA Provider

A second job in the reusable workflow, enabled by input, runs on the CUDA Provider repository's self-hosted GPU runner, in the same `nvidia/cuda` container image as `gpu-runner-smoke.yml`:

- `nvidia-smi` as a visible precondition; the job fails if no GPU is present, it does not skip.
- `cargo test --locked -- --include-ignored` for the Provider: device discovery, the required kernel set, device-resident allocation, and the Provider conformance profiles on real hardware.

It runs for pushes to `main` and for pull requests whose head branch is in the CUDA Provider repository. It does not run for pull requests from forks: a self-hosted runner executes the pull request's code, and a fork's code is not trusted with the machine. For a fork pull request the hosted job still runs, on the unavailable-device path.

Runs are serialised with a concurrency group, as in `gpu-runner-smoke.yml`, because there is one runner.

The long campaigns (long context, multi-GPU, MoE and kernel timing) and the integration test crates stay in Magnetar's daily GPU workflow: they take most of that run and need models and several modules.

The runner's label is an input of the reusable workflow, so the repository-level runner can be named without editing Magnetar.

### D5. Relation to Magnetar's jobs

Nothing is removed from `quality.yml` or `gpu-runner-smoke.yml`. The two levels answer different questions:

| Check | Where | Answers |
|---|---|---|
| Module CI (this change) | module repository, on each pull request | does this module change pass its own tests against the current Runtime? |
| Integration jobs | Magnetar, on each pull request | does the pinned combination work together? |
| Daily GPU run | Magnetar | does the pinned combination work on real hardware, including long campaigns? |

### D6. Rollout

The reusable workflow lands in Magnetar first and is exercised by one module, the CPU Provider, before the others adopt it. The GPU job is added to the CUDA Provider once its runner is registered.

## Risks / Trade-offs

- **Callers track Magnetar's `main`** → a breaking edit to the reusable workflow reaches all modules at once. Mitigation: the workflow is covered by Magnetar's own review, and the CPU Provider's run is checked after any change to it.
- **A module's checks turn red because Magnetar's `main` moved** (new Runtime dependency, stale lockfile) → expected and wanted; the fix is a lockfile refresh in the module. Documented in `SUBMODULES.md`.
- **Build time on every module pull request** → Cargo cache keyed on Magnetar's `Cargo.lock` and the module's; `extract-provider-api-crate` later reduces what Providers compile.
- **Self-hosted GPU runner on a public repository** → the GPU job is limited to same-repository branches and `main`, uses the read-only token and no secret, and runs in a container.
- **One GPU runner, queued jobs** → concurrency group; the job is the Provider's own tests only.
- **Same-name branch picked by accident** (a Magnetar branch that happens to share a module branch's name) → the run log states which Magnetar revision was used, as its first step.

## Migration Plan

1. Reusable workflow in Magnetar, documentation.
2. CPU Provider adopts it; verify a pull request there gets the four checks, against `main` and against a same-name Magnetar branch.
3. The other Providers, the Formats, the Loaders and the two Components adopt it.
4. CUDA Provider enables the GPU job once the runner is registered.

Rollback: delete the caller workflow in a module repository. Nothing in Magnetar's own gates depends on it.

## Open Questions

- The label of the CUDA Provider repository's GPU runner, and whether it runs the same container image as `arc-gpu-magnetar`.
- Should the module checks be required by branch protection in each repository? Recommended once the workflow has run cleanly for a week; it is a repository setting, outside this change.
- Should a scheduled run on each module's `main` detect staleness against Magnetar's `main` between pull requests? Not included; Magnetar's integration jobs already cover the pinned commits.
