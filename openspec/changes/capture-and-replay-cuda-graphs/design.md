## Context

This draft is derived from roadmap proposal 1.0 (2026-10-29), whose reference commit is `09976e7792e610a04aa2b8af6c6991ba7bcbc225`. Repository inspection for this proposal used `acbb95cab732f01ace7dbc4203ebf698cd7a5fa6`. Canonical specs and implemented paths at the latter revision take precedence over historical roadmap wording. This is proposed work, not a stable new API or measured result.

After `append-kv-history-in-place`, a first-native decode step of Qwen2.5-0.5B on an RTX 3060 costs 13.4 ms (GPU CI run 41611795148). The step's phase attribution shows that it is bound by host work, not by the Device:

| Phase (ms per step) | |
|---|---|
| `kernel_execute` (CUDA Provider, 315 launches) | 3.59 |
| `node_dispatch` (Runtime, per node) | 3.13 |
| `output_materialize` (waiting for the Device, then the logits download) | 1.76 |
| `kv_append` (48 `kv-write` dispatches) | 1.30 |
| `kernel_resolve`, `node_input_resolve`, `node_output_bind` | 1.52 |
| `sampling`, `step_observation`, other | 2.0 |

The weights the step reads (about 494 M f32 parameters, 1.98 GB) bound the Device time from below at about 5.5 ms on a 360 GB/s card. About 9.5 ms of the step is per-node host work, by the Runtime and the Provider, that is identical from one decode step to the next.

## Goals / Non-Goals

Deliver the explicitly enumerated C04 outcomes through CUDA Provider; Runtime plan and memory contracts, preserving existing trust, resolution, lifetime and portable Component contracts. Reuse the baseline contracts linked from proposal.md. Broader roadmap work is split into independent changes; adopting an inspiration project or copying its implementation is not a goal.

Non-goals of the initial profile: prefill capture, multi-device segment graphs, concurrent requests sharing one capture, and half precision (B02).

## Decisions

### Initial supported profile (task 1.1)

- A first-native whole-model decode step, one token, F32, one Device.
- In-place KV history (`append-kv-history-in-place`) active for the request. Its stores do not move during the request, so a step's shapes and resources are the same from one step to the next.
- A Provider that reports step capture support. In this change: the CUDA Provider, with CUDA graphs. The Runtime's tests use a test Provider that keeps values resident and replays recorded invocations, so the contract is tested without a GPU. The CPU Providers report host values, which the Runtime copies into fresh input resources per node, so their steps are not replayable as recorded invocations.
- Enabled by default where supported, with a disable control (`ProductionLoadedModel::set_step_replay(false)`), which is also the qualification's before/after switch.

Every other step runs the existing per-node path, unchanged.

### Step capture contract (task 1.3)

The contract lives on `ProviderExecutionApi`, in Rust. No WIT, Component or Operator change is needed: Components never see it, and every captured invocation is an ordinary prepared Kernel invocation the Runtime already validated.

- `supports_step_capture()`: `false` by default.
- `begin_step_capture(request)`: the Provider starts recording every Kernel invocation and resource access until the capture ends. `request.advancing_attributes` names the integer attributes that advance by one per step: `position_offset` (`rope`), `key_count` (`attention`) and `row_offset` (`kv-write`). A captured invocation carrying one of them must be replayable with that attribute increased by the replay offset, or the Provider fails the capture.
- `end_step_capture()`: returns an opaque `StepCaptureId`. When it returns, the captured step's results are available, as after an uncaptured step: a Provider that only records during capture (CUDA) launches the captured work once here.
- `abort_step_capture()`: discards a capture in progress. Its step's work may not have run.
- `replay_step(id, offset, inputs)`: writes each host input into the resource it was captured with, then repeats the captured work with every advancing attribute increased by `offset`. It fails with `StepReplayError::Invalidated` when any resource the capture touched no longer has the storage it had at capture time, or when the offset leaves a captured bound (a KV store's capacity). Nothing runs in that case.
- `release_step_capture(id)`: frees the capture.

A capture never owns the resources it references: the Runtime keeps owning them through the MemoryManager, unchanged. Validation at every replay is what makes a stale reference impossible: a capture is only ever replayed against the exact allocations it recorded.

### Runtime: record once, replay the rest

The engine keeps one recorded step per request:
1. The first decode step runs the per-node path. It allocates every decode-shaped output once, so the second step finds every output's previous buffer already the right size and allocates nothing.
2. The second decode step runs the per-node path inside a capture. The Runtime does all of its usual per-node work: plan revalidation, MemoryManager admission, observations. It records what replay needs: the logits resource and shape, the KV state with its row count, the step's position and the dispatch result.
3. Later decode steps replay it. The Runtime writes the new token, calls `replay_step` with `offset = position - captured position`, sets each layer's KV row count to its captured value plus the offset, and downloads the logits. No node is dispatched by the Runtime.

The recorded step belongs to the request's KV cache, the graph fingerprint and the Provider. It is released with the request (`release_request_state`).

When a capture or a replay fails, the step runs the per-node path instead, so a request never fails because of capture:
- A failed capture is aborted, the request stops trying to capture, and the step runs again uncaptured. A step is idempotent: its KV rows land beyond the committed length, and every intermediate and its admission is replaced by the run that follows.
- An invalidated replay drops the recording. The per-node path runs the step, and the next step may capture again.

Discarding a step or a commit failure needs nothing new: a replayed step writes the same rows beyond the committed length as a per-node one.

### CUDA Provider

- **One stream of its own.** CUDA cannot capture the legacy default stream. The Provider moves all its work to one stream it creates, so its single-stream ordering, which the disabled event tracking relies on, is unchanged.
- **The step offset lives on the Device.** The captured `rope`, `attention` and `kv-write` launches read one device scalar, the step offset, and add it to their captured attribute. A replay writes the scalar, then launches the graph. No captured parameter is patched.
- **Capture-time bounds.** Captured block attention sizes its shared memory for the store's full capacity rather than the step's key count. Captured `kv-write` becomes a copy Kernel, because a memcpy's destination offset is fixed in a graph. Both bounds are checked again at every replay against the offset.
- **Nothing allocates or synchronizes during capture.** Any device allocation, host transfer or synchronization while capturing fails the capture. The embedding lookup checks token ids on the host at replay, from the replay input, instead of downloading its device flag.
- **Validation.** For every resource the captured work reads or writes, the capture records its device address and length. A replay first checks that each resource still has exactly that buffer.

## Risks / Trade-offs

- Address reuse, stream changes and Device loss invalidate capture generations before replay.
- A replay skips per-node observations and per-node evidence; each replayed step records one replay observation instead. The causal per-node chain is still produced by the first two decode steps and by every per-node step.
- A performance win can be below measurement noise or transfer cost; stop promotion until evidence meets the declared objective and uncertainty threshold.
- Wider hardware coverage is constrained by actual machines; record missing hardware as skipped, not passed, and deliver at most a contract or POC for that target.

## Migration and rollback

Land compatibility scaffolding and baseline tests before optimized execution. Keep new paths selectable and disableable by profile. Drain or fence old generations at safe boundaries; in-flight references remain alive. Roll back only to a still-trusted, resource-compatible qualified generation. If no safe fallback exists, return a structured error. Document any public schema migration before implementation and preserve supported old readers/linkers during the declared compatibility window.

## Open questions

Select exact hardware/checkpoint, numerical and downstream-quality thresholds, resource budgets and SLOs per qualification profile before promoting it. The roadmap's reference CPU/CUDA and Qwen checkpoint are the initial comparison, not evidence that other targets are available. Any Tachyon-facing interface additionally needs its independent repository audit and owner agreement. These decisions gate implementation promotion, not creation of this proposal.
