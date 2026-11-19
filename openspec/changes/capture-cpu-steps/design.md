## Context

The Runtime drives capture through `ProviderExecutionApi`:

1. `begin_step_capture`, with the advancing attribute names;
2. the step's per-node dispatch;
3. `end_step_capture`;
4. later, `replay_step(id, offset, inputs)`.

A CUDA capture records Device addresses, so it goes stale when a resource is reallocated.

## Decisions

### Record invocations, run them again

The CPU Provider keeps every resource in host storage under its identity. A recorded `KernelInvocation` therefore reads and writes, at replay, exactly the resources a per-node run would. A replay can never be stale: a missing resource fails it, and the Runtime runs the step per node. This is the test `ReplayingExecutor` the Runtime's own capture tests already use, as a Provider feature.

The invocations run through the wrapped executor's `run_invocation`, which uses the same native kernels as a per-node dispatch. Replay skips validation and observations, which the per-node dispatch already did for these invocations.

### Host writes fail a capture

A replay repeats Kernels, not host writes. Any host write, copy or store reset during a capture would be missing from a replay, so it records a failure, and `end_step_capture` returns it. The Runtime writes a captured step's inputs before the capture begins and through `replay_step` afterwards, and its capture tests already require this behavior of a capturing Provider.

### Offsets

Every advancing integer attribute is increased by the offset with checked arithmetic, before anything runs. Bounds are checked by the Kernels themselves: for example, a `kv-write` past its store fails the replay.

## Risks / Trade-offs

- **Memory.** Captured steps keep their intermediates resident, as on CUDA. On the CPU that is host memory, under the same budgets.
- **Host work that remains.** A replay still runs the Reference CPU code around each native Kernel (input reads, output stores), but no Runtime dispatch.
