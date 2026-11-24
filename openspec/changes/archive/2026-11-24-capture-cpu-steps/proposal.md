## Why

After `speed-up-dense-cpu-matmuls`, a local profile of a Qwen2.5-0.5B decode step on the optimized CPU Provider (4 threads) attributed about 12 ms of a 58 ms step to the Runtime's per-node dispatch of about 315 nodes. That dispatch covers plan resolution, admission, observations and input resolution. The CUDA Provider avoids this cost through step capture and replay (`capture-and-replay-cuda-graphs`). The CPU Provider did not support it, and so had no bucketed prefill replay either (`replay-bucketed-prefill`), and no pooled decode replay (`reuse-decode-captures-across-requests`).

## What Changes

- **Capture.** The optimized CPU Provider supports the step capture contract. A capture records each Kernel invocation the Runtime dispatches while the step runs. A host write during the capture fails it.
- **Replay.** A replay writes the step's inputs, then runs the recorded invocations again directly. Each advancing attribute is increased by the replay offset. Every offset is applied before anything runs, so an offset that does not fit runs nothing. A Kernel failure fails the replay, and the Runtime then runs the step per node.
- **Greedy selection.** The Provider selects the greedy token by reading the logits row in place (`keep-sampling-on-device`).
- **Control.** `MAGNETAR_CPU_STEP_CAPTURE=0` declines capture.

## Capabilities

### New Capabilities

- `cpu-step-capture`: step capture, replay and greedy selection on the optimized CPU Provider.

### Modified Capabilities

None. The Runtime's capture contract is unchanged.

## Impact

- `providers/cpu` (`native::capture`, `native::mod`).
- No Runtime, WIT or Component change.
