## Why

Qwen3.8-Flash-Next (`qwen4_exp`, 180B parameters of which about 6B are active per token) is the reference sparse hybrid model for local inference on consumer hardware. Strata already serves it on the GPU runner (2 × RTX 3060, 31 GB RAM, NVMe) at 23.3 tokens/s without drafting (run 45446491115). Magnetar cannot load it. Its graph needs recurrent state (Gated DeltaNet), sparse attention, a four-stream gated residual, a disk-resident n-gram embedding and 512-expert routing. Its weights also only fit when experts are tiered across VRAM, RAM and SSD. The recurrent-state contract already exists in the graph and WIT, but every executor refuses it, so even Qwen3.5 (the same mixer, dense) cannot run.

## What Changes

The change delivers in lots, each with its own exit gate:

1. **Recurrent state lifecycle** in the first-native Runtime: zero-initialized per sequence, read from the committed value, written to a pending value and committed or aborted atomically with KV.
2. **Qwen3.5 hybrid graph** in the Qwen Component, qualified on Qwen3.5-0.8B-Base against Transformers. Gated-delta-mixer kernels are added for cpu-native and CUDA.
3. **Qwen3.8-Flash-Next blocks**: sparse attention (QSA) with its indexer, the four-stream gated residual, n-gram PLE embedding read from disk, and 512-expert top-10 routing with a gated shared expert.
4. **Expert residency**: experts mapped from disk or held in RAM, an adaptive GPU expert cache, and misses computed on the CPU.
5. **Loaders** for the ISTA-DASLab GGUF and the Hugging Face `qwen4_exp` checkpoint.
6. **Qualification** against the Strata reference on the same runner.
7. **Drafting** with the model's MTP layer, after (6).

## Capabilities

### New Capabilities

- `recurrent-state-lifecycle`: Runtime ownership, initialization, commit and refusal rules for graph recurrent state.

### Modified Capabilities

- `qwen-model-component`: Qwen3.5 hybrid graph and `qwen4_exp` recognition and qualification (added requirements).

## Impact

- Magnetar: magnetar-graph planner, first-native Runtime (graph execution, prepared plans, KV transaction), benchmarks.
- Qwen Component, CPU and CUDA Providers, and the GGUF and Hugging Face loaders, in their own repositories, pinned after validation.
- No WIT change for lots 1–2. The WIT 1.11.0 recurrent functions are used as declared. Later lots add operators additively.
