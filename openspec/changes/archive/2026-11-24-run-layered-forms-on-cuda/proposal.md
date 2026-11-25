## Why

Since `dispatch-layered-model-forms`, the Runtime sends an optional operator form only to a Provider whose every Kernel for the operator declares it. The CUDA Provider declared none of them, so a layered model's graph (issue #187) was refused on CUDA. That refusal was correct, because the CUDA Kernels would have ignored the attributes.

## What changes

The CUDA Provider (`providers/cuda`, astorise/Magnetar-provider-CUDA#42) implements and declares:

| Form | Feature |
|---|---|
| attention score scale | `attention-scale` |
| scaled embedding | `embedding-scale` |
| per-head RMSNorm | `rmsnorm-per-head` |
| weightless RMSNorm | `rmsnorm-without-weight` |
| partial RoPE | `partial-rotary` |
| GELU-gated experts | `moe-experts-activation` |
| per-expert output scales | `per-expert-scale` |

It also adds the softcap and scale `activation` kinds, `gelu` (tanh) and the `[1]`-broadcast `mul`.

Two things stay the same:
- An attention Kernel given no scale computes exactly as before.
- The scaled embedding is written in place, so a step capture holds no freed intermediate.

## Impact

- `providers/cuda`: Kernels, launchers, executor, advertisements, tests.
- Magnetar pins the Provider.
