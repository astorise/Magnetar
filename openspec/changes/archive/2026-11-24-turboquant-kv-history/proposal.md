## Why

The KV history is f32 everywhere (`append-kv-history-in-place`). On a 12 GB
RTX 3060 it bounds context length and concurrency more than the weights do:

| Model | f32 per token | 32,768 tokens |
|---|---|---|
| Qwen2.5-0.5B | 24 KiB | 768 MiB |
| Llama 3.2 1B | 64 KiB | 2 GiB |
| Mistral 7B | 256 KiB | 8 GiB |

Decode attention also reads all of it at every step.

`quantize-kv-history` proposed the framework (a KV storage precision chosen
by the Runtime, stored by the Provider) with plain block formats (`bf16`,
`q8_0`, `q4_0`). This change implements that framework with TurboQuant
(Zandieh et al., ICLR 2026) as its low-bit precision. TurboQuant:

- needs no calibration and no data;
- works online, token by token;
- reports quality neutrality at about 3.5 bits per value on long-context
  benchmarks, where per-block scalar formats lose quality at 4 bits and
  below.

It is the first of two steps. The second applies the same precisions to
DeepSeek's latent KV cache, once that cache exists (`register-deepseek-model-component`,
milestone 2).

## What Changes

- **TurboQuant semantics** (`magnetar-provider-api`, shared by every
  Provider), for each head's `d`-value row `x` (`d` a power of two):
  - the norm `‖x‖` is stored as f32;
  - the unit vector is rotated by a fixed randomized Hadamard transform
    `Π = H·D/√d` (fixed random signs `D`);
  - each rotated coordinate is replaced by the index of its nearest
    Lloyd-Max centroid for the coordinate distribution of a random unit
    vector in `R^d` (density ∝ `(1 − t²)^((d−3)/2)`), at 4 bits (`tq4`) or
    3 bits (`tq3`);
  - the stored value is `‖x‖ · Πᵀ ĉ`.

  Each store also keeps per-column offsets, the mean of the rows of the
  write that starts its history; rows are quantized as their difference
  from them and read back with them added. A component every key shares
  (a key bias, outlier channels) is then kept exactly; attention is
  unchanged by any fixed offsets.

  The codebooks are computed, not tabulated: Lloyd iterations on that
  density in f64, deterministic.
- **KV storage precision** (`magnetar-runtime`): `f32` (default), `tq4`,
  `tq3`, chosen per loaded model from `MAGNETAR_KV_PRECISION`. The choice
  applies only if the Provider advertises the TurboQuant store forms on
  `kv-write` and `attention`; otherwise it falls back to `f32` and records
  why. The store resources keep their logical `[capacity, width]` f32
  contract, so Components and the WIT graph contract do not change. A store
  is created through a new Provider call that names its precision, is
  admitted at its stored size, and the precision joins the KV compatibility
  key.
- **Reference semantics** (CPU Providers): `kv-write` into a TurboQuant
  store writes each row's dequantized values. Attention then computes, in
  f32, on exactly the values a quantized store holds. This is the reference
  every other Provider is compared with.
- **CUDA**:
  - a TurboQuant store keeps 4- or 3-bit codes and one f32 norm per row and
    head;
  - `kv-write` quantizes in the kernel that writes, at the captured step's
    advancing offset;
  - decode attention computes on the codes. Each query head is rotated once
    (`Π q`), a key's score is `q · μ_k + ‖k − μ_k‖ · Σ (Πq)_j c[idx_j]`,
    values accumulate in the rotated domain, and one inverse rotation per
    head, plus the value offsets, ends the step. No stored row is
    dequantized;
  - prefill attention dequantizes the rows it reads into a scratch buffer
    and runs the existing kernels.
- **Qualification**: on real checkpoints, against the `f32` history:
  - greedy agreement over 64-128 tokens;
  - the top-1 logit error;
  - bytes per token and the longest context that fits;
  - TPOT and TTFT at 512 and 4,096 tokens.

  `f32` stays the default until the evidence supports another.

## Capabilities

### New Capabilities

- `quantized-kv-history` (from `quantize-kv-history`, here with TurboQuant
  precisions).

### Modified Capabilities

- `cpu-provider`, `cuda-provider`: TurboQuant KV stores.

## Impact

- `magnetar-provider-api`: TurboQuant semantics and the Provider call
  creating a KV store in a precision (default: not supported).
- `magnetar-core`: the feature names.
- `magnetar-runtime`: precision policy, store creation and admission,
  compatibility key, reporting.
- `providers/cpu`, `providers/cuda`: stores and kernels.
- `integration-tests/production-loading`, GPU workflow: campaigns.

## Non-goals

- The inner-product variant (a 1-bit QJL sketch of the residual) is not
  part of this change. It is considered for 3-bit keys if `tq3` keys lose
  greedy agreement.
- Outlier-channel splits (the paper's 2.5/3.5-bit mixes), FP8, tiering and
  paged KV are out of scope.
- DeepSeek's latent cache is the next change.
