## Why

DeepSeek's Multi-head Latent Attention (MLA) projects each token to one
latent `c_kv` (`kv_lora_rank` values, 512 in V2-Lite and V3) and one shared
rotated key `k_pe` (`qk_rope_head_dim`, 64). The model's keys and values are
up-projections of that latent. Today the DeepSeek Component expands them
(`express-latent-attention-forms`) and the KV history holds each head's key
and value:

| DeepSeek-V2-Lite (27 layers, 16 heads) | f32 per token | 32,768 tokens |
|---|---|---|
| Expanded K and V (`16 × 192 + 16 × 128`) | 540 KiB | 16.9 GiB |
| Latent `[c_kv | k_pe]` (`512 + 64`) | 60.8 KiB | 1.9 GiB |

The latent history is 8.9 times smaller, and decode attention reads all of
it at every step. It is also the cache TurboQuant should apply to next
(`turboquant-kv-history`, second step).

## What Changes

- **Absorbed decode attention.** A query head's score against a key is
  `q_nope · (W_UK c_kv) + q_pe · k_pe = (W_UKᵀ q_nope) · c_kv + q_pe · k_pe`,
  and its output is `W_UV (Σ p c_kv)`. A decode step therefore:
  - absorbs each head's `q_nope` into the latent space with that head's
    `W_UKᵀ` (`batched-matmul`);
  - attends with `[W_UKᵀ q_nope | q_pe]` (`latent + rope` wide) over the
    latent history as one key head shared by every query head, the values
    being the first `latent` values of each key row;
  - up-projects each head's latent output with its `W_UV`
    (`batched-matmul`).
- **Prefill stays expanded.** Its attention runs on the expanded keys and
  values of its own rows, which costs fewer operations than attending in the
  latent width. It also writes the latent rows to the history.
- **Operators** (`magnetar-core`, Reference CPU, CPU and CUDA Providers):
  - `batched-matmul` (in the catalog, not implemented): `a [T, G·K]` times
    one `[K, N]` block of `b` per group `g` (or an `[N, K]` block read
    transposed, `transpose_b`), giving `[T, G·N]`;
  - `attention.value_head_stride`: the value of a KV head is the first
    `value_head_dimension` values of a `value_head_stride`-wide head of the
    V input, so V may be the key history itself.
- **Weights.** The loaders also emit each latent layer's
  `self_attn.k_absorb` (`[heads · qk_nope, latent]`) and
  `self_attn.v_absorb` (`[heads · v_head, latent]`): the latent
  up-projection's key and value rows as stored, in every form they read
  it.
- **History geometry** (`magnetar-runtime`): a latent layer keeps one store
  of `latent + rope` wide rows and no value store.
- **The DeepSeek Component** emits both graphs: the expanded prefill writing
  the latent rows, the absorbed decode reading them.

## Capabilities

### New Capabilities

- `latent-attention-history`: the latent KV history and absorbed decode.

### Modified Capabilities

- `cpu-provider`, `cuda-provider`: `batched-matmul`, `value_head_stride`.

## Impact

- `magnetar-core`, `magnetar-provider-api`, `magnetar-runtime`: operator
  semantics, Reference CPU, KV geometry.
- `providers/cpu`, `providers/cuda`: kernels.
- `loaders/huggingface`, `loaders/gguf`: the absorbed weights.
- `components/deepseek`: graphs, rebuilt and repinned.
- Acceptance: the tiny DeepSeek-V2 checkpoints against their `transformers`
  references, DeepSeek-V2-Lite-Chat on CUDA.

## Non-goals

- TurboQuant on the latent history is the next step, once this one holds.
- An absorbed prefill (for a prefill that continues a history) and
  FlashMLA-style kernels are out of scope.
