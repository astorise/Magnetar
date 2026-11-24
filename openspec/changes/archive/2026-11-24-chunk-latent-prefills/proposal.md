## Why

A long prompt is prefilled in passes of one graph of the chunk length
(`chunk-long-prefills`) only when every attention reads its K and V from
the KV history, so a later pass sees the earlier passes' rows. Multi-head
latent attention's prefill attends with keys and values expanded from its
own rows (`cache-latent-attention-history`) while the history holds the
latent rows: it declines chunking and holds every prompt row's activations
at once. DeepSeek-V2-Lite-Chat Q8_0 across two 12 GB GPUs then prefills
16,384 tokens with a `tq4` latent history but not with an `f32` one (run
44389385779: out of the second GPU's memory at layer 25,
`bound-one-pass-prefill-memory`), and no history precision reaches 32,768
tokens.

Adding a graph export to the Component contract's world would rebuild every
Component, and the later passes would still need to read the latent
history. The absorbed form reads it, but costs 1,088 multiply-adds per
query, key and head against the expanded form's 320.

## What Changes

- **Latent K and V in `attention`.** An `attention` node with
  `latent_kv_width = L` reads K and V as latent rows `[c | k_rope]` (`L`
  values, then one rope key shared by every head) and takes two more inputs,
  `w_k [L, head_count * (head_dimension - rope)]` and
  `w_v [L, head_count * value_head_dimension]`: head `h`'s key is
  `[c W_k_h | k_rope]` and its value `c W_v_h`. Its scores and values are
  the expanded attention's over every key row it reads, the earlier passes'
  included. Kernel feature `attention-latent-kv`.
- **Providers.** The Reference CPU, the CPU Providers and CUDA expand the
  key rows they read (`key_count`), from a TurboQuant store's values when
  the history is one, then attend as the expanded form does.
- **The DeepSeek Component's prefill** attends through this form over the
  latent history its `latent` node writes, instead of keys and values
  computed beside it. Its prefill then continues a history, so long prompts
  run in chunks; the decode step keeps the absorbed form.

## Capabilities

### New Capabilities

- `latent-prefill-chunks`: latent attention that reads its history in the
  expanded form, so latent prefills run in chunks.

### Modified Capabilities

None.

## Impact

- `magnetar-core`: the `latent_kv_width` attribute and the
  `attention-latent-kv` feature.
- `magnetar-runtime`: dispatch of five-input attention; Reference CPU.
- `magnetar-provider-api`: `latent_kv_heads` semantics.
- `providers/cpu`, `providers/cuda`: the form and its advertisement.
- `components/deepseek`: the prefill graph; its fixture Component and digest.
