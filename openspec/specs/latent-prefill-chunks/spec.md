# latent-prefill-chunks Specification

## Purpose
This specification defines chunked prefills for multi-head latent attention: attention over latent keys and values, and latent prefills run in bounded chunks.
## Requirements
### Requirement: Attention Over Latent Keys And Values

The Runtime SHALL accept an `attention` node with `latent_kv_width = L` and five inputs `(q, k, v, w_k, w_v)`, whose K and V rows are latent rows of `L` values and one rope key shared by every head, and SHALL compute it as the attention whose head `h` has the key `[c W_k_h | k_rope]` and the value `c W_v_h` for every key row it reads; a Kernel serves it only when it advertises `attention-latent-kv`.

#### Scenario: Reference semantics
- **WHEN** the Reference CPU runs such a node over an f32 history
- **THEN** its output equals the expanded attention over keys and values computed by `matmul` from the same rows

#### Scenario: CUDA against the Reference CPU
- **WHEN** CUDA runs such a node over the first `key_count` rows of an f32 or `tq4` history, with packed projections and queries after earlier rows
- **THEN** its output matches the Reference CPU's expansion and attention of the same history values within the attention tolerance

### Requirement: Latent Prefills In Chunks

The DeepSeek Component's prefill SHALL attend through this form over the latent history its `latent` node writes, so that the Runtime prefills a prompt longer than the chunk length in passes that read the earlier passes' rows.

#### Scenario: Tiny checkpoints in passes
- **WHEN** the tiny DeepSeek-V2 checkpoints are prefilled in passes of 4 rows on the Reference CPU
- **THEN** the model keeps the chunk length and its logits match `transformers` within 1e-3 with identical tokens
