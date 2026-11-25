# latent-attention-history Specification

## Purpose
This specification defines the latent history of multi-head latent attention (the compressed K/V rows a layer stores instead of each head's keys and values), the absorbed decode attention that reads it, and the grouped matmul and strided values it relies on.
## Requirements
### Requirement: Latent KV History

A latent-attention layer SHALL keep its KV history as one store of `kv_lora_rank + qk_rope_head_dim` values per token (the normalized latent and the rotated shared key) and no value store, and its decode attention SHALL read the values from those rows.

#### Scenario: Smaller history
- **WHEN** DeepSeek-V2-Lite generates
- **THEN** each layer's history holds 576 values per token, and the generated tokens match the expanded form's within the conformance tolerance

### Requirement: Absorbed Decode Attention

A decode step over a latent history SHALL absorb each head's position-independent query into the latent space, attend over the latent history as one key head shared by every query head with the expanded form's score scale, and up-project each head's latent output, so that it computes the expanded form's attention.

#### Scenario: Same attention
- **WHEN** the tiny DeepSeek-V2 checkpoints generate on the Reference CPU
- **THEN** their logits match the `transformers` references within the existing tolerance

### Requirement: Grouped Matmul And Strided Values

`batched-matmul` SHALL multiply each of `group_count` column groups of its input by the matching row group of its weight, and `attention.value_head_stride` SHALL let V rows hold heads wider than the value, the value being each head's leading `value_head_dimension` values.

#### Scenario: V read from the keys
- **WHEN** attention takes the key history as V with `value_head_stride` equal to the key head width
- **THEN** its output equals attention with V the leading `value_head_dimension` values of each key head
