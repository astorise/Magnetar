## ADDED Requirements

### Requirement: Attention with a value head width of its own

The catalog SHALL accept `attention.value_head_dimension`, and the Reference CPU SHALL implement it: V rows of `kv_head_count * value_head_dimension` values, an output of `head_count * value_head_dimension` per row, and Q and K rows of `head_dimension` per head. A Provider SHALL receive the attribute only if its attention Kernels advertise `attention-value-head-dimension`.

#### Scenario: Narrower values
- **WHEN** two heads attend with Q/K width 3 and V width 2
- **THEN** the output has two values per head and matches a direct computation within f32 tolerance

#### Scenario: Mismatched V rows
- **WHEN** V rows do not have `kv_head_count * value_head_dimension` values
- **THEN** the Kernel refuses the invocation

### Requirement: Head-wise concatenation

The catalog SHALL accept `concat` with `axis = "heads"`, `head_count` and `second_head_count` (1 or `head_count`, defaulting to `head_count`). Such a node SHALL join `[T, head_count * a]` and `[T, second_head_count * b]` rows into `[T, head_count * (a + b)]`, each output head holding the first input's head followed by the second input's, whose only head is given to every head when `second_head_count` is 1. Without `axis`, `concat` SHALL keep joining rows. The catalog SHALL refuse an unknown `axis`.

#### Scenario: A shared rotated key
- **WHEN** a node joins `[T, 16 * 128]` keys with a `[T, 64]` rotated key, `second_head_count = 1`
- **THEN** every one of the 16 output heads ends with the same 64 rotated values

#### Scenario: Ambiguous widths are refused
- **WHEN** the second input holds neither one nor `head_count` heads, or the output width differs from `head_count * (a + b)`
- **THEN** the catalog refuses the node

### Requirement: YaRN RoPE frequency scaling

The catalog SHALL accept `rope.frequency_scaling = "yarn"` with `frequency_scaling_factor`, `original_context_length`, `beta_fast`, `beta_slow`, `attention_factor` and `truncate`, and every Kernel implementing it SHALL use `RopeYarnFrequencyScaling` for the correction range and each pair's frequency. `cos` and `sin` SHALL be multiplied by `attention_factor`. A missing parameter SHALL fail the invocation.

#### Scenario: DeepSeek-V2-Lite's frequencies
- **WHEN** the Reference CPU rotates 64 dimensions with base 1e4, factor 40, 4096 original positions, beta 32/1 and a truncated range
- **THEN** each pair's frequency matches transformers 5.23 within 2e-6 relative, and the correction range is pairs 10 to 23
