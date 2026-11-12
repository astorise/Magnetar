## ADDED Requirements

### Requirement: KV storage precision selection

The Runtime SHALL select a KV storage precision for K and for V per Model Instance from the instance setting, the request override and the environment control, restricted to precisions the selected Provider advertises for `kv-write` and for every attention variant in the plan, falling back toward `f32` and recording the reason when a requested precision is not advertised. `f32` SHALL remain the default until qualification promotes another.

#### Scenario: Requested precision advertised

- **WHEN** a request asks for `q8_0` KV and the CUDA Provider advertises `q8_0` for `kv-write` and every attention variant of the plan
- **THEN** the instance's K and V stores use `q8_0`

#### Scenario: Requested precision not advertised

- **WHEN** a request asks for `q8_0` KV on a Provider that only advertises `f32`
- **THEN** the stores use `f32` and the outcome reports the fallback

### Requirement: Logical f32 contract

KV stores in a non-`f32` precision SHALL keep their logical f32 graph descriptors: `kv-write` SHALL take f32 rows and attention SHALL compute as if reading f32 values equal to the stored values dequantized. No WIT graph contract or Component change SHALL be required to use a KV precision.

#### Scenario: Unchanged Component

- **WHEN** an existing Component's graph runs with `bf16` KV
- **THEN** it links and runs unchanged, and only the Provider's stored representation differs

### Requirement: Admission at stored size

The Memory Manager SHALL admit each KV store at the size the Provider reports for its stored precision, and that size SHALL equal the Provider's actual allocation.

#### Scenario: Longer context admitted

- **WHEN** a Mistral 7B `Q4_K_M` instance requests a context that does not fit a 12 GB Device with `f32` KV but fits with `q8_0`
- **THEN** admission succeeds with `q8_0` and fails with `f32`

### Requirement: Quantized KV write and read

A Provider advertising a KV precision SHALL quantize rows on `kv-write` (32-value blocks within a row with one scale, round to nearest even; `bf16` round to nearest even) at the write offset, including within a captured step without allocating, and SHALL dequantize stored values in every advertised attention variant before softmax and accumulation in f32. Its attention output SHALL equal the Reference CPU's attention over the dequantized history within summation-order tolerance.

#### Scenario: Kernel conformance

- **WHEN** K and V rows are written in `q8_0` and attention runs over 1 to 4,096 keys
- **THEN** the result matches the Reference CPU on the dequantized history, and the downloaded store equals the dequantized rows exactly

### Requirement: Precision in KV compatibility

KV precision (K and V) SHALL be part of KV layout metadata and of the KV compatibility key, and a history SHALL NOT be read in a precision other than the one it was written in.

#### Scenario: Prefix reuse across precisions

- **WHEN** a cached prefix written in `f32` is offered to a request using `q8_0` KV
- **THEN** reuse is rejected as incompatible and the prefix is re-prefilled

### Requirement: Quantized KV qualification

A KV precision SHALL NOT become a default, or be reported production-validated for a model profile, until long-context campaigns on that profile's real checkpoint and real Device record logit error, greedy token agreement and a perplexity proxy within declared thresholds against `f32`, together with VRAM per token, the longest admitted context, TPOT and TTFT.

#### Scenario: Evidence is incomplete

- **WHEN** only kernel conformance or a short-context run is available
- **THEN** the precision stays opt-in and the support matrix does not claim it validated
