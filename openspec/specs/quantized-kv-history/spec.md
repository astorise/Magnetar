# quantized-kv-history Specification

## Purpose
This specification defines TurboQuant KV history: its storage precision, store offsets, precision resolution and admission, and its place in KV cache compatibility.
## Requirements
### Requirement: TurboQuant KV Storage Precision

A KV store SHALL be creatable in the `tq4` or `tq3` precision: each head's row, less the store's offsets, stored as its f32 norm and the 4- or 3-bit indices of the nearest Lloyd-Max centroids of its coordinates after a fixed randomized Hadamard rotation, as `magnetar-provider-api` defines them. A store SHALL keep its logical `[capacity, width]` f32 descriptor, and attention SHALL compute as on the dequantized values the store holds.

#### Scenario: Quantized history keeps the graph contract
- **WHEN** a model runs with `MAGNETAR_KV_PRECISION=tq4` on a Provider that advertises TurboQuant stores
- **THEN** its Component and graphs are unchanged and every `kv-write` stores rows that read back as their TurboQuant dequantization

### Requirement: TurboQuant Store Offsets

A TurboQuant store SHALL keep one f32 offset per column of its rows, set by the write at row 0 to the mean of the rows that write computes and zero before it, and SHALL hold every row as its offsets plus the TurboQuant dequantization of its difference from them.

#### Scenario: A shared component is kept exactly
- **WHEN** a prefill writes rows that share a large per-column component into a `tq4` store
- **THEN** the store's offsets are those rows' mean, and a row equal to the offsets reads back exactly

### Requirement: KV Precision Resolution And Admission

The Runtime SHALL resolve the KV precision of each role (K, V) once per loaded model from `MAGNETAR_KV_PRECISION`, `MAGNETAR_KV_PRECISION_K` and `MAGNETAR_KV_PRECISION_V`, SHALL fall back to `f32` and record why when the Provider does not advertise the precision on `kv-write` and every active attention Kernel or a head width is not a power of two, and SHALL admit each store at its stored size.

#### Scenario: Unsupported Provider
- **WHEN** `tq4` is requested on a Provider that does not advertise TurboQuant stores
- **THEN** the stores are f32 and the fallback is recorded

### Requirement: KV Precision In Compatibility

A KV history's layout metadata SHALL record its precisions, and a history SHALL only be continued, reused or restored with the same precisions.

#### Scenario: Mismatched precision
- **WHEN** a history written in `tq4` is offered to an execution resolved to `f32`
- **THEN** it is refused as incompatible
