## 1. Measurement

- [x] 1.1 Resident sizes of E2B's per-layer and token tables per artifact and Provider, from the tensor headers and storage rules.
- [ ] 1.2 Device time and bytes written of a packed lookup, rows against the whole table (`packed_embedding_lookup_times`, GPU workflow).

## 2. CUDA Provider

- [x] 2.1 `packed_element`: optional row ids in every dequantization kernel; an invalid id sets the lookup's flag.
- [x] 2.2 `embedding_lookup` reads the looked-up rows of an untransposed packed table in any format; a transposed one is still dequantized whole.

## 3. Tests

- [x] 3.1 Every packed format: the looked-up rows equal the dequantized table's; invalid ids fail; the flag resets.
- [ ] 3.2 The tests and the diagnostic on the GPU runner.

## 4. Validation

- [x] 4.1 `cargo fmt`, `clippy -D warnings`, tests and the NVRTC compile of `kernels.cu` in the CUDA Provider; Magnetar pins it.
