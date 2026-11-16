## 1. Model and Runtime

- [x] 1.1 `ModelDType::Q4`, `ModelQuantizationFormat::GgufQ4`, `PackedWeightFormat::Q4_0`, `Q4_0_VALUES`, `dequantize_q4_0`.
- [x] 1.2 Storage size, dequantization and packed-format mapping in Model Loading.

## 2. Formats and loader

- [x] 2.1 `formats/gguf` reads GGML type 2; `formats/safetensors` treats the dtype as block-quantized.
- [x] 2.2 `loaders/gguf` keeps `Q4_0` projections packed with transposed storage.

## 3. Providers

- [x] 3.1 Optimized CPU Provider: the IQ4_NL paths take the `Q4_0` table.
- [x] 3.2 CUDA: IQ4 format 2, `__vsub4` decoding; upload, dequantize, decode and gated matmuls.

## 4. Tests

- [x] 4.1 `dequantize_q4_0` against `gguf-py` bit for bit.
- [x] 4.2 GGUF round trip, loader packed-projection test, CPU packed tests, CUDA i-quant conformance.
- [x] 4.3 The CUDA conformance tests pass on the GPU runner (two RTX 3060, GPU workflow run 248, `kernels` suite).
