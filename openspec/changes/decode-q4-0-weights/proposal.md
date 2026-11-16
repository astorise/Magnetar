## Why

Google's QAT Gemma 4 GGUF files (`google/gemma-4-*-it-qat-q4_0-gguf`, issue #187) store their attention and MLP weights as `Q4_0` (GGML type 2). `formats/gguf` refused that type, so these files could not load at all. The real E2B and 26B-A4B headers hold `Q4_0`, `Q6_K`, `F16` and `F32` tensors, and `Q4_0` was the only one of these types not supported.

## What changes

`Q4_0` (32 values in 18 bytes: an `f16` scale `d` and sixteen bytes of 4-bit quants `q`, each value `d * (q - 8)`) becomes a supported block format end to end:
- **Model and quantization:** `ModelDType::Q4`, `ModelQuantizationFormat::GgufQ4`, `PackedWeightFormat::Q4_0`, `dequantize_q4_0`, and the Runtime's storage size, dequantization and packed-format mapping.
- **`formats/gguf`:** reads GGML type 2. `formats/safetensors` treats the dtype as block-quantized.
- **`loaders/gguf`:** keeps a `Q4_0` projection packed with transposed storage declared, like `Q8_0`.
- **Providers:** the optimized CPU Provider and the CUDA Provider run packed `Q4_0` weights. A Provider that declines them gets them dequantized at load, as for every packed format.

## Impact

- **Magnetar crates:** `magnetar-model`, `magnetar-provider-api` and `magnetar-runtime`.
- **Submodules:** `formats/gguf`, `formats/safetensors`, `loaders/gguf`, `providers/cpu` and `providers/cuda`.
- **Breaking:** the new `ModelDType` and `PackedWeightFormat` variants require the submodules' exhaustive matches to cover them, so the five submodules are pinned together.
