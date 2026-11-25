## Context

The Component side asked two things to be checked against real runs on #198.
The first was the RoPE pairing: interleaved for latent attention, half-split
for the non-MLA DeepSeekMoE path. The second was the meaning of an absent
`mixture-of-experts-layers` or `expert-routing`. The tiny bundles exercise the
latent-attention path end to end.

## Decisions

- **Remove the hidden-size rule rather than exempt latent attention.**
  - The rule is not an invariant of the decoders the Runtime serves. Gemma 4
    and Qwen3 break it as well as DeepSeek-V2, because each declares a head
    width that differs from `hidden_size / heads`.
  - The loaders read a declared head width, and derive it only when the
    checkpoint declares none, in which case the rule holds by construction.
  - The projections' expected shapes catch a head width inconsistent with the
    weights.
- **Refuse, not guess, renormalized V2 routing.** The two reference
  implementations disagree, and no published V2 configuration declares it.
  Following the Hugging Face flag matched neither.
- **Interleaved pairing for latent attention** is confirmed by the logit
  match: a wrong pairing changes every rotated pair. The half-split
  DeepSeekMoE path is not covered by these bundles.
- **Fixture size.** The three bundles total about 0.5 MB, so all are kept.
  The generator and reference scripts are checked in beside them, since no
  other repository holds them.
- **Tiny GGUF files written by a script, not by llama.cpp's converter.**
  The script writes the converter's metadata keys and tensor names,
  including the split form's transposed `attn_k_b`. It follows the
  converter's source, which was not run. The wider `wide` bundles exist
  because `Q8_0` and `Q4_0` quantize rows whose length is a multiple of 32.
  At the first bundles' widths (latent 12, 6-wide key heads), the latent
  up-projection would have stayed f32.
- **The real file is `Q8_0`.** The `moe-experts` CUDA Kernel reads `Q8_0`
  experts packed and expands every other format to f32 for each call. So
  the smaller `IQ4_XS` and `IQ4_NL` exports, whose types Magnetar supports,
  would expand about 0.7 GB per expert matrix, in every layer at every
  step. The `Q8_0` file,
  16.7 GB, runs across two GPUs like Qwen1.5-MoE's. The 2024 `Q4_K_M`
  export holds `Q5_0` matrices, which Magnetar does not read.
- **The CUDA comparison turns int8 activations off.** Rounding the
  activations to 8 bits per block moves the logits by more than 1e-3. The
  test refuses to run without `MAGNETAR_CUDA_INT8_ACTIVATIONS=0`.
