## Decisions

### `Q4_0` is `IQ4_NL` with a linear table

Both formats store 32 values in 18 bytes, in the same layout:
- an `f16` scale `d`;
- sixteen bytes whose low nibbles are values 0..15 and high nibbles values 16..31.

They differ only in what a nibble `q` stands for: an entry of `IQ4_NL`'s non-linear table, or `q - 8`. So every `IQ4_NL` path decodes `Q4_0` through a 16-entry table, `Q4_0_VALUES = [-8, ..., 7]`:
- **Runtime:** `dequantize_q4_0` and `dequantize_iq4_nl` share one run decoder.
- **Optimized CPU Provider:** the scalar decode, the AVX2 int8 matmul (`vpshufb` through the table) and the vector run path take the table.
- **CUDA:** `Q4_0` is the IQ4 kernels' format 2. It is stored like `IQ4_NL`, with one `f32` scale per run and the 16 quant bytes in order, and `__vsub4(nibbles, 0x08080808)` turns four nibbles into `q - 8` where IQ4 looks them up.

### `ModelDType::Q4`, as `Q8_0` is `ModelDType::Q8`

The new dtype follows `Q8`'s naming, and `ModelDType::parse` accepts `"q4"` and `"q4_0"`.

## Evidence

- **Dequantization:** `dequantize_q4_0` matches `gguf-py`'s reference bit for bit on 16 random blocks (`k_quant_dequantization_matches_the_gguf_reference_bit_for_bit`).
- **Format:** `formats/gguf` round-trips GGML type 2. Q4_1, which stays unsupported, is refused.
- **Loader:** a `Q4_0` projection stays packed with transposed storage declared, and Model Loading reads its values transposed. The test fails if `Q4_0` is left out of the block-quantized formats.
- **Optimized CPU Provider:** `Q4_0` joins every packed-format test against the Reference CPU, including the odd-run-count tail. Giving the AVX2 path the `IQ4_NL` table makes the test fail.
- **CUDA:** `Q4_0` joins the i-quant conformance test, which covers download, expansion, the int8 decode matmul with an addend and the gated matmul, against the Reference CPU. It runs on the GPU runner of the CUDA PR.
- **Not run:** a real QAT file end to end.
